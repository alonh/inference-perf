# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import re
from datetime import datetime
from typing import Any, List, Optional

import yaml
from pydantic import BaseModel, model_validator

from inference_perf.circuit_breaker import CircuitBreakerConfig
from inference_perf.config.apis import APIConfig
from inference_perf.config.client.filestorage import StorageConfig
from inference_perf.config.client.modelserver import ModelServerClientConfig
from inference_perf.config.datagen import DataConfig, DataGenType
from inference_perf.config.loadgen import (
    ConcurrentLoadStage,
    LoadConfig,
    LoadType,
    StandardLoadStage,
    TraceSessionReplayLoadStage,
)
from inference_perf.config.metrics import MetricsClientConfig
from inference_perf.config.reportgen import ReportConfig
from inference_perf.config.utils import CustomTokenizerConfig


class Config(BaseModel):
    api: APIConfig = APIConfig()
    data: DataConfig = DataConfig()
    load: LoadConfig = LoadConfig()
    metrics: Optional[MetricsClientConfig] = None
    report: ReportConfig = ReportConfig()
    storage: Optional[StorageConfig] = StorageConfig()
    server: Optional[ModelServerClientConfig] = None
    tokenizer: Optional[CustomTokenizerConfig] = None
    circuit_breakers: Optional[List[CircuitBreakerConfig]] = None

    @model_validator(mode="after")
    def validate_trace_replay_load_type(self) -> "Config":
        """Validate that trace replay data types use trace_session_replay load type."""
        if self.data.type in (DataGenType.OTelTraceReplay, DataGenType.WekaTraceReplay):
            if self.load.type != LoadType.TRACE_SESSION_REPLAY:
                raise ValueError(
                    f"data.type '{self.data.type.value}' requires load.type 'trace_session_replay', "
                    f"but got '{self.load.type.value}'. Trace replay with dependencies requires "
                    f"session-based load dispatch to properly handle event dependencies and timing."
                )
        return self


# Names whose value is credential material and must never be persisted. The resolved config
# reaches durable storage twice — reportgen.base.generate_config_report writes it to
# <run>/config.yaml, and read_config logs it in full at startup — so a key supplied via
# `--api.headers '{"MY_API_KEY":"..."}'` or `server.api_key` would otherwise sit in plaintext
# in every run directory and every captured log.
SECRET_NAME_PATTERN = re.compile(r"api[-_]?key|token|secret|password|passwd|authorization|credential", re.IGNORECASE)

REDACTED_VALUE = "***REDACTED***"

# Every value under `headers` is treated as credential material regardless of its name: the
# header names a gateway wants for auth are arbitrary (RITS_API_KEY, x-tenant-sig, ...), and
# guessing which ones are sensitive fails open. The header NAMES survive redaction, so the
# saved config still shows which headers the run sent — only the values are withheld.
ALL_VALUES_SECRET_KEYS = frozenset({"headers"})


def redact_secrets(value: Any, all_values_secret: bool = False) -> Any:
    """A copy of `value` with every credential-bearing entry replaced by a placeholder.

    Recurses through mappings and sequences and leaves anything else referenced as-is,
    unwrapped: the startup log dumps a config whose `load.stages` already hold pydantic stage
    objects, and copying those would change how yaml renders them. `None` is passed through
    rather than masked, so `api_key: null` still reads as "unset" instead of looking like a
    secret that was set.

    Never mutates its argument — `read_config` hands the same dict to `Config(**merged_cfg)`
    immediately afterwards.
    """
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for k, v in value.items():
            named_secret = isinstance(k, str) and SECRET_NAME_PATTERN.search(k) is not None
            if (all_values_secret or named_secret) and v is not None and not isinstance(v, (dict, list, tuple)):
                out[k] = REDACTED_VALUE
            else:
                nested_all_secret = isinstance(k, str) and k.lower() in ALL_VALUES_SECRET_KEYS
                out[k] = redact_secrets(v, all_values_secret or nested_all_secret)
        return out
    if isinstance(value, list):
        return [redact_secrets(v, all_values_secret) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(v, all_values_secret) for v in value)
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def read_config(config_file: Optional[str] = None, cli_overrides: Optional[dict[str, Any]] = None) -> Config:
    logger = logging.getLogger(__name__)
    cfg: dict[str, Any] = {}
    if config_file:
        logger.info("Using configuration from: %s", config_file)
        with open(config_file, "r") as stream:
            cfg = yaml.safe_load(stream) or {}

    default_cfg = Config().model_dump(mode="json")
    merged_cfg = deep_merge(default_cfg, cfg)

    if cli_overrides:
        merged_cfg = deep_merge(merged_cfg, cli_overrides)

    # Handle timestamp substitution in storage paths
    if "storage" in merged_cfg and merged_cfg["storage"]:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for storage_type in ["local_storage", "google_cloud_storage", "simple_storage_service"]:
            if (
                storage_type in merged_cfg["storage"]
                and merged_cfg["storage"][storage_type]
                and "path" in merged_cfg["storage"][storage_type]
            ):
                path = merged_cfg["storage"][storage_type]["path"]
                if path and "{timestamp}" in path:
                    merged_cfg["storage"][storage_type]["path"] = path.replace("{timestamp}", timestamp)

    # Handle stage type conversion based on load type
    if "load" in merged_cfg and "stages" in merged_cfg["load"] and merged_cfg["load"]["stages"]:
        load_type = merged_cfg["load"].get("type", "constant")
        stages = merged_cfg["load"]["stages"]

        if load_type == "concurrent":
            # Convert to ConcurrentLoadStage objects
            concurrent_stages = []
            for stage in stages:
                concurrent_stages.append(ConcurrentLoadStage(**stage))
            merged_cfg["load"]["stages"] = concurrent_stages
        elif load_type == "trace_session_replay":
            # Convert to TraceSessionReplayLoadStage objects
            trace_session_stages = []
            for stage in stages:
                trace_session_stages.append(TraceSessionReplayLoadStage(**stage))
            merged_cfg["load"]["stages"] = trace_session_stages
        else:
            # Convert to StandardLoadStage objects for constant/poisson/trace_replay
            standard_stages = []
            for stage in stages:
                standard_stages.append(StandardLoadStage(**stage))
            merged_cfg["load"]["stages"] = standard_stages

    # Redacted for the log only; Config() below receives the unmodified merged_cfg.
    logger.info(
        "Benchmarking with the following config:\n\n%s\n",
        yaml.dump(redact_secrets(merged_cfg), sort_keys=False, default_flow_style=False),
    )
    return Config(**merged_cfg)
