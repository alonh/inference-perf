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
"""Credentials must not reach durable storage.

The resolved config leaves the process twice — logged in full by ``read_config`` and written
to ``<run>/config.yaml`` by ``ReportGenerator.generate_config_report`` — so the redaction has
to hold at both. The last two tests go through those real call paths rather than calling
``redact_secrets`` directly, because a correct helper that a write site forgets to call is
exactly the failure this guards against.
"""

import logging

import yaml

from inference_perf.config import REDACTED_VALUE, Config, read_config, redact_secrets

SECRET = "s3cr3t-key-value"


def test_named_secret_fields_are_masked() -> None:
    out = redact_secrets({"api_key": SECRET, "auth_token": SECRET, "password": SECRET, "model_name": "qwen"})
    assert out == {
        "api_key": REDACTED_VALUE,
        "auth_token": REDACTED_VALUE,
        "password": REDACTED_VALUE,
        "model_name": "qwen",
    }


def test_every_header_value_is_masked_but_names_survive() -> None:
    """Auth header names are gateway-specific, so name-matching alone fails open."""
    out = redact_secrets({"api": {"headers": {"RITS_API_KEY": SECRET, "x-tenant-sig": SECRET}}})
    assert out["api"]["headers"] == {"RITS_API_KEY": REDACTED_VALUE, "x-tenant-sig": REDACTED_VALUE}


def test_none_is_not_masked() -> None:
    """`api_key: null` should keep reading as unset, not as a withheld secret."""
    assert redact_secrets({"api_key": None}) == {"api_key": None}


def test_input_is_not_mutated() -> None:
    """read_config passes the same dict to Config() straight after logging it."""
    original = {"api": {"headers": {"RITS_API_KEY": SECRET}}}
    redact_secrets(original)
    assert original["api"]["headers"]["RITS_API_KEY"] == SECRET


def test_nested_and_non_mapping_values_are_traversed() -> None:
    out = redact_secrets({"stages": [{"token": SECRET}, {"rate": 5}], "count": 3, "flag": True})
    assert out == {"stages": [{"token": REDACTED_VALUE}, {"rate": 5}], "count": 3, "flag": True}


def test_unknown_objects_pass_through_unwrapped() -> None:
    """load.stages holds pydantic objects by the time read_config logs; copying them would
    change how yaml renders them."""
    sentinel = object()
    assert redact_secrets({"stages": [sentinel]})["stages"][0] is sentinel


def _replay_config(path: str) -> str:
    cfg = {
        "api": {"type": "chat", "headers": {"RITS_API_KEY": SECRET}},
        "data": {"type": "shared_prefix"},
        "load": {"type": "constant", "stages": [{"rate": 1, "duration": 1}]},
        "server": {"type": "vllm", "model_name": "m", "base_url": "http://x", "api_key": SECRET},
    }
    with open(path, "w") as f:
        yaml.dump(cfg, f)
    return path


def test_startup_log_does_not_contain_the_secret(tmp_path, caplog) -> None:
    path = _replay_config(str(tmp_path / "config.yml"))
    with caplog.at_level(logging.INFO, logger="inference_perf.config.config"):
        config = read_config(path, {})
    assert SECRET not in caplog.text
    assert REDACTED_VALUE in caplog.text
    # Redacted for the log only — the running config must still hold the real key.
    assert config.api.headers is not None
    assert config.api.headers["RITS_API_KEY"] == SECRET


def test_saved_config_report_does_not_contain_the_secret(tmp_path) -> None:
    from inference_perf.reportgen.base import ReportGenerator

    config = read_config(_replay_config(str(tmp_path / "config.yml")), {})
    report = ReportGenerator.generate_config_report(_StubReportGen(config))  # type: ignore[arg-type]
    dumped = yaml.dump(report.contents)
    assert SECRET not in dumped
    assert dumped.count(REDACTED_VALUE) >= 2  # the header value and server.api_key
    assert "RITS_API_KEY" in dumped  # the name is kept: which headers were sent is provenance


class _StubReportGen:
    """generate_config_report reads only self.config; instantiating the real ReportGenerator
    would require a metrics client and a datagen."""

    def __init__(self, config: Config) -> None:
        self.config = config
