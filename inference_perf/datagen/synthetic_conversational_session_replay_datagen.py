# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synthetic conversational session replay data generator."""

from __future__ import annotations

import logging
from multiprocessing.managers import SyncManager
from typing import Optional

from inference_perf.config import APIConfig, DataConfig
from inference_perf.datagen.replay_graph_session_datagen import ReplayGraphSessionGeneratorBase
from inference_perf.datagen.synthetic_conversational_replay_graph_builder import generate_replay_graph_sessions
from inference_perf.utils.custom_tokenizer import CustomTokenizer

logger = logging.getLogger(__name__)


class SyntheticConversationalSessionReplayDataGenerator(ReplayGraphSessionGeneratorBase):
    """Direct ReplayGraph-based synthetic conversational session replay generator."""

    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer],
        mp_manager: Optional[SyncManager] = None,
        base_seed: Optional[int] = None,
        num_workers: int = 1,
    ) -> None:
        super().__init__(api_config, config, tokenizer, mp_manager=mp_manager, base_seed=base_seed, num_workers=num_workers)

        if config.synthetic_conversational_session_replay is None:
            raise ValueError(
                "synthetic_conversational_session_replay configuration is required for "
                "SyntheticConversationalSessionReplayDataGenerator"
            )

        self.synthetic_config = config.synthetic_conversational_session_replay
        sessions = generate_replay_graph_sessions(self.synthetic_config)
        self.initialize_sessions(sessions)
        logger.info("Initialized %d synthetic conversational replay sessions", len(self.sessions))
