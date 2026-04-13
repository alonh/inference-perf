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

"""Builder utilities for synthetic conversational ReplayGraph sessions.

Generates deterministic ReplayGraph-backed sessions from configurable
distributions. Each session represents one linear multi-turn conversation
with shared system-prompt prefix characteristics suitable for replay and
prefix-caching studies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

import numpy as np

from inference_perf.datagen.replay_graph_session_datagen import ReplayGraphSession
from inference_perf.datagen.replay_graph_types import GraphCall, GraphEvent, InputSegment, ReplayGraph, ReplayMessage

if TYPE_CHECKING:
    from inference_perf.config import SyntheticConversationalSessionReplayConfig, SyntheticTraceDistribution

logger = logging.getLogger(__name__)


def sample_distribution(
    dist_type: str, dist_min: int, dist_max: int, mean: float, std_dev: float, count: int, rng: np.random.RandomState
) -> Any:
    """Sample from a distribution, returning integer values clipped to [min, max]."""
    if dist_type == "fixed":
        values = np.full(count, mean)
    elif dist_type == "uniform":
        values = rng.uniform(dist_min, dist_max, size=count)
    elif dist_type == "lognormal":
        target_mean = mean - dist_min
        if target_mean <= 0:
            values = np.full(count, dist_min)
        else:
            sigma = np.log(1 + (std_dev / max(target_mean, 1)) ** 2) ** 0.5
            mu = np.log(target_mean) - sigma**2 / 2
            values = rng.lognormal(mean=mu, sigma=sigma, size=count) + dist_min
    elif dist_type == "normal":
        values = rng.normal(loc=mean, scale=std_dev, size=count)
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

    return np.clip(np.round(values), dist_min, dist_max).astype(int)


def _sample_from_config(
    config: "SyntheticTraceDistribution | None", count: int, rng: np.random.RandomState, default_value: int = 10
) -> Any:
    """Sample from a SyntheticTraceDistribution config, or return fixed default."""
    if config is None:
        return np.full(count, default_value, dtype=int)
    return sample_distribution(config.type, config.min, config.max, config.mean, config.std_dev, count, rng)


@dataclass
class TurnBlueprint:
    input_tokens: int
    output_tokens: int
    wait_ms: int
    user_text: str = ""
    assistant_text: str = ""


@dataclass
class ConversationBlueprint:
    conversation_id: str
    system_prompt: str
    turns: List[TurnBlueprint] = field(default_factory=list)


def generate_random_text(rng: np.random.RandomState, num_tokens: int, vocab_size: int = 30000) -> str:
    """Generate random text approximating a target token count."""
    if num_tokens <= 0:
        return ""
    words_needed = max(1, int(num_tokens * 0.8))
    word_ids = rng.randint(0, vocab_size, size=words_needed)
    return " ".join(f"w{wid}" for wid in word_ids)


def _estimate_token_count_from_messages(messages: List[Dict[str, str]]) -> int:
    return sum(len(message["content"]) // 4 for message in messages)


def build_conversation_blueprints(config: "SyntheticConversationalSessionReplayConfig") -> List[ConversationBlueprint]:
    """Build deterministic conversation blueprints from configuration."""
    rng = np.random.RandomState(config.seed)

    shared_prefix_text = generate_random_text(rng, config.shared_system_prompt_len)
    turn_counts = _sample_from_config(config.turns_per_conversation, config.num_conversations, rng, default_value=4)
    dynamic_lens = _sample_from_config(config.dynamic_system_prompt_len, config.num_conversations, rng, default_value=2000)

    total_turns = int(turn_counts.sum())
    all_input_tokens = _sample_from_config(config.input_tokens_per_turn, total_turns, rng, default_value=1400)
    all_output_tokens = _sample_from_config(config.output_tokens_per_turn, total_turns, rng, default_value=526)
    all_wait_ms = _sample_from_config(config.inter_turn_wait_ms, total_turns, rng, default_value=100)

    blueprints: List[ConversationBlueprint] = []
    turn_cursor = 0

    for i in range(config.num_conversations):
        n_turns = int(turn_counts[i])
        dynamic_suffix_text = generate_random_text(rng, int(dynamic_lens[i]))
        system_prompt = shared_prefix_text + "\n\n" + dynamic_suffix_text

        turns: List[TurnBlueprint] = []
        for turn_idx in range(n_turns):
            input_tokens = int(all_input_tokens[turn_cursor])
            output_tokens = int(all_output_tokens[turn_cursor])
            wait_ms = 0 if turn_idx == 0 else int(all_wait_ms[turn_cursor])
            turns.append(
                TurnBlueprint(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    wait_ms=wait_ms,
                    user_text=generate_random_text(rng, input_tokens),
                    assistant_text=generate_random_text(rng, output_tokens),
                )
            )
            turn_cursor += 1

        blueprints.append(ConversationBlueprint(conversation_id=f"conv_{i:04d}", system_prompt=system_prompt, turns=turns))

    logger.info(
        "Built %d synthetic conversation blueprints (seed=%d, total_turns=%d)",
        config.num_conversations,
        config.seed,
        total_turns,
    )
    return blueprints


def build_replay_graph_from_blueprint(
    blueprint: ConversationBlueprint,
    model_name: str,
) -> ReplayGraph:
    """Build a ReplayGraph directly from a linear conversation blueprint."""
    events: Dict[str, GraphEvent] = {}
    root_event_ids: List[str] = []

    conversation_messages: List[Dict[str, str]] = [{"role": "system", "content": blueprint.system_prompt}]

    for turn_idx, turn in enumerate(blueprint.turns):
        event_id = f"event_{turn_idx:03d}_turn_{turn_idx}"
        call_id = f"turn_{turn_idx}"

        current_messages = [dict(msg) for msg in conversation_messages]
        current_messages.append({"role": "user", "content": turn.user_text})

        total_input_tokens = _estimate_token_count_from_messages(current_messages)
        predecessor_event_ids: List[str] = []
        input_segments: List[InputSegment] = []

        if turn_idx == 0:
            input_segments.append(
                InputSegment(
                    type="unique",
                    message_count=len(current_messages),
                    token_count=total_input_tokens,
                )
            )
            root_event_ids.append(event_id)
        else:
            prev_event_id = f"event_{turn_idx - 1:03d}_turn_{turn_idx - 1}"
            predecessor_event_ids = [prev_event_id]

            shared_message_count = len(current_messages) - 2
            shared_messages = current_messages[:shared_message_count]
            output_message = current_messages[shared_message_count : shared_message_count + 1]
            unique_messages = current_messages[shared_message_count + 1 :]

            if shared_messages:
                input_segments.append(
                    InputSegment(
                        type="shared",
                        message_count=len(shared_messages),
                        token_count=_estimate_token_count_from_messages(shared_messages),
                        source_event_id=prev_event_id,
                    )
                )
            if output_message:
                input_segments.append(
                    InputSegment(
                        type="output",
                        message_count=1,
                        token_count=_estimate_token_count_from_messages(output_message),
                        source_event_id=prev_event_id,
                    )
                )
            if unique_messages:
                input_segments.append(
                    InputSegment(
                        type="unique",
                        message_count=len(unique_messages),
                        token_count=_estimate_token_count_from_messages(unique_messages),
                    )
                )

        graph_call = GraphCall(
            call_id=call_id,
            model=model_name,
            messages=[ReplayMessage(role=msg["role"], text=msg["content"]) for msg in current_messages],
            expected_output=turn.assistant_text,
            input_segments=input_segments,
            total_input_tokens=total_input_tokens,
            expected_output_tokens=turn.output_tokens,
            temperature=None,
            max_tokens_recorded=turn.output_tokens,
        )

        events[event_id] = GraphEvent(
            event_id=event_id,
            call=graph_call,
            predecessor_event_ids=predecessor_event_ids,
            predecessor_dependency_types={pred_id: "causal" for pred_id in predecessor_event_ids},
            wait_ms=turn.wait_ms,
            t_start_ms=0,
            t_end_ms=0,
        )

        conversation_messages.append({"role": "user", "content": turn.user_text})
        conversation_messages.append({"role": "assistant", "content": turn.assistant_text})

    return ReplayGraph(
        events=events,
        root_event_ids=root_event_ids,
        source_file=f"synthetic://{blueprint.conversation_id}",
    )


def generate_replay_graph_sessions(
    config: "SyntheticConversationalSessionReplayConfig",
) -> List[ReplayGraphSession]:
    """Generate ReplayGraph sessions directly from synthetic conversation configuration."""
    blueprints = build_conversation_blueprints(config)
    model_name = config.model_name or "synthetic-model"

    sessions: List[ReplayGraphSession] = []
    for session_index, blueprint in enumerate(blueprints):
        graph = build_replay_graph_from_blueprint(blueprint, model_name=model_name)
        sessions.append(
            ReplayGraphSession(
                session_id=blueprint.conversation_id,
                source_id=f"synthetic://{blueprint.conversation_id}",
                session_index=session_index,
                graph=graph,
            )
        )

    logger.info("Generated %d synthetic conversational replay sessions", len(sessions))
    return sessions
