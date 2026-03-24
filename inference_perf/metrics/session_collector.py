# Copyright 2025 The Kubernetes Authors.
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

"""Session-level metrics collector for agentic workflows."""

from typing import List
from inference_perf.apis import SessionLifecycleMetric


class SessionMetricsCollector:
    """Collects session-level lifecycle metrics.

    Used to collect metrics for multi-turn agentic workflows where multiple
    requests form a logical session. The collector stores metrics generated
    by LoadGenerator and provides them to ReportGenerator for analysis.

    This decouples LoadGenerator and ReportGenerator, allowing them to
    communicate through the collector without direct dependencies.
    """

    def __init__(self) -> None:
        """Initialize the session metrics collector."""
        self._session_metrics: List[SessionLifecycleMetric] = []

    def record_metric(self, metric: SessionLifecycleMetric) -> None:
        """Record a completed session's lifecycle metric.

        Args:
            metric: The session lifecycle metric to record
        """
        self._session_metrics.append(metric)

    def get_metrics(self) -> List[SessionLifecycleMetric]:
        """Return all recorded session lifecycle metrics.

        Returns:
            A copy of all recorded session metrics
        """
        return self._session_metrics

    def clear(self) -> None:
        """Clear all recorded metrics.

        Useful for resetting state between test runs or stages.
        """
        self._session_metrics.clear()


# Made with Bob
