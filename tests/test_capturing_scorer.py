"""Tests for _CapturingScorer, _link_prompt_to_traces, and predict_fn eval flow.

Covers:
- _CapturingScorer captures single Feedback scores into captured dict
- _CapturingScorer captures guidelines (list of Feedback) scores
- _CapturingScorer passes results through unchanged to caller
- _CapturingScorer handles None row_index gracefully
- _wrap_scorers_for_capture wraps all scorers
- _link_prompt_to_traces links prompt to all traces in a run
- _link_prompt_to_traces is non-fatal on errors
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from mlflow.entities import Feedback

from server.evaluation import (
    _CapturingScorer,
    _wrap_scorers_for_capture,
    _link_prompt_to_traces,
    RowScore,
)


# ---------------------------------------------------------------------------
# _CapturingScorer
# ---------------------------------------------------------------------------

class TestCapturingScorer:

    def _make_inner_scorer(self, return_value):
        """Create a mock inner scorer that returns the given value from run()."""
        inner = MagicMock()
        inner.name = "test_scorer"
        inner.run.return_value = return_value
        return inner

    def test_captures_single_feedback_score(self):
        """A single Feedback result is captured as (value, rationale, None)."""
        captured: dict[int, RowScore] = {}
        feedback = Feedback(name="quality", value=4.5, rationale="good response")
        inner = self._make_inner_scorer(feedback)

        scorer = _CapturingScorer(inner, captured)
        result = scorer.run(inputs={"request": "test", "_row_index": 0}, outputs={"response": "hi"})

        assert result is feedback  # passthrough
        assert 0 in captured
        assert captured[0] == (4.5, "good response", None)

    def test_captures_pass_fail_string_normalized(self):
        """String values like 'yes'/'no' are normalized to 1.0/0.0."""
        captured: dict[int, RowScore] = {}
        feedback = Feedback(name="safety", value="yes", rationale="safe")
        inner = self._make_inner_scorer(feedback)

        scorer = _CapturingScorer(inner, captured)
        scorer.run(inputs={"request": "test", "_row_index": 1}, outputs={"response": "hi"})

        assert captured[1][0] == 1.0

    def test_captures_guidelines_list(self):
        """A list of Feedback (guidelines scorer) is captured as pass/total summary."""
        captured: dict[int, RowScore] = {}
        feedbacks = [
            Feedback(name="rule_a", value="yes", rationale="passes rule a"),
            Feedback(name="rule_b", value="no", rationale="fails rule b"),
            Feedback(name="rule_c", value="yes", rationale="passes rule c"),
        ]
        inner = self._make_inner_scorer(feedbacks)

        scorer = _CapturingScorer(inner, captured)
        result = scorer.run(inputs={"request": "test", "_row_index": 2}, outputs={"response": "hi"})

        assert result is feedbacks
        assert 2 in captured
        score, rationale, details = captured[2]
        assert score == "2/3"  # 2 passes out of 3
        assert rationale is None
        assert len(details) == 3

    def test_no_capture_when_row_index_missing(self):
        """If _row_index is not in inputs, nothing is captured."""
        captured: dict[int, RowScore] = {}
        feedback = Feedback(name="quality", value=3.0, rationale="ok")
        inner = self._make_inner_scorer(feedback)

        scorer = _CapturingScorer(inner, captured)
        scorer.run(inputs={"request": "test"}, outputs={"response": "hi"})

        assert len(captured) == 0

    def test_passthrough_result_unchanged(self):
        """The original scorer result is returned unchanged."""
        captured: dict[int, RowScore] = {}
        feedback = Feedback(name="quality", value=5.0, rationale="great")
        inner = self._make_inner_scorer(feedback)

        scorer = _CapturingScorer(inner, captured)
        result = scorer.run(inputs={"request": "test", "_row_index": 0}, outputs={"response": "hi"})

        assert result is feedback
        assert result.value == 5.0

    def test_delegates_to_inner_run(self):
        """The inner scorer's run() is called with the same kwargs."""
        captured: dict[int, RowScore] = {}
        inner = self._make_inner_scorer(Feedback(name="q", value=1.0))

        scorer = _CapturingScorer(inner, captured)
        scorer.run(
            inputs={"request": "x", "_row_index": 0},
            outputs={"response": "y"},
            expectations={"expected": "z"},
        )

        inner.run.assert_called_once_with(
            inputs={"request": "x", "_row_index": 0},
            outputs={"response": "y"},
            expectations={"expected": "z"},
            trace=None,
            session=None,
        )


# ---------------------------------------------------------------------------
# _wrap_scorers_for_capture
# ---------------------------------------------------------------------------

class TestWrapScorersForCapture:

    def test_wraps_all_scorers(self):
        captured = {}
        inner1 = MagicMock()
        inner1.name = "scorer1"
        inner2 = MagicMock()
        inner2.name = "scorer2"

        wrapped = _wrap_scorers_for_capture([inner1, inner2], captured)

        assert len(wrapped) == 2
        assert all(isinstance(s, _CapturingScorer) for s in wrapped)

    def test_wrapped_names_match_inner(self):
        captured = {}
        inner = MagicMock()
        inner.name = "my_scorer"

        wrapped = _wrap_scorers_for_capture([inner], captured)

        assert wrapped[0].name == "my_scorer"


# ---------------------------------------------------------------------------
# _link_prompt_to_traces
# ---------------------------------------------------------------------------

class TestLinkPromptToTraces:

    def test_links_prompt_to_all_traces(self):
        """Calls link_prompt_versions_to_trace for each trace in the run."""
        mock_trace1 = MagicMock()
        mock_trace1.info.request_id = "trace-1"
        mock_trace2 = MagicMock()
        mock_trace2.info.request_id = "trace-2"

        mock_client = MagicMock()
        mock_pv = MagicMock()
        mock_client.get_prompt_version.return_value = mock_pv

        with (
            patch("server.evaluation.get_mlflow_client", return_value=mock_client),
            patch("server.evaluation.mlflow") as mock_mlflow,
        ):
            mock_mlflow.search_traces.return_value = [mock_trace1, mock_trace2]
            _link_prompt_to_traces("run-123", "catalog.schema.prompt", "1")

        assert mock_client.link_prompt_versions_to_trace.call_count == 2
        mock_client.link_prompt_versions_to_trace.assert_any_call(
            prompt_versions=[mock_pv], trace_id="trace-1",
        )
        mock_client.link_prompt_versions_to_trace.assert_any_call(
            prompt_versions=[mock_pv], trace_id="trace-2",
        )

    def test_non_fatal_on_search_traces_failure(self):
        """If search_traces raises, the function logs a warning but does not raise."""
        mock_client = MagicMock()
        mock_client.get_prompt_version.return_value = MagicMock()

        with (
            patch("server.evaluation.get_mlflow_client", return_value=mock_client),
            patch("server.evaluation.mlflow") as mock_mlflow,
        ):
            mock_mlflow.search_traces.side_effect = RuntimeError("search failed")
            # Should not raise
            _link_prompt_to_traces("run-123", "catalog.schema.prompt", "1")

        mock_client.link_prompt_versions_to_trace.assert_not_called()

    def test_non_fatal_on_individual_link_failure(self):
        """If one trace fails to link, the others still get linked."""
        mock_trace1 = MagicMock()
        mock_trace1.info.request_id = "trace-1"
        mock_trace2 = MagicMock()
        mock_trace2.info.request_id = "trace-2"

        mock_client = MagicMock()
        mock_pv = MagicMock()
        mock_client.get_prompt_version.return_value = mock_pv
        mock_client.link_prompt_versions_to_trace.side_effect = [
            RuntimeError("link failed"), None,
        ]

        with (
            patch("server.evaluation.get_mlflow_client", return_value=mock_client),
            patch("server.evaluation.mlflow") as mock_mlflow,
        ):
            mock_mlflow.search_traces.return_value = [mock_trace1, mock_trace2]
            _link_prompt_to_traces("run-123", "catalog.schema.prompt", "1")

        # Both traces attempted despite first failing
        assert mock_client.link_prompt_versions_to_trace.call_count == 2

    def test_empty_traces_is_noop(self):
        """No error when the run has no traces."""
        mock_client = MagicMock()
        mock_client.get_prompt_version.return_value = MagicMock()

        with (
            patch("server.evaluation.get_mlflow_client", return_value=mock_client),
            patch("server.evaluation.mlflow") as mock_mlflow,
        ):
            mock_mlflow.search_traces.return_value = []
            _link_prompt_to_traces("run-123", "catalog.schema.prompt", "1")

        mock_client.link_prompt_versions_to_trace.assert_not_called()


