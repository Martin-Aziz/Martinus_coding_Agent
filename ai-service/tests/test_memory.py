"""Tests for SQLiteMemoryStore (app/memory.py).

Covers: insert/retrieve basics, cross-session filtering, semantic
deduplication, composite scoring formula, importance sub-score, recency
decay, WAL journal mode, and score breakdown fields.
"""
import math
import sqlite3
from pathlib import Path

import pytest

from app.memory import (
    ACCESS_COUNT_SOFTCAP,
    DEDUP_SIMILARITY_THRESHOLD,
    IMPORTANCE_ACCESS_W,
    IMPORTANCE_OUTCOME_W,
    IMPORTANCE_PHASE_W,
    OUTCOME_PRIORS,
    PHASE_AFFINITY_BONUS,
    PHASE_AFFINITY_NEUTRAL,
    WEIGHT_IMPORTANCE,
    WEIGHT_LEXICAL,
    WEIGHT_RECENCY,
    WEIGHT_SEMANTIC,
    SQLiteMemoryStore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store(tmp_path: Path) -> SQLiteMemoryStore:
    """Return a fresh SQLiteMemoryStore backed by a temp database."""
    return SQLiteMemoryStore(str(tmp_path / "memory.sqlite3"))


# ---------------------------------------------------------------------------
# Basic insert / retrieve
# ---------------------------------------------------------------------------

class TestInsertRetrieve:
    def test_insert_returns_positive_id(self, tmp_path: Path) -> None:
        """Insert returns a positive integer row id."""
        store = _make_store(tmp_path)
        row_id = store.insert(
            session_id="s1",
            content="Hello world test content",
            tags=["tag-a"],
            workspace_path="/repo",
            source_phase="Planning",
            outcome_label="success",
        )
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_retrieve_returns_inserted_content(self, tmp_path: Path) -> None:
        """Retrieve surfaces a previously inserted memory whose content matches the query."""
        store = _make_store(tmp_path)
        store.insert(
            session_id="s1",
            content="Use pytest fixtures for database isolation in tests",
            tags=["testing"],
            workspace_path="/repo",
            source_phase="Testing",
            outcome_label="success",
        )
        results = store.retrieve(
            session_id="s1",
            query="pytest fixtures database isolation",
            top_k=5,
            workspace_path="/repo",
        )
        assert len(results) >= 1
        assert any("pytest" in item["content"] for item in results)

    def test_retrieve_empty_store_returns_empty_list(self, tmp_path: Path) -> None:
        """Retrieve on an empty store returns an empty list without raising."""
        store = _make_store(tmp_path)
        results = store.retrieve(
            session_id="s1",
            query="anything",
            top_k=5,
            workspace_path="/repo",
        )
        assert results == []

    def test_retrieve_honours_top_k(self, tmp_path: Path) -> None:
        """Retrieve returns at most top_k results even when more entries exist."""
        store = _make_store(tmp_path)
        for i in range(10):
            store.insert(
                session_id="s1",
                content=f"Memory entry number {i} about architecture and design patterns",
                tags=["arch"],
                workspace_path="/repo",
                source_phase="Architecture",
                outcome_label="success",
            )
        results = store.retrieve(
            session_id="s1",
            query="architecture design patterns",
            top_k=3,
            workspace_path="/repo",
        )
        assert len(results) <= 3

    def test_retrieve_scores_are_descending(self, tmp_path: Path) -> None:
        """Results returned by retrieve are sorted by descending composite score."""
        store = _make_store(tmp_path)
        store.insert(
            session_id="s1",
            content="Architecture uses hybrid retrieval with hash-vector cosine reranking",
            tags=["arch"],
            workspace_path="/repo",
            source_phase="Architecture",
            outcome_label="success",
        )
        store.insert(
            session_id="s1",
            content="Verification runs go test and npm typecheck commands",
            tags=["verify"],
            workspace_path="/repo",
            source_phase="Verification",
            outcome_label="success",
        )
        results = store.retrieve(
            session_id="s1",
            query="hash-vector cosine architecture retrieval",
            top_k=5,
            workspace_path="/repo",
        )
        scores = [float(item["score"]) for item in results]
        assert scores == sorted(scores, reverse=True)

    def test_score_breakdown_fields_present(self, tmp_path: Path) -> None:
        """Each result exposes a score_breakdown dict with all expected sub-score keys."""
        store = _make_store(tmp_path)
        store.insert(
            session_id="s1",
            content="Composite scoring combines semantic lexical recency importance",
            tags=["scoring"],
            workspace_path="/repo",
            source_phase="Planning",
            outcome_label="success",
        )
        results = store.retrieve(
            session_id="s1",
            query="composite scoring semantic lexical",
            top_k=1,
            workspace_path="/repo",
        )
        assert len(results) == 1
        breakdown = results[0]["score_breakdown"]
        for key in ("hash_vector_cosine", "lexical", "recency", "importance",
                    "outcome_score", "access_score", "phase_score",
                    "cross_session_penalty"):
            assert key in breakdown, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Cross-session filtering
# ---------------------------------------------------------------------------

class TestCrossSession:
    def test_cross_session_retrieval_finds_other_session(self, tmp_path: Path) -> None:
        """include_cross_session=True returns memories from a different session on the same workspace."""
        store = _make_store(tmp_path)
        store.insert(
            session_id="session-old",
            content="Verification command: go test ./... and npm run typecheck",
            tags=["verify"],
            workspace_path="/repo",
            source_phase="Verification",
            outcome_label="success",
        )
        results = store.retrieve(
            session_id="session-new",
            query="verification command go test",
            top_k=5,
            workspace_path="/repo",
            include_cross_session=True,
        )
        assert len(results) == 1
        assert results[0]["session_id"] == "session-old"

    def test_cross_session_disabled_excludes_other_session(self, tmp_path: Path) -> None:
        """include_cross_session=False hides memories inserted under a different session id."""
        store = _make_store(tmp_path)
        store.insert(
            session_id="session-old",
            content="Verification command: go test ./... and npm run typecheck",
            tags=["verify"],
            workspace_path="/repo",
            source_phase="Verification",
            outcome_label="success",
        )
        results = store.retrieve(
            session_id="session-new",
            query="verification command go test",
            top_k=5,
            workspace_path="/repo",
            include_cross_session=False,
        )
        assert len(results) == 0

    def test_cross_session_penalty_applied(self, tmp_path: Path) -> None:
        """Cross-session results carry a cross_session_penalty less than 1.0 in score_breakdown."""
        store = _make_store(tmp_path)
        store.insert(
            session_id="session-other",
            content="Architecture uses hybrid retrieval cosine reranking recency decay",
            tags=["arch"],
            workspace_path="/repo",
            source_phase="Architecture",
            outcome_label="success",
        )
        results = store.retrieve(
            session_id="session-new",
            query="hybrid retrieval cosine reranking",
            top_k=5,
            workspace_path="/repo",
            include_cross_session=True,
        )
        assert len(results) == 1
        penalty = results[0]["score_breakdown"]["cross_session_penalty"]
        assert penalty < 1.0

    def test_same_session_penalty_is_one(self, tmp_path: Path) -> None:
        """Memories retrieved in the same session have cross_session_penalty == 1.0."""
        store = _make_store(tmp_path)
        store.insert(
            session_id="s1",
            content="Planning phase creates detailed implementation blueprints",
            tags=["planning"],
            workspace_path="/repo",
            source_phase="Planning",
            outcome_label="success",
        )
        results = store.retrieve(
            session_id="s1",
            query="planning implementation blueprint",
            top_k=5,
            workspace_path="/repo",
        )
        assert len(results) == 1
        assert results[0]["score_breakdown"]["cross_session_penalty"] == 1.0


# ---------------------------------------------------------------------------
# Semantic deduplication
# ---------------------------------------------------------------------------

class TestDeduplication:
    def test_near_duplicate_updates_existing_row(self, tmp_path: Path) -> None:
        """Inserting near-identical content updates the existing row rather than creating a new one."""
        store = _make_store(tmp_path)
        first_id = store.insert(
            session_id="s1",
            content="The quick brown fox jumps over the lazy dog",
            tags=["dedup"],
            workspace_path="/repo",
            source_phase="Testing",
            outcome_label="success",
        )
        second_id = store.insert(
            session_id="s1",
            content="The quick brown fox jumps over the lazy dog",
            tags=["dedup"],
            workspace_path="/repo",
            source_phase="Testing",
            outcome_label="success",
        )
        assert first_id == second_id

    def test_distinct_content_creates_new_row(self, tmp_path: Path) -> None:
        """Inserting clearly distinct content always creates a separate row."""
        store = _make_store(tmp_path)
        id_a = store.insert(
            session_id="s1",
            content="Python asyncio event loop task scheduling",
            tags=["python"],
            workspace_path="/repo",
            source_phase="Architecture",
            outcome_label="success",
        )
        id_b = store.insert(
            session_id="s1",
            content="Kubernetes horizontal pod autoscaler memory limits",
            tags=["devops"],
            workspace_path="/repo",
            source_phase="Deployment",
            outcome_label="success",
        )
        assert id_a != id_b

    def test_retrieve_after_dedup_returns_one_entry(self, tmp_path: Path) -> None:
        """After deduplication, retrieve returns only one entry for near-duplicate inserts."""
        store = _make_store(tmp_path)
        content = "Architecture uses hybrid retrieval hash vector cosine reranking recency"
        store.insert(
            session_id="s1",
            content=content,
            tags=["arch"],
            workspace_path="/repo",
            source_phase="Architecture",
            outcome_label="success",
        )
        store.insert(
            session_id="s1",
            content=content,
            tags=["arch"],
            workspace_path="/repo",
            source_phase="Architecture",
            outcome_label="success",
        )
        results = store.retrieve(
            session_id="s1",
            query="hybrid retrieval hash vector cosine",
            top_k=10,
            workspace_path="/repo",
        )
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Outcome label normalisation
# ---------------------------------------------------------------------------

class TestOutcomeNormalisation:
    def test_known_outcome_labels_are_preserved(self, tmp_path: Path) -> None:
        """Known outcome labels (success, partial, failure, unknown) are stored unchanged."""
        store = _make_store(tmp_path)
        for label in ("success", "partial", "failure", "unknown"):
            row_id = store.insert(
                session_id="s1",
                content=f"Memory with outcome {label} label attached for testing",
                tags=[label],
                workspace_path="/repo",
                source_phase="Testing",
                outcome_label=label,
            )
            results = store.retrieve(
                session_id="s1",
                query=f"outcome {label} label",
                top_k=1,
                workspace_path="/repo",
            )
            assert any(r["outcome_label"] == label for r in results)

    def test_unknown_outcome_label_normalised(self, tmp_path: Path) -> None:
        """An unrecognised outcome label is normalised to 'unknown'."""
        store = _make_store(tmp_path)
        store.insert(
            session_id="s1",
            content="Memory with unrecognised outcome label stored here",
            tags=["test"],
            workspace_path="/repo",
            source_phase="Testing",
            outcome_label="bogus_label",
        )
        results = store.retrieve(
            session_id="s1",
            query="unrecognised outcome label",
            top_k=1,
            workspace_path="/repo",
        )
        assert len(results) == 1
        assert results[0]["outcome_label"] == "unknown"


# ---------------------------------------------------------------------------
# WAL journal mode
# ---------------------------------------------------------------------------

class TestWALMode:
    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        """PRAGMA journal_mode returns 'wal' after SQLiteMemoryStore initialization."""
        store = _make_store(tmp_path)
        conn = sqlite3.connect(store.db_path)
        try:
            row = conn.execute("PRAGMA journal_mode").fetchone()
            assert row is not None
            assert row[0].lower() == "wal"
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# _composite_score method
# ---------------------------------------------------------------------------

class TestCompositeScoreMethod:
    def test_composite_score_known_inputs(self, tmp_path: Path) -> None:
        """_composite_score with known inputs equals the explicit formula value."""
        store = _make_store(tmp_path)
        semantic = 0.8
        lexical = 0.6
        recency = 0.5
        importance = 0.4
        cross_penalty = 1.0

        result = store._composite_score(semantic, lexical, recency, importance, cross_penalty)

        expected = cross_penalty * (
            WEIGHT_SEMANTIC * semantic
            + WEIGHT_LEXICAL * lexical
            + WEIGHT_RECENCY * recency
            + WEIGHT_IMPORTANCE * importance
        )
        assert abs(result - expected) < 1e-9

    def test_composite_score_cross_penalty_scales_linearly(self, tmp_path: Path) -> None:
        """Halving cross_penalty exactly halves the composite score."""
        store = _make_store(tmp_path)
        base = store._composite_score(0.8, 0.6, 0.5, 0.4, 1.0)
        halved = store._composite_score(0.8, 0.6, 0.5, 0.4, 0.5)
        assert abs(halved - base * 0.5) < 1e-9

    def test_composite_score_zero_inputs_returns_zero(self, tmp_path: Path) -> None:
        """All-zero sub-scores produce a composite score of zero."""
        store = _make_store(tmp_path)
        assert store._composite_score(0.0, 0.0, 0.0, 0.0, 1.0) == 0.0

    def test_composite_score_weights_sum_contribution(self, tmp_path: Path) -> None:
        """With all sub-scores equal to 1.0 and penalty 1.0, result equals sum of weights."""
        store = _make_store(tmp_path)
        result = store._composite_score(1.0, 1.0, 1.0, 1.0, 1.0)
        expected_weight_sum = WEIGHT_SEMANTIC + WEIGHT_LEXICAL + WEIGHT_RECENCY + WEIGHT_IMPORTANCE
        assert abs(result - expected_weight_sum) < 1e-9


# ---------------------------------------------------------------------------
# _importance_score method
# ---------------------------------------------------------------------------

class TestImportanceScoreMethod:
    def test_importance_score_success_same_phase(self, tmp_path: Path) -> None:
        """Success outcome + same phase + zero access count gives the expected importance."""
        store = _make_store(tmp_path)
        result = store._importance_score(
            outcome_label="success",
            access_count=0,
            source_phase="Architecture",
            current_phase="Architecture",
        )
        outcome_score = OUTCOME_PRIORS["success"]           # 1.0
        access_score = math.log1p(0) / math.log1p(ACCESS_COUNT_SOFTCAP)  # 0.0
        phase_score = PHASE_AFFINITY_BONUS                  # 1.0
        expected = (
            IMPORTANCE_OUTCOME_W * outcome_score
            + IMPORTANCE_ACCESS_W * access_score
            + IMPORTANCE_PHASE_W * phase_score
        )
        assert abs(result - expected) < 1e-9

    def test_importance_score_failure_different_phase(self, tmp_path: Path) -> None:
        """Failure outcome + different phase produces the expected lower importance value."""
        store = _make_store(tmp_path)
        result = store._importance_score(
            outcome_label="failure",
            access_count=0,
            source_phase="Planning",
            current_phase="Architecture",
        )
        outcome_score = OUTCOME_PRIORS["failure"]           # 0.2
        access_score = math.log1p(0) / math.log1p(ACCESS_COUNT_SOFTCAP)  # 0.0
        phase_score = PHASE_AFFINITY_NEUTRAL                # 0.5
        expected = (
            IMPORTANCE_OUTCOME_W * outcome_score
            + IMPORTANCE_ACCESS_W * access_score
            + IMPORTANCE_PHASE_W * phase_score
        )
        assert abs(result - expected) < 1e-9

    def test_importance_score_access_count_increases_score(self, tmp_path: Path) -> None:
        """Higher access_count produces a strictly higher importance score."""
        store = _make_store(tmp_path)
        low = store._importance_score("success", 0, "P", "P")
        high = store._importance_score("success", 10, "P", "P")
        assert high > low

    def test_importance_score_softcap_access_count(self, tmp_path: Path) -> None:
        """access_count equal to ACCESS_COUNT_SOFTCAP yields an access_score of 1.0."""
        store = _make_store(tmp_path)
        softcap = int(ACCESS_COUNT_SOFTCAP)
        result = store._importance_score(
            outcome_label="unknown",
            access_count=softcap,
            source_phase="P",
            current_phase="P",
        )
        outcome_score = OUTCOME_PRIORS["unknown"]
        access_score = math.log1p(softcap) / math.log1p(ACCESS_COUNT_SOFTCAP)  # should be 1.0
        phase_score = PHASE_AFFINITY_BONUS
        expected = (
            IMPORTANCE_OUTCOME_W * outcome_score
            + IMPORTANCE_ACCESS_W * access_score
            + IMPORTANCE_PHASE_W * phase_score
        )
        assert abs(result - expected) < 1e-9

    def test_importance_score_unknown_outcome_label(self, tmp_path: Path) -> None:
        """Unrecognised outcome labels fall back to the 'unknown' prior (0.4)."""
        store = _make_store(tmp_path)
        result_unknown = store._importance_score("unknown", 0, "P", "Q")
        result_bogus = store._importance_score("not_a_real_label", 0, "P", "Q")
        # OUTCOME_PRIORS.get falls back to OUTCOME_PRIORS["unknown"] for both
        assert abs(result_unknown - result_bogus) < 1e-9


# ---------------------------------------------------------------------------
# Recency decay
# ---------------------------------------------------------------------------

class TestRecencyDecay:
    def test_recently_inserted_has_higher_recency_in_breakdown(self, tmp_path: Path) -> None:
        """A freshly inserted memory has a recency score close to 1.0 in score_breakdown."""
        store = _make_store(tmp_path)
        store.insert(
            session_id="s1",
            content="Freshly inserted memory for recency decay verification test",
            tags=["recency"],
            workspace_path="/repo",
            source_phase="Testing",
            outcome_label="success",
        )
        results = store.retrieve(
            session_id="s1",
            query="freshly inserted recency decay",
            top_k=1,
            workspace_path="/repo",
        )
        assert len(results) == 1
        recency = float(results[0]["score_breakdown"]["recency"])
        # Just created, so recency should be very close to exp(0) = 1.0
        assert recency > 0.99


# ---------------------------------------------------------------------------
# Access count increment
# ---------------------------------------------------------------------------

class TestAccessCountIncrement:
    def test_access_count_incremented_after_retrieve(self, tmp_path: Path) -> None:
        """access_count in score_breakdown increases after each successful retrieve call."""
        store = _make_store(tmp_path)
        store.insert(
            session_id="s1",
            content="Access count should increase with every retrieval call",
            tags=["access"],
            workspace_path="/repo",
            source_phase="Testing",
            outcome_label="success",
        )
        first = store.retrieve(
            session_id="s1",
            query="access count retrieval call",
            top_k=1,
            workspace_path="/repo",
        )
        second = store.retrieve(
            session_id="s1",
            query="access count retrieval call",
            top_k=1,
            workspace_path="/repo",
        )
        assert len(first) == 1 and len(second) == 1
        first_access = float(first[0]["score_breakdown"]["access_score"])
        second_access = float(second[0]["score_breakdown"]["access_score"])
        assert second_access > first_access


# ---------------------------------------------------------------------------
# Hybrid retrieval ordering
# ---------------------------------------------------------------------------

class TestHybridRetrieval:
    def test_hybrid_retrieval_success_ranked_above_failure(self, tmp_path: Path) -> None:
        """With identical query overlap, success-outcome memory ranks above failure-outcome memory."""
        store = _make_store(tmp_path)
        store.insert(
            session_id="s1",
            content=(
                "Architecture uses hybrid retrieval with hash-vector "
                "cosine reranking and recency decay."
            ),
            tags=["phase-output", "architecture"],
            workspace_path="/repo",
            source_phase="Architecture",
            outcome_label="success",
        )
        store.insert(
            session_id="s1",
            content=(
                "Architecture retrieval failed due to poor context "
                "selection and missing ranking."
            ),
            tags=["phase-output", "architecture"],
            workspace_path="/repo",
            source_phase="Architecture",
            outcome_label="failure",
        )
        results = store.retrieve(
            session_id="s1",
            query="hash-vector cosine retrieval reranking for architecture phase",
            top_k=2,
            workspace_path="/repo",
            include_cross_session=True,
        )
        assert len(results) == 2
        assert results[0]["score"] >= results[1]["score"]
        assert results[0]["outcome_label"] == "success"
        assert "hash_vector_cosine" in results[0]["score_breakdown"]
