"""apps/ai-service/app/memory.py
================================
SQLite-backed episodic memory store for Martinus.

=== Algorithm 4.10: Composite Top-k Memory Retrieval ===

Implements the Park2023GenerativeAgents memory framework with composite scoring:

    score(m_i) = p_session · (
        w_sem  · s_sem(m_i)    +     # hash-vector cosine similarity (0.50)
        w_lex  · s_lex(m_i)    +     # BM25 score, normalized (0.25)
        w_rec  · r(m_i)        +     # recency: exp(-Δt / τ), τ=48h (0.15)
        w_imp  · imp(m_i)             # importance (0.10)
    )

where:
    imp(m_i) = (
        w_o · outcome_prior(m_i)  +  # success=1.0, partial=0.6, failure=0.2 (0.50)
        w_a · ln(1+c_i)/ln(1+C_max) + # access count normalized (0.30)
        w_p · phase_affinity(m_i)     # same phase=1.0, else=0.5 (0.20)
    )
    p_session = 1.0 if same session, else 0.92 (manually chosen mild in-session
    preference — not grid-searched; configurable via MEMORY_CROSS_SESSION_PENALTY)

Embeddings are computed locally using 512-dimensional feature hashing
(unigrams + bigrams) — lexical signal only, no external embedding model required.
For a true semantic (dense) upgrade, swap _embed_text for an OllamaEmbedder
calling POST /api/embeddings with a sentence-transformer model (future work).

Source: Park et al. (2023), "Generative Agents: Interactive Simulacra of Human Behavior"
in Proc. ACM UIST, adapted as Equation 4.2 in thesis §4.12.
"""
import json
import math
import os
import re
import sqlite3
import time
from contextlib import closing
from collections import Counter
from datetime import datetime, timezone
from hashlib import blake2b
from pathlib import Path
from typing import Dict, List, Optional


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]{2,}")

# Increased from 256 to 512 for higher embedding resolution.
# More dimensions reduce hash collision probability for technical tokens.
EMBEDDING_DIMENSION = 512

BM25_K1 = 1.4
BM25_B = 0.75
RECENCY_HALFLIFE_HOURS = 48.0

# Park2023GenerativeAgents composite scoring weights (Algorithm 4.10 from thesis).
# Balances hash-vector cosine similarity (lexical feature hashing), BM25 lexical scoring,
# recency decay (48-hour half-life), and importance (outcome + access + affinity).
# Both w_sem and w_lex are lexical signals; w_sem uses feature-hash cosine, w_lex uses BM25.
# Weights sum to 1.0 and are applied uniformly across all phases.
#
# v2 rebalance: WEIGHT_SEMANTIC raised from 0.50 to 0.55, WEIGHT_LEXICAL lowered from
# 0.25 to 0.20. Both are lexical signals; reducing BM25 weight prevents double-counting
# lexical overlap while keeping the total at 1.0.
WEIGHT_SEMANTIC = 0.55  # hash-vector cosine (lexical); variable name retained for formula compat
WEIGHT_LEXICAL = 0.20
WEIGHT_RECENCY = 0.15
WEIGHT_IMPORTANCE = 0.10   # combines outcome, access count, phase affinity

# Sub-weights within importance component.
IMPORTANCE_OUTCOME_W = 0.50
IMPORTANCE_ACCESS_W = 0.30
IMPORTANCE_PHASE_W = 0.20

OUTCOME_PRIORS = {
    "success": 1.0,
    "partial": 0.6,
    "failure": 0.2,
    "unknown": 0.4,
}

# Bonus applied when retrieved memory's source_phase matches the query phase.
PHASE_AFFINITY_BONUS = 1.0
PHASE_AFFINITY_NEUTRAL = 0.5

# Access count is normalised with a soft cap.
ACCESS_COUNT_SOFTCAP = 20.0

# Algorithm 4.10 v2: MMR (Maximal Marginal Relevance) diversification.
# λ=1.0 reduces to plain top-k by score; λ=0.0 selects maximally diverse items.
# Default 0.7 is strongly relevance-biased. Configurable via MMR_LAMBDA env var.
# Source: Carbonell, J. & Goldstein, J. (1998). The use of MMR, diversity-based
# reranking for reordering documents and producing summaries. SIGIR '98, 335-336.
# https://doi.org/10.1145/290941.291025
MMR_LAMBDA_DEFAULT = 0.7

# Algorithm N.2: Semantic Deduplication Filter.
# Entries with hash-vector cosine > threshold are treated as near-duplicates;
# the newer content replaces the older entry rather than inserting a duplicate.
# Source: Broder, A. Z. (1997). On the resemblance and containment of documents.
# Proc. Compression and Complexity of Sequences, 21-29.
# https://doi.org/10.1109/SEQUEN.1997.666900
DEDUP_SIMILARITY_THRESHOLD = 0.95


class SQLiteMemoryStore:
    """
    Persistent episodic memory backed by a local SQLite database.

    Each memory entry records the phase output, provider, model, outcome label,
    a 512-dim hash embedding, and access metadata. Retrieval blends hash-vector
    cosine similarity (lexical feature hashing), BM25 lexical scoring, recency
    decay, and importance weighting into a single composite score and returns the
    top-k candidates.

    The schema is forward-compatible: ``_ensure_column`` adds new columns to
    existing databases without data loss.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        for attempt in range(3):
            try:
                conn = sqlite3.connect(self.db_path, timeout=10)
                conn.row_factory = sqlite3.Row
                return conn
            except sqlite3.OperationalError:
                if attempt == 2:
                    raise
                time.sleep(0.05)

    def _initialize(self) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    workspace_key TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    tags_json TEXT NOT NULL,
                    source_phase TEXT NOT NULL DEFAULT '',
                    source_provider TEXT NOT NULL DEFAULT '',
                    source_model TEXT NOT NULL DEFAULT '',
                    outcome_label TEXT NOT NULL DEFAULT 'unknown',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    embedding_json TEXT NOT NULL DEFAULT '[]',
                    access_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-8000")
            required_columns = {
                "workspace_key": "TEXT NOT NULL DEFAULT ''",
                "source_phase": "TEXT NOT NULL DEFAULT ''",
                "source_provider": "TEXT NOT NULL DEFAULT ''",
                "source_model": "TEXT NOT NULL DEFAULT ''",
                "outcome_label": "TEXT NOT NULL DEFAULT 'unknown'",
                "metadata_json": "TEXT NOT NULL DEFAULT '{}'",
                "embedding_json": "TEXT NOT NULL DEFAULT '[]'",
                "access_count": "INTEGER NOT NULL DEFAULT 0",
            }
            for col, col_type in required_columns.items():
                self._ensure_column(conn, col, col_type)

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_session "
                "ON memory_entries(session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_workspace "
                "ON memory_entries(workspace_key)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_created "
                "ON memory_entries(created_at)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_phase "
                "ON memory_entries(source_phase)"
            )
            conn.commit()

    def _ensure_column(
        self,
        conn: sqlite3.Connection,
        column_name: str,
        column_type: str,
    ) -> None:
        columns = conn.execute(
            "PRAGMA table_info(memory_entries)"
        ).fetchall()
        known = {str(row[1]) for row in columns}
        if column_name in known:
            return
        conn.execute(
            "ALTER TABLE memory_entries "
            f"ADD COLUMN {column_name} {column_type}"
        )

    def _workspace_key(self, workspace_path: str) -> str:
        cleaned = workspace_path.strip().lower()
        if not cleaned:
            return ""
        digest = blake2b(
            cleaned.encode("utf-8"), digest_size=8
        ).hexdigest()
        return digest

    def _normalize_outcome_label(self, outcome_label: str) -> str:
        normalized = outcome_label.strip().lower()
        if normalized in OUTCOME_PRIORS:
            return normalized
        return "unknown"

    def _embed_text(self, text: str) -> List[float]:
        """
        Produce a 512-dimensional lexical hash vector via feature hashing.

        Uses unigram + bigram hashing (Blake2b) into a fixed-dim space.
        This is a *lexical* signal: cosine similarity on these vectors measures
        token-identity overlap, not semantic relatedness. Two sentences with no
        shared vocabulary will score near zero even if semantically equivalent.
        """
        vector = [0.0] * EMBEDDING_DIMENSION
        tokens = TOKEN_RE.findall(text.lower())
        if not tokens:
            return vector

        # Unigram features
        unigrams = Counter(tokens)
        for token, freq in unigrams.items():
            self._hash_into(vector, token, float(freq))

        # Bigram features (captures co-occurrence at document level)
        for i in range(len(tokens) - 1):
            bigram = tokens[i] + "_" + tokens[i + 1]
            self._hash_into(vector, bigram, 0.5)

        norm = math.sqrt(sum(v * v for v in vector))
        if norm == 0:
            return vector
        return [v / norm for v in vector]

    def _hash_into(
        self, vector: List[float], token: str, weight: float
    ) -> None:
        digest = blake2b(
            token.encode("utf-8"), digest_size=8
        ).digest()
        hashed = int.from_bytes(digest, byteorder="big", signed=False)
        index = hashed % EMBEDDING_DIMENSION
        sign = 1.0 if (hashed & 1) else -1.0
        vector[index] += sign * weight

    def _deserialize_embedding(
        self,
        serialized: str,
        fallback_text: str,
    ) -> List[float]:
        try:
            payload = json.loads(serialized)
            if isinstance(payload, list) and payload:
                values = [float(item) for item in payload]
                if len(values) == EMBEDDING_DIMENSION:
                    return values
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
        return self._embed_text(fallback_text)

    def _cosine_similarity(
        self,
        left: List[float],
        right: List[float],
    ) -> float:
        if not left or not right:
            return 0.0
        length = min(len(left), len(right))
        dot = sum(left[i] * right[i] for i in range(length))
        return max(0.0, dot)

    def _parse_created_at(self, value: str) -> datetime:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return datetime.now(timezone.utc)

    def _bm25_scores(
        self,
        query_terms: List[str],
        docs: List[Dict[str, object]],
    ) -> Dict[int, float]:
        if not query_terms or not docs:
            return {}

        doc_count = len(docs)
        average_doc_len = max(
            1.0,
            sum(float(doc["token_count"]) for doc in docs)
            / float(doc_count),
        )

        document_frequency: Dict[str, int] = {
            term: 0 for term in query_terms
        }
        for doc in docs:
            term_counter = doc["term_counter"]
            assert isinstance(term_counter, Counter)
            for term in query_terms:
                if term_counter.get(term, 0) > 0:
                    document_frequency[term] += 1

        raw_scores: Dict[int, float] = {}
        for doc in docs:
            term_counter = doc["term_counter"]
            assert isinstance(term_counter, Counter)
            token_count = max(1.0, float(doc["token_count"]))
            bm25_total = 0.0
            for term in query_terms:
                term_frequency = float(term_counter.get(term, 0))
                if term_frequency <= 0:
                    continue
                df = float(document_frequency.get(term, 0))
                idf = math.log(
                    1.0
                    + ((doc_count - df + 0.5) / (df + 0.5))
                )
                denominator = term_frequency + BM25_K1 * (
                    1.0
                    - BM25_B
                    + BM25_B * (token_count / average_doc_len)
                )
                bm25_total += idf * (
                    term_frequency
                    * (BM25_K1 + 1.0)
                    / max(denominator, 1e-9)
                )

            row = doc["row"]
            assert isinstance(row, sqlite3.Row)
            raw_scores[int(row["id"])] = bm25_total

        if not raw_scores:
            return {}

        max_score = max(raw_scores.values())
        if max_score <= 0:
            return {k: 0.0 for k in raw_scores}

        return {k: v / max_score for k, v in raw_scores.items()}

    def _mmr_select(
        self,
        scored: List[Dict[str, object]],
        top_k: int,
        lambda_param: float,
    ) -> List[Dict[str, object]]:
        """
        Algorithm 4.10 (MMR extension): Maximal Marginal Relevance selection.

        Iteratively picks the candidate that maximises:
            λ · score(m) − (1 − λ) · max_{s ∈ S} cos(embed(m), embed(s))
        where S is the set of already-selected items.

        When λ=1.0 this reduces to plain top-k by score.

        Source: Carbonell, J. & Goldstein, J. (1998). SIGIR '98.
        https://doi.org/10.1145/290941.291025
        """
        if not scored:
            return []
        if top_k >= len(scored) or lambda_param >= 1.0:
            return scored[:top_k]

        selected: List[Dict[str, object]] = []
        remaining = list(scored)

        while len(selected) < top_k and remaining:
            if not selected:
                selected.append(remaining.pop(0))
                continue

            selected_embeddings = [
                self._deserialize_embedding(
                    str(item.get("_embedding_json", "[]")),
                    str(item["content"]),
                )
                for item in selected
            ]

            best_idx = 0
            best_mmr = float("-inf")
            for idx, candidate in enumerate(remaining):
                cand_embed = self._deserialize_embedding(
                    str(candidate.get("_embedding_json", "[]")),
                    str(candidate["content"]),
                )
                max_sim = max(
                    (self._cosine_similarity(cand_embed, sel_embed)
                     for sel_embed in selected_embeddings),
                    default=0.0,
                )
                mmr_score = (
                    lambda_param * float(candidate["score"])
                    - (1.0 - lambda_param) * max_sim
                )
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = idx

            selected.append(remaining.pop(best_idx))

        return selected

    def _find_near_duplicate(
        self,
        embedding: List[float],
        workspace_key: str,
        session_id: str,
    ) -> Optional[int]:
        """
        Algorithm N.2: Semantic Deduplication Filter.

        Returns the row id of an existing entry whose hash-vector cosine
        similarity exceeds DEDUP_SIMILARITY_THRESHOLD, or None if no
        near-duplicate exists. Checks the last 200 entries for performance.
        """
        if not workspace_key:
            return None
        with closing(self._connect()) as conn:
            rows = conn.execute(
                """
                SELECT id, embedding_json FROM memory_entries
                WHERE (session_id = ? OR workspace_key = ?)
                  AND workspace_key != ''
                ORDER BY id DESC
                LIMIT 200
                """,
                (session_id, workspace_key),
            ).fetchall()

        for row in rows:
            existing = self._deserialize_embedding(str(row["embedding_json"]), "")
            if not existing:
                continue
            if self._cosine_similarity(embedding, existing) >= DEDUP_SIMILARITY_THRESHOLD:
                return int(row["id"])
        return None

    def _importance_score(
        self,
        outcome_label: str,
        access_count: int,
        source_phase: str,
        current_phase: str,
    ) -> float:
        """
        Compute the importance sub-score (ACAN-inspired) for a memory entry.

        Combines three signals:
            - outcome prior: success=1.0, partial=0.6, failure=0.2, unknown=0.4
            - access frequency: soft-capped log normalisation
            - phase affinity: boost for memories from the same phase
        """
        outcome_score = OUTCOME_PRIORS.get(outcome_label, OUTCOME_PRIORS["unknown"])

        # Access frequency: soft-capped log normalisation.
        raw_access = max(0, access_count)
        access_score = math.log1p(raw_access) / math.log1p(ACCESS_COUNT_SOFTCAP)

        # Phase affinity: boost memories from the same phase.
        if current_phase and source_phase == current_phase:
            phase_score = PHASE_AFFINITY_BONUS
        else:
            phase_score = PHASE_AFFINITY_NEUTRAL

        return (
            IMPORTANCE_OUTCOME_W * outcome_score
            + IMPORTANCE_ACCESS_W * access_score
            + IMPORTANCE_PHASE_W * phase_score
        )

    def _composite_score(
        self,
        semantic: float,
        lexical: float,
        recency: float,
        importance: float,
        cross_penalty: float,
    ) -> float:
        """
        Algorithm 4.10: composite scoring formula.

        score = cross_penalty · (w_sem·semantic + w_lex·lexical + w_rec·recency + w_imp·importance)
        """
        return cross_penalty * (
            WEIGHT_SEMANTIC * semantic
            + WEIGHT_LEXICAL * lexical
            + WEIGHT_RECENCY * recency
            + WEIGHT_IMPORTANCE * importance
        )

    def insert(
        self,
        session_id: str,
        content: str,
        tags: List[str],
        workspace_path: str = "",
        source_phase: str = "",
        source_provider: str = "",
        source_model: str = "",
        outcome_label: str = "unknown",
        metadata: Optional[Dict[str, str]] = None,
    ) -> int:
        payload = json.dumps(tags)
        metadata_payload = json.dumps(metadata or {})
        embedding = self._embed_text(content)
        embedding_payload = json.dumps(embedding)
        created_at = datetime.now(timezone.utc).isoformat()
        workspace_key = self._workspace_key(workspace_path)
        normalized_outcome = self._normalize_outcome_label(outcome_label)

        # Algorithm N.2: check for near-duplicate before inserting.
        # Use a single connection for both the dedup check and the actual write.
        with closing(self._connect()) as conn:
            dup_rows = conn.execute(
                """
                SELECT id, embedding_json FROM memory_entries
                WHERE (session_id = ? OR workspace_key = ?)
                  AND workspace_key != ''
                ORDER BY id DESC
                LIMIT 200
                """,
                (session_id, workspace_key),
            ).fetchall() if workspace_key else []

            dup_id: Optional[int] = None
            for row in dup_rows:
                existing = self._deserialize_embedding(str(row["embedding_json"]), "")
                if not existing:
                    continue
                if self._cosine_similarity(embedding, existing) >= DEDUP_SIMILARITY_THRESHOLD:
                    dup_id = int(row["id"])
                    break

            if dup_id is not None:
                conn.execute(
                    "UPDATE memory_entries SET content = ?, embedding_json = ?, created_at = ? WHERE id = ?",
                    (content, embedding_payload, created_at, dup_id),
                )
                conn.commit()
                return dup_id

            cursor = conn.execute(
                """
                INSERT INTO memory_entries
                (
                    session_id, workspace_key, content, tags_json,
                    source_phase, source_provider, source_model,
                    outcome_label, metadata_json, embedding_json,
                    access_count, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    session_id,
                    workspace_key,
                    content,
                    payload,
                    source_phase,
                    source_provider,
                    source_model,
                    normalized_outcome,
                    metadata_payload,
                    embedding_payload,
                    created_at,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def retrieve(
        self,
        session_id: str,
        query: str,
        top_k: int,
        workspace_path: str = "",
        include_cross_session: bool = True,
        current_phase: str = "",
    ) -> List[Dict[str, object]]:
        """
        Retrieve the top-k most relevant memory entries using ACAN-inspired
        composite scoring:
            score = α·hash-vector-cosine + β·lexical(BM25) + γ·recency + δ·importance
        where importance aggregates outcome prior, access frequency, and
        phase affinity — matching the memory formula from the architecture
        blueprint (Frontiers in Psychology, 2025).

        Algorithm upgrade (v4): cross_session_penalty is read once per retrieval
        call and cached for the scoring loop instead of being read from os.getenv
        on every scored document (old: O(n) env reads; new: O(1) env read).

        After retrieval, access_count is incremented for returned entries
        to track popularity and improve future importance scoring.
        """
        workspace_key = self._workspace_key(workspace_path)

        # Cache cross-session penalty once per call — avoids O(n) env reads in the loop.
        cross_session_penalty_value = float(
            os.getenv("MEMORY_CROSS_SESSION_PENALTY", "0.92")
        )

        with closing(self._connect()) as conn:
            if include_cross_session and workspace_key:
                rows = conn.execute(
                    """
                    SELECT
                        id, session_id, content, tags_json,
                        source_phase, source_provider, source_model,
                        outcome_label, metadata_json, embedding_json,
                        access_count, created_at
                    FROM memory_entries
                    WHERE session_id = ? OR workspace_key = ?
                    ORDER BY id DESC
                    LIMIT 800
                    """,
                    (session_id, workspace_key),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT
                        id, session_id, content, tags_json,
                        source_phase, source_provider, source_model,
                        outcome_label, metadata_json, embedding_json,
                        access_count, created_at
                    FROM memory_entries
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT 500
                    """,
                    (session_id,),
                ).fetchall()

        query_terms = TOKEN_RE.findall(query.lower())
        query_embedding = self._embed_text(query)

        documents: List[Dict[str, object]] = []
        for row in rows:
            content = str(row["content"])
            term_counter = Counter(TOKEN_RE.findall(content.lower()))
            token_count = sum(term_counter.values())
            documents.append(
                {
                    "row": row,
                    "content": content,
                    "term_counter": term_counter,
                    "token_count": token_count,
                    "embedding": self._deserialize_embedding(
                        str(row["embedding_json"]),
                        content,
                    ),
                    "created_at": self._parse_created_at(
                        str(row["created_at"])
                    ),
                }
            )

        lexical_scores = self._bm25_scores(query_terms, documents)
        now = datetime.now(timezone.utc)

        scored: List[Dict[str, object]] = []
        for doc in documents:
            row = doc["row"]
            assert isinstance(row, sqlite3.Row)
            identifier = int(row["id"])

            semantic = self._cosine_similarity(
                query_embedding, doc["embedding"]
            )
            lexical = lexical_scores.get(identifier, 0.0)

            age_hours = max(
                0.0,
                (now - doc["created_at"]).total_seconds() / 3600.0,
            )
            recency = math.exp(-age_hours / RECENCY_HALFLIFE_HOURS)

            # --- Importance component (ACAN-inspired) ---
            outcome_label = self._normalize_outcome_label(
                str(row["outcome_label"])
            )
            importance = self._importance_score(
                outcome_label,
                int(row["access_count"]),
                str(row["source_phase"]),
                current_phase,
            )

            # Reconstruct sub-scores for score_breakdown reporting.
            outcome_score = OUTCOME_PRIORS.get(outcome_label, OUTCOME_PRIORS["unknown"])
            raw_access = max(0, int(row["access_count"]))
            access_score = math.log1p(raw_access) / math.log1p(ACCESS_COUNT_SOFTCAP)
            if current_phase and str(row["source_phase"]) == current_phase:
                phase_score = PHASE_AFFINITY_BONUS
            else:
                phase_score = PHASE_AFFINITY_NEUTRAL

            # Cross-session penalty: read once per retrieval call (cached above).
            # Manually chosen mild in-session preference; configurable via
            # MEMORY_CROSS_SESSION_PENALTY env var (default 0.92).
            cross_penalty = (
                cross_session_penalty_value
                if str(row["session_id"]) != session_id
                else 1.0
            )

            composite = self._composite_score(semantic, lexical, recency, importance, cross_penalty)

            if composite <= 0:
                continue

            scored.append(
                {
                    "id": identifier,
                    "session_id": str(row["session_id"]),
                    "score": composite,
                    "content": doc["content"],
                    "tags": json.loads(str(row["tags_json"])),
                    "source_phase": str(row["source_phase"]),
                    "source_provider": str(row["source_provider"]),
                    "source_model": str(row["source_model"]),
                    "outcome_label": outcome_label,
                    "created_at": str(row["created_at"]),
                    "score_breakdown": {
                        "hash_vector_cosine": semantic,
                        "lexical": lexical,
                        "recency": recency,
                        "importance": importance,
                        "outcome_score": outcome_score,
                        "access_score": access_score,
                        "phase_score": phase_score,
                        "cross_session_penalty": cross_penalty,
                    },
                    "_embedding_json": str(row["embedding_json"]),  # internal: used by MMR
                }
            )

        scored.sort(key=lambda item: float(item["score"]), reverse=True)

        # Algorithm 4.10 v2: MMR diversification after score-based ranking.
        mmr_lambda = float(os.getenv("MMR_LAMBDA", str(MMR_LAMBDA_DEFAULT)))
        top = self._mmr_select(scored, top_k, mmr_lambda)

        # Strip internal MMR field before returning to callers.
        for item in top:
            item.pop("_embedding_json", None)

        # Increment access_count for returned entries to track usage.
        if top:
            ids = [int(item["id"]) for item in top]
            placeholders = ",".join("?" * len(ids))
            with closing(self._connect()) as conn:
                conn.execute(
                    "UPDATE memory_entries "
                    "SET access_count = access_count + 1 "
                    f"WHERE id IN ({placeholders})",
                    ids,
                )
                conn.commit()

        return top
