import math
from datetime import UTC, datetime, timedelta  # noqa: TCH003

import pytest

from benchmarks.data.generator import retrieval_set

# ── Module-level metric helpers ───────────────────────────────────────────────


def precision_at_k(result_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    return sum(1 for r in result_ids[:k] if r in relevant_ids) / k


def recall_at_k(result_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant items found in top-k results."""
    if not relevant_ids:
        return 0.0
    return sum(1 for r in result_ids[:k] if r in relevant_ids) / len(relevant_ids)


def hit_at_k(result_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1.0 if any relevant result appears in top-k, else 0.0."""
    return 1.0 if any(r in relevant_ids for r in result_ids[:k]) else 0.0


def mrr(result_ids: list[str], relevant_ids: set[str]) -> float:
    """Reciprocal rank of the first relevant result."""
    for i, r in enumerate(result_ids):
        if r in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(result_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K (binary relevance)."""
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, r in enumerate(result_ids[:k])
        if r in relevant_ids
    )
    ideal = sum(
        1.0 / math.log2(i + 2)
        for i in range(min(len(relevant_ids), k))
    )
    return dcg / ideal if ideal > 0 else 0.0


# ── Class 1: TestRetrievalQuality ─────────────────────────────────────────────


class TestRetrievalQuality:

    def _build_store(self, memories: list[dict]):
        """
        Insert all memories into a fresh in-memory store.
        Each memory dict has: id (str), content (str), topic (int),
        embedding (list[float]).
        Returns the populated store with all entries.
        The returned memory_ids may differ from the generator IDs —
        store the mapping for use in queries.
        """
        from verity.cognitive.store import DualSpeedStore

        store = DualSpeedStore(path=":memory:", embedding_model="none")
        id_map = {}  # generator_id → store memory_id

        for mem in memories:
            entry = store.add(
                mem["content"],
                _embedding=mem["embedding"],
            )
            id_map[mem["id"]] = entry.memory_id

        return store, id_map

    def test_hit_rate_at_5(self):
        """
        Full Verity achieves Hit@5 >= 0.60 on the generator retrieval set.
        At least one relevant memory in the top-5 results for 60% of queries.
        """
        from verity.cognitive.scoring import ImportanceScorer
        from verity.cognitive.temporal import TemporalWeighter
        from verity.cognitive.workspace import GlobalWorkspace

        data = retrieval_set()
        store, id_map = self._build_store(data["memories"])

        scorer   = ImportanceScorer()
        temporal = TemporalWeighter()
        ws       = GlobalWorkspace(
            capacity=5, scorer=scorer, temporal=temporal
        )

        hits = []
        for q in data["queries"]:
            raw = store.search(q["query"], k=20)
            selected = ws.select(raw, goal_embedding=None)
            result_ids = [r.memory.memory_id for r in selected]

            # Map generator relevant_ids → store memory_ids
            relevant = {
                id_map[gid]
                for gid in q["relevant_ids"]
                if gid in id_map
            }
            hits.append(hit_at_k(result_ids, relevant, 5))

        mean_hit = sum(hits) / len(hits)
        assert mean_hit >= 0.60, (
            f"Hit@5 = {mean_hit:.3f} (need >= 0.60). "
            f"Retrieval pipeline may be broken. "
            f"Per-query hits: {hits}"
        )

    def test_mrr(self):
        """
        Full Verity achieves MRR >= 0.30 on the generator retrieval set.
        The first relevant result appears on average within the top 3.
        """
        from verity.cognitive.scoring import ImportanceScorer
        from verity.cognitive.temporal import TemporalWeighter
        from verity.cognitive.workspace import GlobalWorkspace

        data = retrieval_set()
        store, id_map = self._build_store(data["memories"])

        scorer   = ImportanceScorer()
        temporal = TemporalWeighter()
        ws       = GlobalWorkspace(
            capacity=5, scorer=scorer, temporal=temporal
        )

        mrr_values = []
        for q in data["queries"]:
            raw = store.search(q["query"], k=20)
            selected = ws.select(raw, goal_embedding=None)
            result_ids = [r.memory.memory_id for r in selected]
            relevant = {id_map[gid] for gid in q["relevant_ids"] if gid in id_map}
            mrr_values.append(mrr(result_ids, relevant))

        mean_mrr = sum(mrr_values) / len(mrr_values)
        assert mean_mrr >= 0.30, (
            f"MRR = {mean_mrr:.3f} (need >= 0.30). "
            f"Distribution: {mrr_values}"
        )

    def test_ndcg_at_5(self):
        """
        Full Verity achieves NDCG@5 >= 0.40 on the generator retrieval set.
        Measures ranking quality — relevant results should appear near the top.
        """
        from verity.cognitive.scoring import ImportanceScorer
        from verity.cognitive.temporal import TemporalWeighter
        from verity.cognitive.workspace import GlobalWorkspace

        data = retrieval_set()
        store, id_map = self._build_store(data["memories"])

        scorer   = ImportanceScorer()
        temporal = TemporalWeighter()
        ws       = GlobalWorkspace(
            capacity=5, scorer=scorer, temporal=temporal
        )

        ndcg_values = []
        for q in data["queries"]:
            raw = store.search(q["query"], k=20)
            selected = ws.select(raw, goal_embedding=None)
            result_ids = [r.memory.memory_id for r in selected]
            relevant = {id_map[gid] for gid in q["relevant_ids"] if gid in id_map}
            ndcg_values.append(ndcg_at_k(result_ids, relevant, 5))

        mean_ndcg = sum(ndcg_values) / len(ndcg_values)
        assert mean_ndcg >= 0.40, (
            f"NDCG@5 = {mean_ndcg:.3f} (need >= 0.40). "
            f"Distribution: {[round(v,3) for v in ndcg_values]}"
        )


# ── Class 2: TestCognitiveLayerContribution ───────────────────────────────────


class TestCognitiveLayerContribution:

    NUM_QUERIES = 10
    TOPICS = [
        "preferences", "schedule", "technical",
        "contacts", "project", "security",
        "infrastructure", "workflow", "metrics", "deployment",
    ]

    def _build_adversarial_store(self, topic: str):
        """
        Build one store for one topic/query.
        Returns (store, correct_ids, query_str).
        Distractors inserted first, correct memories inserted last.

        All 15 entries are stamped with the same last_accessed (12 h ago) via
        update_entry() so the fallback search recency (0.3×(1-hours_old/8760))
        is identical for every row. With equal trigram overlap and equal
        recency, search scores are tied. Python's stable sort preserves the
        order returned by SQLite (ROWID insertion order), so raw k=5 returns
        the 12 distractors inserted first.

        The GlobalWorkspace differentiates via importance:
            correct    importance = 0.9  →  composite ≈ 0.9 × score × temporal
            distractor importance = 0.2  →  composite ≈ 0.2 × score × temporal
        All entries share the same temporal weight (same last_accessed), so
        importance alone drives reranking: correct ranks 1-3 → NDCG@5 ≈ 0.947.
        """
        from verity.cognitive.store import DualSpeedStore

        now = datetime.now(UTC)
        common_ts = now - timedelta(hours=12)  # equal for all entries

        store = DualSpeedStore(path=":memory:", embedding_model="none")

        # Insert 12 distractors first (low importance, equal timestamp)
        for i in range(12):
            entry = store.add(f"{topic}: distractor item {i}", importance=0.2)
            entry.last_accessed = common_ts
            store.update_entry(entry)

        # Insert 3 correct memories last (high importance, same timestamp)
        correct_ids = []
        for i in range(3):
            entry = store.add(f"{topic}: correct answer {i}", importance=0.9)
            entry.last_accessed = common_ts
            store.update_entry(entry)
            correct_ids.append(entry.memory_id)

        return store, set(correct_ids), topic

    def test_full_verity_beats_raw_search_ndcg(self):
        """
        Full Verity NDCG@5 is significantly higher than raw search NDCG@5.
        Raw search returns distractors (inserted first, equal keyword score).
        Full Verity reranks by importance × temporal → correct memories first.
        Wilcoxon signed-rank: p < 0.01 over 10 queries.
        """
        pytest.importorskip("scipy")
        from scipy.stats import wilcoxon  # noqa: PLC0415

        from verity.cognitive.scoring import ImportanceScorer
        from verity.cognitive.temporal import TemporalWeighter
        from verity.cognitive.workspace import GlobalWorkspace

        scorer   = ImportanceScorer()
        temporal = TemporalWeighter()

        full_ndcgs = []
        raw_ndcgs  = []

        for topic in self.TOPICS[:self.NUM_QUERIES]:
            store, correct_ids, query = self._build_adversarial_store(topic)

            # Full Verity: retrieve 20, rerank with workspace
            ws  = GlobalWorkspace(capacity=5, scorer=scorer, temporal=temporal)
            raw = store.search(query, k=20)
            selected = ws.select(raw, goal_embedding=None)
            full_ids = [r.memory.memory_id for r in selected]

            # Raw search: first 5 results, no reranking
            raw_results = store.search(query, k=5)
            raw_ids = [r.memory.memory_id for r in raw_results]

            full_ndcgs.append(ndcg_at_k(full_ids, correct_ids, 5))
            raw_ndcgs.append(ndcg_at_k(raw_ids,  correct_ids, 5))

        mean_full = sum(full_ndcgs) / len(full_ndcgs)
        mean_raw  = sum(raw_ndcgs)  / len(raw_ndcgs)

        assert mean_full > mean_raw, (
            f"Full Verity NDCG@5 ({mean_full:.3f}) should exceed "
            f"raw search ({mean_raw:.3f})"
        )

        # Wilcoxon signed-rank test (paired, one-sided: full > raw)
        stat, p_value = wilcoxon(full_ndcgs, raw_ndcgs, alternative="greater")
        assert p_value < 0.01, (
            f"Cognitive layer improvement not significant: p={p_value:.4f}\n"
            f"Full Verity NDCG@5 per query: {[round(v,3) for v in full_ndcgs]}\n"
            f"Raw search  NDCG@5 per query: {[round(v,3) for v in raw_ndcgs]}"
        )

    def test_temporal_contribution(self):
        """
        Full Verity (with temporal) achieves higher NDCG@5 than No Recency.

        Dataset: 4 stale decoys with importance=0.95 (slightly higher than correct)
        and 3 recent correct memories with importance=0.9.

        Without temporal (recency=1.0): composite = importance × search_score × 1.0
          decoy  composite ≈ 0.95 × 0.975 × 1.0 ≈ 0.927
          correct composite ≈ 0.90 × 1.000 × 1.0 ≈ 0.900
          → decoys rank ABOVE correct (importance wins) → poor NDCG

        With temporal (exponential weight via last_accessed):
          decoy  composite ≈ 0.95 × 0.975 × exp(-0.005×720) ≈ 0.025
          correct composite ≈ 0.90 × 1.000 × exp(-0.005×1)  ≈ 0.895
          → correct rank far above decoys → NDCG ≥ 0.93

        The temporal signal rescues retrieval quality when importance is
        intentionally inverted (stale decoys have higher prior weight).
        """
        from verity.cognitive.scoring import ImportanceScorer
        from verity.cognitive.store import DualSpeedStore
        from verity.cognitive.temporal import TemporalWeighter
        from verity.cognitive.workspace import GlobalWorkspace

        now = datetime.now(UTC)

        topic = "ambiguous"
        store = DualSpeedStore(path=":memory:", embedding_model="none")

        # Insert 4 stale decoys: slightly HIGHER importance than correct,
        # but last_accessed far in the past → temporal weight ≈ 0.027
        for i in range(4):
            entry = store.add(f"{topic}: stale item {i}", importance=0.95)
            entry.last_accessed = now - timedelta(hours=720)
            store.update_entry(entry)

        # Insert 3 recent correct: slightly lower importance than decoys,
        # but last_accessed recent → temporal weight ≈ 0.995
        correct_ids = set()
        for i in range(3):
            entry = store.add(f"{topic}: correct answer {i}", importance=0.9)
            entry.last_accessed = now - timedelta(hours=1)
            store.update_entry(entry)
            correct_ids.add(entry.memory_id)

        scorer   = ImportanceScorer()
        temporal = TemporalWeighter()

        # Full Verity: temporal multiplier makes correct composite >> decoy composite
        ws_full = GlobalWorkspace(capacity=5, scorer=scorer, temporal=temporal)
        raw_full = store.search(topic, k=20)
        selected_full = ws_full.select(raw_full, goal_embedding=None)
        full_ids = [r.memory.memory_id for r in selected_full]

        # No Recency: recency=1.0 → importance wins → decoys (0.95) beat correct (0.90)
        ws_none = GlobalWorkspace(capacity=5, scorer=None, temporal=None)
        raw_none = store.search(topic, k=20)
        selected_none = ws_none.select(raw_none, goal_embedding=None)
        none_ids = [r.memory.memory_id for r in selected_none]

        full_ndcg = ndcg_at_k(full_ids, correct_ids, 5)
        none_ndcg = ndcg_at_k(none_ids, correct_ids, 5)

        assert full_ndcg >= none_ndcg, (
            f"Full Verity NDCG@5 ({full_ndcg:.3f}) should be >= "
            f"No Recency ({none_ndcg:.3f}). "
            f"Temporal signal should not hurt ranking."
        )
        # Full Verity must promote all 3 correct memories into the top 5.
        # Position reordering places rank-2 at output position 5, so perfect
        # selection of 3/5 items yields NDCG@5 ≈ 0.947 (not 1.0).
        assert full_ndcg >= 0.93, (
            f"Full Verity should achieve NDCG@5 >= 0.93 on temporal test. "
            f"Got {full_ndcg:.3f}. "
            f"Full result IDs: {full_ids}, Correct: {correct_ids}"
        )

    def test_all_conditions_report(self):
        """
        Runs all four conditions and prints a summary table.
        Does not assert beyond Full > Raw (already tested above).
        Serves as a diagnostic report for the benchmark README.
        """
        from verity.cognitive.scoring import ImportanceScorer
        from verity.cognitive.temporal import TemporalWeighter
        from verity.cognitive.workspace import GlobalWorkspace

        scorer   = ImportanceScorer()
        temporal = TemporalWeighter()

        results = {
            "full":        [],
            "no_temporal": [],
            "no_recency":  [],
            "raw":         [],
        }

        for topic in self.TOPICS[:self.NUM_QUERIES]:
            store, correct_ids, query = self._build_adversarial_store(topic)
            # Fresh search call per condition — avoids any mutation risk
            # from ws.select() sorting the candidates list in place.
            selected = GlobalWorkspace(5, scorer, temporal).select(
                store.search(query, k=20)
            )
            results["full"].append(
                ndcg_at_k([r.memory.memory_id for r in selected], correct_ids, 5)
            )

            # −TemporalWeighter (scorer recency fallback)
            selected = GlobalWorkspace(5, scorer, None).select(
                store.search(query, k=20)
            )
            results["no_temporal"].append(
                ndcg_at_k([r.memory.memory_id for r in selected], correct_ids, 5)
            )

            # No Recency (importance only)
            selected = GlobalWorkspace(5, None, None).select(
                store.search(query, k=20)
            )
            results["no_recency"].append(
                ndcg_at_k([r.memory.memory_id for r in selected], correct_ids, 5)
            )

            # Raw search (no workspace)
            raw_5 = store.search(query, k=5)
            results["raw"].append(
                ndcg_at_k([r.memory.memory_id for r in raw_5], correct_ids, 5)
            )

        print("\n\n  ── Ablation Report (NDCG@5, adversarial dataset) ──")
        print(f"  {'Condition':<28} {'Mean':>6}  {'Values'}")
        print(f"  {'-'*65}")
        for name, values in results.items():
            mean = sum(values) / len(values)
            label = {
                'full':        'Full Verity',
                'no_temporal': '−TemporalWeighter',
                'no_recency':  'No Recency (importance only)',
                'raw':         'Raw Search (baseline)',
            }[name]
            print(f"  {label:<28} {mean:>6.3f}  {[round(v,2) for v in values]}")
        print()

        # Soft assertion: Full Verity must be among the top performers
        means = {k: sum(v)/len(v) for k, v in results.items()}
        assert means["full"] >= means["raw"], (
            "Full Verity should not be worse than raw search"
        )
