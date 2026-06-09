"""Step E: Knowledge Graph Completion via semantic similarity and LLM classification."""

import hashlib
import json
import logging
from collections.abc import Callable
from typing import Literal

from llama_index.core.program import LLMTextCompletionProgram
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.cache import cache_manager
from src.config import DEFAULT_EMBEDDING_MODEL, DEFAULT_CHAT_MODEL
from src.graph import get_embedding_dimensions
from src.llama_setup import get_embed_model, get_llm
from src.schema_adapter import SchemaAdapter, SorosAdapter

logger = logging.getLogger(__name__)

# Bump when prompt or schema changes to invalidate classification cache.
# v3: classifier sees each concept's existing graph neighborhood (bab + typed
# out-edges + shared neighbors) in addition to name/description/formula.
# v4: cache key includes schema name (soros/yhoga) so the two vocabularies'
# classifications no longer collide; results now carry a `description` field
# (populated by the Yhoga LINTAS_BUKU prompt, empty for Soros).
CLASSIFICATION_PROMPT_VERSION = "v4"

# Cap out-edges per concept in the prompt to bound prompt size.
NEIGHBORHOOD_MAX_EDGES = 5


def _is_rate_limit_error(exception: BaseException) -> bool:
    """Check if exception is a rate limit error (HTTP 429)."""
    error_str = str(exception).lower()
    if "429" in error_str or "rate limit" in error_str:
        return True
    status = getattr(exception, "status_code", None)
    if status == 429:
        return True
    response = getattr(exception, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if resp_status == 429:
            return True
    return False


def _is_retriable_error(exception: BaseException) -> bool:
    """Retriable: 429 rate limits, 5xx server errors, and transient network issues."""
    if _is_rate_limit_error(exception):
        return True

    error_str = str(exception).lower()
    # Transient network / server hints
    transient_markers = (
        "timeout",
        "timed out",
        "connection reset",
        "connection aborted",
        "connection error",
        "temporarily unavailable",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "502",
        "503",
        "504",
    )
    if any(marker in error_str for marker in transient_markers):
        return True

    status = getattr(exception, "status_code", None)
    if isinstance(status, int) and 500 <= status < 600:
        return True
    response = getattr(exception, "response", None)
    if response is not None:
        resp_status = getattr(response, "status_code", None)
        if isinstance(resp_status, int) and 500 <= resp_status < 600:
            return True

    return False


@retry(
    retry=retry_if_exception(_is_retriable_error),
    wait=wait_exponential(multiplier=2, min=10, max=120),
    stop=stop_after_attempt(5),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        "Transient error, retrying in %.1f seconds (attempt %d/5): %s",
        retry_state.next_action.sleep if retry_state.next_action else 10,
        retry_state.attempt_number,
        retry_state.outcome.exception() if retry_state.outcome else "unknown",
    ),
)
def _get_embeddings_raw(texts: list[str], model: str) -> list[list[float]]:
    """Get embeddings from API without caching."""
    embed_model = get_embed_model(model)
    return embed_model.get_text_embedding_batch(texts)


def build_embed_text(node: dict, adapter: SchemaAdapter | None = None) -> str:
    """Build the text to embed for a node.

    When `adapter` is provided, delegates to its `build_embed_text`. Otherwise
    falls back to the soros-flavored layout (formula/variables/kondisi) for
    backward compat with older callers.
    """
    if adapter is not None:
        return adapter.build_embed_text(node)
    text = f"{node['name']}. {node.get('description', node['name'])}"
    if node.get("formula"):
        text += f" Rumus: {', '.join(node['formula'])}"
    if node.get("variables"):
        text += f" Variabel: {', '.join(node['variables'])}"
    if node.get("kondisi"):
        text += f" Kondisi: {', '.join(node['kondisi'])}"
    return text


def get_embeddings(
    texts: list[str],
    model: str | None = None,
    use_cache: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> list[list[float]]:
    """Get embeddings for a list of texts with optional caching."""
    model = model or DEFAULT_EMBEDDING_MODEL
    total = len(texts)

    if not use_cache:
        if progress_callback:
            progress_callback(0, total, "Fetching embeddings from API...")
        result = _get_embeddings_raw(texts, model)
        if progress_callback:
            progress_callback(total, total, "Embeddings complete")
        return result

    embeddings = []
    uncached_indices = []
    uncached_texts = []
    cached_count = 0

    for i, text in enumerate(texts):
        cached = cache_manager.get_embedding(text, model)
        if cached is not None:
            embeddings.append(cached)
            cached_count += 1
            logger.debug("Loaded embedding from cache for text %d", i)
            if progress_callback:
                progress_callback(
                    cached_count,
                    total,
                    f"Loading from cache ({cached_count}/{total})",
                )
        else:
            embeddings.append(None)
            uncached_indices.append(i)
            uncached_texts.append(text)

    if uncached_texts:
        logger.info(
            "Fetching embeddings for %d uncached texts (model: %s)",
            len(uncached_texts),
            model,
        )
        if progress_callback:
            progress_callback(
                cached_count,
                total,
                f"Fetching {len(uncached_texts)} embeddings from API...",
            )

        new_embeddings = _get_embeddings_raw(uncached_texts, model)

        for idx, emb in zip(uncached_indices, new_embeddings):
            embeddings[idx] = emb
            cache_manager.put_embedding(texts[idx], model, emb)
            cached_count += 1
            if progress_callback:
                progress_callback(
                    cached_count,
                    total,
                    f"Embedding {cached_count}/{total}",
                )

    if progress_callback:
        progress_callback(total, total, "Embeddings complete")

    return embeddings


def find_similar_pairs_ann(
    driver,
    nodes: list[dict],
    embeddings: list[list[float]],
    model: str,
    threshold: float = 0.8,
    top_k: int = 10,
    allowed_names: set[str] | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    adapter: SchemaAdapter | None = None,
    cross_grade_only: bool = False,
    skip_existing_typed_edges: bool = False,
) -> list[dict]:
    """Find similar pairs using Neo4j Vector Index (ANN search).

    This is O(n log n) instead of O(n²) for the brute-force approach.

    Args:
        driver: Neo4j driver
        nodes: List of dicts with 'name', 'description', 'labels' keys, and
            (for Yhoga) 'grade'.
        embeddings: Corresponding embedding vectors
        model: Embedding model name (for index creation)
        threshold: Minimum similarity score
        top_k: Number of similar nodes to find per query
        allowed_names: If provided, only pairs where BOTH source and target names
            appear in this set are kept. Used by single-doc scope to restrict
            matches to the selected document's nodes even though the vector
            index contains embeddings for the entire graph.
        progress_callback: Optional progress callback
        adapter: SchemaAdapter for the target KG. Defaults to SorosAdapter for
            backward compat.
        cross_grade_only: If True (Yhoga LINTAS_BUKU scope), drop any pair where
            source.grade == target.grade. The grade lookup comes from the
            ``nodes`` list; nodes missing a grade are conservatively excluded
            from cross-grade pairs.
        skip_existing_typed_edges: If True, drop any pair where a non-SIMILAR_TO
            edge already exists between source and target. Prevents redundant
            discovery (e.g. concepts already connected by within-book extraction
            edges) and prevents re-classifying pairs already linked by a prior
            LINTAS_BUKU_* completion run. Adapter-implemented via
            ``adapter.get_pairs_with_existing_typed_edges``.

    Returns list of dicts with 'source', 'target', 'similarity'.
    """
    if adapter is None:
        adapter = SorosAdapter()
    dimensions = get_embedding_dimensions(model)
    if progress_callback:
        progress_callback(0, len(nodes), "Creating vector index...")

    adapter.create_vector_indexes(driver, dimensions=dimensions)

    # Build name → grade lookup for the cross-grade filter. Built unconditionally
    # so it's available regardless of flag; cost is trivial for 342-node Yhoga.
    grade_by_name: dict[str, str] = {
        n["name"]: (n.get("grade") or "") for n in nodes
    }

    # Fast path: skip uploading embeddings that Neo4j already has under the
    # same model. The `embedding` property is the source of truth for the
    # vector index; re-setting it with identical bytes is pure overhead.
    node_names = [n["name"] for n in nodes]
    already_current = adapter.get_nodes_with_current_embedding(driver, node_names, model)
    to_store = [
        (n, e) for n, e in zip(nodes, embeddings) if n["name"] not in already_current
    ]

    if to_store:
        if progress_callback:
            progress_callback(
                0,
                len(nodes),
                f"Storing embeddings on {len(to_store)} nodes "
                f"({len(already_current)} already current)...",
            )
        store_nodes = [p[0] for p in to_store]
        store_embs = [p[1] for p in to_store]
        adapter.store_node_embeddings(driver, store_nodes, store_embs, model)
    else:
        logger.info(
            "All %d embeddings already current on Neo4j, skipping upload", len(nodes)
        )
        if progress_callback:
            progress_callback(
                0, len(nodes), "Embeddings already current, skipping upload"
            )

    pairs = []
    seen_pairs: set[tuple[str, str]] = set()
    total = len(nodes)

    for i, (node, emb) in enumerate(zip(nodes, embeddings)):
        if progress_callback:
            progress_callback(i, total, f"Querying similar for {node['name'][:30]}...")

        try:
            similar = adapter.query_similar_nodes(
                driver,
                query_embedding=emb,
                k=top_k,
                threshold=threshold,
                exclude_name=node["name"],
            )
            for s in similar:
                # The vector index covers the whole graph; restrict to the
                # caller's allowed set when given (e.g. single-doc scope).
                if allowed_names is not None and s["name"] not in allowed_names:
                    continue
                if cross_grade_only:
                    source_grade = grade_by_name.get(node["name"], "")
                    target_grade = grade_by_name.get(s["name"], "")
                    # Drop same-grade pairs and pairs with missing grade info
                    # (latter would silently let same-grade leak through if
                    # both nodes happen to lack a grade label).
                    if (
                        not source_grade
                        or not target_grade
                        or source_grade == target_grade
                    ):
                        continue
                pair_key = tuple(sorted([node["name"], s["name"]]))
                if pair_key not in seen_pairs:
                    seen_pairs.add(pair_key)
                    pairs.append(
                        {
                            "source": node["name"],
                            "target": s["name"],
                            "similarity": float(s["score"]),
                            "source_grade": grade_by_name.get(node["name"], ""),
                            "target_grade": grade_by_name.get(s["name"], ""),
                        }
                    )
        except Exception as e:
            logger.warning("Failed to query similar for %s: %s", node["name"], e)

    if progress_callback:
        progress_callback(total, total, f"Found {len(pairs)} similar pairs via ANN")

    if skip_existing_typed_edges and pairs:
        already = adapter.get_pairs_with_existing_typed_edges(driver, pairs)
        if already:
            before = len(pairs)
            pairs = [
                p for p in pairs
                if tuple(sorted([p["source"], p["target"]])) not in already
            ]
            logger.info(
                "Dropped %d/%d pairs already typed-connected (non-SIMILAR_TO edge present)",
                before - len(pairs),
                before,
            )
            if progress_callback:
                progress_callback(
                    total,
                    total,
                    f"Found {len(pairs)} similar pairs ({before - len(pairs)} skipped: already typed-connected)",
                )

    return pairs


# Classification prompt for typed relationships.
# Bump CLASSIFICATION_PROMPT_VERSION at the top of this module whenever the
# prompt text or schema changes, to invalidate cached classifications.
_CLASSIFICATION_PROMPT = """\
Anda adalah ahli kurikulum yang mengklasifikasikan hubungan antar konsep pembelajaran.

Diberikan dua konsep yang memiliki kemiripan semantik tinggi:

- Konsep A: "{name_a}"
  Deskripsi: "{desc_a}"
  Rumus: {formula_a}
  Variabel: {variables_a}
  Kondisi: {kondisi_a}

- Konsep B: "{name_b}"
  Deskripsi: "{desc_b}"
  Rumus: {formula_b}
  Variabel: {variables_b}
  Kondisi: {kondisi_b}

{neighborhood_block}
Klasifikasikan hubungan antara A dan B sebagai SALAH SATU dari:
1. **isPrerequisiteOf**: A harus dipelajari sebelum B (ketergantungan urutan ketat).
   Contoh: "Bilangan Bulat" → "Persamaan Linear".
2. **supports**: A membantu pemahaman B atau A diterapkan dalam B (ketergantungan fungsional).
   Contoh: "Hukum Newton" → "Gaya Gesek".
3. **analogousTo**: A dan B memiliki pola/prinsip yang sama (hubungan struktural, anti-silo).
   Contoh: "Laju Reaksi Kimia" ↔ "Kecepatan Reaksi Biologis".
4. **none**: Tidak ada hubungan bermakna meski mirip secara semantik.

Gunakan rumus, variabel, dan kondisi untuk membedakan `supports` (berbagi konteks penerapan)
dari `analogousTo` (berbagi struktur tapi konteks berbeda).

Kembalikan JSON dengan field:
- "rel_type": salah satu dari "isPrerequisiteOf", "supports", "analogousTo", "none"
- "confidence": skor 0.0–1.0 menyatakan keyakinan klasifikasi
"""


class Classification(BaseModel):
    """Structured output for a single pair classification (Soros 3-type)."""

    rel_type: Literal["isPrerequisiteOf", "supports", "analogousTo", "none"]
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


# ── Yhoga LINTAS_BUKU classifier — 5-type closed cross-book vocab ───────────
# Vocab per docs/yhoga-ontology.ttl:151-185. Emitted only when adapter.spec
# carries the LINTAS_BUKU classifier_vocab (i.e. schema='yhoga').

_LINTAS_BUKU_CLASSIFICATION_PROMPT = """\
Anda adalah ahli kurikulum sains yang menganalisis keterkaitan **lintas-buku** antara konsep
pada Mata Pelajaran berbeda (Biologi/Fisika/Kimia Kelas XII pada Kurikulum Merdeka).

Diberikan dua konsep yang memiliki kemiripan semantik tinggi tetapi berasal dari **buku berbeda**:

- Konsep A: "{name_a}"
  Mata Pelajaran A: "{grade_a}"
  Deskripsi A: "{desc_a}"
  Materi Pokok A: {materi_pokok_a}

- Konsep B: "{name_b}"
  Mata Pelajaran B: "{grade_b}"
  Deskripsi B: "{desc_b}"
  Materi Pokok B: {materi_pokok_b}

{neighborhood_block}
Klasifikasikan hubungan **lintas-buku** antara A dan B sebagai SALAH SATU dari:

1. **LINTAS_BUKU_SAMA_DENGAN**: A dan B merujuk pada entitas/konsep yang sama secara semantik,
   meskipun dibahas dalam buku berbeda. Contoh: "Energi" (Fisika) ↔ "Energi" (Kimia) — keduanya
   merujuk pada konsep energi yang sama. Bersifat simetris.

2. **LINTAS_BUKU_APLIKASI_DARI**: A adalah penerapan/manifestasi konsep B di domain disiplin lain.
   Contoh: "Difusi" (Biologi) adalah aplikasi dari "Gerak Brown" (Fisika). Asimetris — arahkan
   dari aplikasi ke prinsip dasar.

3. **LINTAS_BUKU_PRASYARAT_UNTUK**: Konsep A pada satu mata pelajaran adalah prasyarat untuk
   memahami konsep B pada mata pelajaran lain. Contoh: "Stoikiometri" (Kimia) prasyarat untuk
   "Termokimia Lanjutan" (Kimia/Fisika lintas-disiplin). Asimetris — arahkan dari prasyarat ke
   konsep yang membutuhkannya.

4. **LINTAS_BUKU_MEMPERDALAM**: Konsep A memperdalam/memperluas pemahaman konsep B di buku lain.
   Contoh: "Termodinamika" (Fisika) memperdalam "Reaksi Endoterm/Eksoterm" (Kimia). Asimetris.

5. **LINTAS_BUKU_BERKAITAN_DENGAN**: Konsep A dan B berkaitan secara umum lintas buku tanpa
   hubungan struktural ketat — fallback bila empat tipe di atas tidak tepat. Simetris.

6. **none**: Kemiripan semantik hanya bersifat permukaan; A dan B tidak benar-benar terkait
   secara curricular. Pilih ini bila ragu.

PENTING: Anda HANYA boleh mengklasifikasikan hubungan ketika Mata Pelajaran A ≠ Mata Pelajaran B.
Jika keduanya dari mata pelajaran yang sama, kembalikan "none".

Kembalikan JSON dengan field:
- "rel_type": salah satu dari "LINTAS_BUKU_SAMA_DENGAN", "LINTAS_BUKU_APLIKASI_DARI",
  "LINTAS_BUKU_PRASYARAT_UNTUK", "LINTAS_BUKU_MEMPERDALAM", "LINTAS_BUKU_BERKAITAN_DENGAN", "none"
- "confidence": skor 0.0–1.0 menyatakan keyakinan klasifikasi
- "description": satu kalimat singkat (≤25 kata) yang menjelaskan ALASAN keterkaitan
  lintas-buku ini. Wajib bahasa Indonesia. Kosongkan ("") jika rel_type = "none".
"""


class LintasBukuClassification(BaseModel):
    """Structured output for a single Yhoga LINTAS_BUKU_* pair classification."""

    rel_type: Literal[
        "LINTAS_BUKU_SAMA_DENGAN",
        "LINTAS_BUKU_APLIKASI_DARI",
        "LINTAS_BUKU_PRASYARAT_UNTUK",
        "LINTAS_BUKU_MEMPERDALAM",
        "LINTAS_BUKU_BERKAITAN_DENGAN",
        "none",
    ]
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    description: str = Field(default="")


def _format_list(items) -> str:
    """Format a list of strings for prompt inclusion."""
    if not items:
        return "(tidak ada)"
    return ", ".join(str(x) for x in items)


def _format_out_edges(edges: list[dict] | None) -> str:
    if not edges:
        return "  (tidak ada)"
    lines = []
    for e in edges:
        conf = e.get("confidence")
        conf_str = (
            f" (conf: {conf:.2f})" if isinstance(conf, (int, float)) else ""
        )
        lines.append(f'  - {e["rel_type"]} → "{e["target"]}"{conf_str}')
    return "\n".join(lines)


def _compute_shared_neighbors(ctx_a: dict, ctx_b: dict) -> list[str]:
    """Concepts that both A and B already point to via a typed edge."""
    targets_a = {
        e["target"] for e in ctx_a.get("out_edges", []) if e.get("target")
    }
    targets_b = {
        e["target"] for e in ctx_b.get("out_edges", []) if e.get("target")
    }
    return sorted(targets_a & targets_b)


_SOROS_VOCAB_ADVICE = (
    "Gunakan konteks graf ini sebagai bukti tambahan. Jika A sudah punya "
    "banyak hubungan `isPrerequisiteOf` dengan konsep di Bab yang sama "
    "dengan B, klasifikasi `isPrerequisiteOf` lebih mungkin benar. Tetangga "
    "bersama menandakan A dan B kemungkinan `analogousTo`.\n"
)

_LINTAS_BUKU_VOCAB_ADVICE = (
    "Gunakan konteks graf ini sebagai bukti tambahan. Tetangga bersama "
    "menandakan A dan B kemungkinan `LINTAS_BUKU_BERKAITAN_DENGAN` atau "
    "`LINTAS_BUKU_SAMA_DENGAN`. Jika hubungan yang sudah ada di graf "
    "menunjukkan A merupakan kasus khusus dari konsep yang lebih umum di "
    "buku lain (mis. lewat BAGIAN_DARI / TERDIRI_DARI), pertimbangkan "
    "`LINTAS_BUKU_APLIKASI_DARI`. Jika tipe lain tidak tepat, gunakan "
    "`LINTAS_BUKU_BERKAITAN_DENGAN` sebagai fallback.\n"
)


def _build_neighborhood_block(
    name_a: str,
    bab_a: str | None,
    out_edges_a: list[dict] | None,
    name_b: str,
    bab_b: str | None,
    out_edges_b: list[dict] | None,
    shared_neighbors: list[str] | None,
    vocab_hint: str = "soros",
) -> str:
    """Render the KONTEKS GRAF prompt section. Empty string when no signal.

    ``vocab_hint`` selects the closing advice paragraph: ``'soros'`` references
    the 3-type pedagogical vocab, ``'lintas_buku'`` references the LINTAS_BUKU_*
    cross-book vocab.
    """
    if not (bab_a or bab_b or out_edges_a or out_edges_b or shared_neighbors):
        return ""

    if bab_a and bab_b:
        bab_note = (
            "(Bab yang sama)" if bab_a == bab_b else "(Bab berbeda)"
        )
    else:
        bab_note = ""

    advice = (
        _LINTAS_BUKU_VOCAB_ADVICE
        if vocab_hint == "lintas_buku"
        else _SOROS_VOCAB_ADVICE
    )

    return (
        "========================\n"
        "KONTEKS GRAF (HUBUNGAN YANG SUDAH ADA)\n"
        "========================\n"
        f'Konsep A ("{name_a}") berada di Bab: {bab_a or "(tidak diketahui)"}\n'
        f"Hubungan A yang sudah ada di graf:\n"
        f"{_format_out_edges(out_edges_a)}\n\n"
        f'Konsep B ("{name_b}") berada di Bab: {bab_b or "(tidak diketahui)"} '
        f"{bab_note}\n"
        f"Hubungan B yang sudah ada di graf:\n"
        f"{_format_out_edges(out_edges_b)}\n\n"
        f"Tetangga bersama (target yang sama-sama dirujuk A dan B): "
        f"{_format_list(shared_neighbors)}\n\n"
        f"{advice}"
    )


def _context_fingerprint(
    ctx_a: dict, ctx_b: dict, shared_neighbors: list[str]
) -> str:
    """Stable short hash of the neighborhood payload for cache keying."""
    payload = {
        "a": {
            "bab": ctx_a.get("bab"),
            "edges": ctx_a.get("out_edges", []),
        },
        "b": {
            "bab": ctx_b.get("bab"),
            "edges": ctx_b.get("out_edges", []),
        },
        "shared": shared_neighbors,
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(blob.encode("utf-8")).hexdigest()[:12]


_classify_retry = retry(
    retry=retry_if_exception(_is_retriable_error),
    wait=wait_exponential(multiplier=2, min=10, max=120),
    stop=stop_after_attempt(5),
    reraise=True,
    before_sleep=lambda retry_state: logger.warning(
        "Classify transient error, retrying in %.1fs (attempt %d/5): %s",
        retry_state.next_action.sleep if retry_state.next_action else 10,
        retry_state.attempt_number,
        retry_state.outcome.exception() if retry_state.outcome else "unknown",
    ),
)


@_classify_retry
def classify_similar_pair(
    name_a: str,
    desc_a: str,
    name_b: str,
    desc_b: str,
    llm_model: str | None = None,
    formula_a: list[str] | None = None,
    variables_a: list[str] | None = None,
    kondisi_a: list[str] | None = None,
    formula_b: list[str] | None = None,
    variables_b: list[str] | None = None,
    kondisi_b: list[str] | None = None,
    bab_a: str | None = None,
    out_edges_a: list[dict] | None = None,
    bab_b: str | None = None,
    out_edges_b: list[dict] | None = None,
    shared_neighbors: list[str] | None = None,
    grade_a: str | None = None,
    grade_b: str | None = None,
    materi_pokok_a: str | None = None,
    materi_pokok_b: str | None = None,
    adapter: SchemaAdapter | None = None,
) -> tuple[str, float, str]:
    """Use LLM to classify the relationship between two similar konsep.

    Returns ``(rel_type, confidence, description)`` where ``rel_type`` is one
    of the labels in ``adapter.spec.classifier_vocab`` (Soros default: the
    3-type pedagogical set; Yhoga: the 5-type LINTAS_BUKU_* set). ``description``
    is the LLM's one-sentence rationale, used by the Yhoga LINTAS_BUKU pipeline
    and empty for the Soros path. Retries on 429 / 5xx / transient network errors.

    Optional neighborhood args (bab, out_edges, shared_neighbors) inject the
    KONTEKS GRAF section so the LLM can use existing typed edges as evidence.
    """
    model = llm_model or DEFAULT_CHAT_MODEL
    llm = get_llm(model)

    prompt, output_cls = _single_classifier_artifacts(adapter)
    program = LLMTextCompletionProgram.from_defaults(
        llm=llm,
        output_cls=output_cls,
        prompt_template_str=prompt,
    )

    neighborhood_block = _build_neighborhood_block(
        name_a=name_a,
        bab_a=bab_a,
        out_edges_a=out_edges_a,
        name_b=name_b,
        bab_b=bab_b,
        out_edges_b=out_edges_b,
        shared_neighbors=shared_neighbors,
        vocab_hint="lintas_buku" if _is_lintas_buku_vocab(adapter) else "soros",
    )

    if _is_lintas_buku_vocab(adapter):
        result = program(
            name_a=name_a,
            grade_a=grade_a or "(tidak diketahui)",
            desc_a=desc_a or name_a,
            materi_pokok_a=materi_pokok_a or "(tidak ada)",
            name_b=name_b,
            grade_b=grade_b or "(tidak diketahui)",
            desc_b=desc_b or name_b,
            materi_pokok_b=materi_pokok_b or "(tidak ada)",
            neighborhood_block=neighborhood_block,
        )
        return result.rel_type, float(result.confidence), result.description or ""

    result = program(
        name_a=name_a,
        desc_a=desc_a or name_a,
        formula_a=_format_list(formula_a),
        variables_a=_format_list(variables_a),
        kondisi_a=_format_list(kondisi_a),
        name_b=name_b,
        desc_b=desc_b or name_b,
        formula_b=_format_list(formula_b),
        variables_b=_format_list(variables_b),
        kondisi_b=_format_list(kondisi_b),
        neighborhood_block=neighborhood_block,
    )
    return result.rel_type, float(result.confidence), ""


# ── Batched classification ─────────────────────────────────────────

_BATCH_CLASSIFICATION_PROMPT = """\
Anda adalah ahli kurikulum yang mengklasifikasikan hubungan antar konsep pembelajaran.

Untuk SETIAP pasangan konsep berikut, klasifikasikan hubungan antara A dan B sebagai
SALAH SATU dari:
1. **isPrerequisiteOf**: A harus dipelajari sebelum B (ketergantungan urutan ketat).
   Contoh: "Bilangan Bulat" → "Persamaan Linear".
2. **supports**: A membantu pemahaman B atau A diterapkan dalam B (ketergantungan fungsional).
   Contoh: "Hukum Newton" → "Gaya Gesek".
3. **analogousTo**: A dan B memiliki pola/prinsip yang sama (hubungan struktural, anti-silo).
   Contoh: "Laju Reaksi Kimia" ↔ "Kecepatan Reaksi Biologis".
4. **none**: Tidak ada hubungan bermakna meski mirip secara semantik.

Gunakan rumus, variabel, dan kondisi untuk membedakan `supports` (berbagi konteks penerapan)
dari `analogousTo` (berbagi struktur tapi konteks berbeda).

PASANGAN:
{pairs_block}

Kembalikan JSON dengan field "items" berisi array. Setiap elemen harus memiliki:
- "pair_index": int (sesuai index pasangan di atas, dimulai dari 0)
- "rel_type": salah satu dari "isPrerequisiteOf", "supports", "analogousTo", "none"
- "confidence": skor 0.0–1.0 menyatakan keyakinan klasifikasi

WAJIB kembalikan SATU entri untuk SETIAP pasangan (total {n_pairs} entri).
"""


class ClassificationItem(BaseModel):
    """One pair's classification inside a batch response."""

    pair_index: int = Field(ge=0)
    rel_type: Literal["isPrerequisiteOf", "supports", "analogousTo", "none"]
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class ClassificationBatch(BaseModel):
    """Structured output for a batch pair classification (Soros 3-type)."""

    items: list[ClassificationItem]


# ── Yhoga LINTAS_BUKU batch classifier ──────────────────────────────────────

_LINTAS_BUKU_BATCH_CLASSIFICATION_PROMPT = """\
Anda adalah ahli kurikulum sains yang menganalisis keterkaitan **lintas-buku** antara konsep
pada Mata Pelajaran berbeda (Biologi/Fisika/Kimia Kelas XII pada Kurikulum Merdeka).

Untuk SETIAP pasangan konsep di bawah ini, klasifikasikan hubungan **lintas-buku** sebagai
SALAH SATU dari:

1. **LINTAS_BUKU_SAMA_DENGAN**: A dan B merujuk pada entitas yang sama meskipun di buku
   berbeda. Simetris. Contoh: "Energi" (Fisika) ↔ "Energi" (Kimia).
2. **LINTAS_BUKU_APLIKASI_DARI**: A adalah penerapan konsep B di disiplin lain. Asimetris.
   Contoh: "Difusi" (Biologi) ← "Gerak Brown" (Fisika).
3. **LINTAS_BUKU_PRASYARAT_UNTUK**: A pada satu MP adalah prasyarat memahami B pada MP lain.
   Asimetris.
4. **LINTAS_BUKU_MEMPERDALAM**: A memperdalam pemahaman B di buku lain. Asimetris.
   Contoh: "Termodinamika" (Fisika) → "Reaksi Endoterm" (Kimia).
5. **LINTAS_BUKU_BERKAITAN_DENGAN**: berkaitan umum lintas-buku, fallback. Simetris.
6. **none**: kemiripan hanya permukaan; tidak terkait secara curricular, atau Mata Pelajaran
   A == Mata Pelajaran B.

ATURAN: Klasifikasikan sebagai salah satu dari 5 tipe LINTAS_BUKU_* HANYA jika Mata
Pelajaran A ≠ Mata Pelajaran B. Jika sama, kembalikan "none".

================ PASANGAN KONSEP ================
{pairs_block}
=================================================

Kembalikan JSON dengan field "items": array berisi satu entri per pasangan, dimana setiap entri
punya field:
- "pair_index": int (sesuai index pasangan di atas, dimulai dari 0)
- "rel_type": salah satu dari "LINTAS_BUKU_SAMA_DENGAN", "LINTAS_BUKU_APLIKASI_DARI",
  "LINTAS_BUKU_PRASYARAT_UNTUK", "LINTAS_BUKU_MEMPERDALAM", "LINTAS_BUKU_BERKAITAN_DENGAN", "none"
- "confidence": skor 0.0–1.0
- "description": satu kalimat singkat (≤25 kata) menjelaskan alasan keterkaitan lintas-buku.
  Wajib bahasa Indonesia. Kosongkan ("") jika rel_type = "none".

WAJIB kembalikan SATU entri untuk SETIAP pasangan (total {n_pairs} entri).
"""


class LintasBukuClassificationItem(BaseModel):
    """One pair's LINTAS_BUKU classification inside a batch response."""

    pair_index: int = Field(ge=0)
    rel_type: Literal[
        "LINTAS_BUKU_SAMA_DENGAN",
        "LINTAS_BUKU_APLIKASI_DARI",
        "LINTAS_BUKU_PRASYARAT_UNTUK",
        "LINTAS_BUKU_MEMPERDALAM",
        "LINTAS_BUKU_BERKAITAN_DENGAN",
        "none",
    ]
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    description: str = Field(default="")


class LintasBukuClassificationBatch(BaseModel):
    """Structured output for a Yhoga LINTAS_BUKU batch classification."""

    items: list[LintasBukuClassificationItem]


# ── Vocab-dispatched classifier artifacts ────────────────────────────────────


def _is_lintas_buku_vocab(adapter: SchemaAdapter | None) -> bool:
    """True if the adapter's classifier_vocab is the LINTAS_BUKU 5-type set."""
    if adapter is None:
        return False
    vocab = getattr(adapter.spec, "classifier_vocab", ())
    return bool(vocab) and vocab[0].startswith("LINTAS_BUKU_")


def _single_classifier_artifacts(adapter: SchemaAdapter | None):
    """Return (prompt_str, pydantic_output_cls) for the single-pair classifier.

    Soros default unless adapter carries the LINTAS_BUKU vocab.
    """
    if _is_lintas_buku_vocab(adapter):
        return _LINTAS_BUKU_CLASSIFICATION_PROMPT, LintasBukuClassification
    return _CLASSIFICATION_PROMPT, Classification


def _batch_classifier_artifacts(adapter: SchemaAdapter | None):
    """Return (prompt_str, batch_pydantic_cls) for the batched classifier."""
    if _is_lintas_buku_vocab(adapter):
        return (
            _LINTAS_BUKU_BATCH_CLASSIFICATION_PROMPT,
            LintasBukuClassificationBatch,
        )
    return _BATCH_CLASSIFICATION_PROMPT, ClassificationBatch


def _format_pair_block(
    index: int,
    info_a: dict,
    info_b: dict,
    shared_neighbors: list[str] | None = None,
) -> str:
    """Render one pair block for the Soros 3-type batch prompt.

    info_a/info_b may carry optional `bab` and `out_edges` keys; when present
    a KONTEKS GRAF subsection is appended to the pair block.
    """
    block = (
        f"[{index}]\n"
        f"  Konsep A: \"{info_a['name']}\"\n"
        f"  Deskripsi A: \"{info_a['description'] or info_a['name']}\"\n"
        f"  Rumus A: {_format_list(info_a.get('formula'))}\n"
        f"  Variabel A: {_format_list(info_a.get('variables'))}\n"
        f"  Kondisi A: {_format_list(info_a.get('kondisi'))}\n"
        f"  Konsep B: \"{info_b['name']}\"\n"
        f"  Deskripsi B: \"{info_b['description'] or info_b['name']}\"\n"
        f"  Rumus B: {_format_list(info_b.get('formula'))}\n"
        f"  Variabel B: {_format_list(info_b.get('variables'))}\n"
        f"  Kondisi B: {_format_list(info_b.get('kondisi'))}"
    )

    neighborhood = _build_neighborhood_block(
        name_a=info_a["name"],
        bab_a=info_a.get("bab"),
        out_edges_a=info_a.get("out_edges"),
        name_b=info_b["name"],
        bab_b=info_b.get("bab"),
        out_edges_b=info_b.get("out_edges"),
        shared_neighbors=shared_neighbors,
        vocab_hint="soros",
    )
    if neighborhood:
        # Indent the block so it visually nests under the pair entry.
        indented = "\n".join("  " + line for line in neighborhood.splitlines())
        block += "\n" + indented
    return block


def _format_lintas_buku_pair_block(
    index: int,
    info_a: dict,
    info_b: dict,
    shared_neighbors: list[str] | None = None,
) -> str:
    """Render one pair block for the Yhoga LINTAS_BUKU batch prompt.

    Uses Mata Pelajaran (grade) + Materi Pokok instead of Soros's
    formula/variables/kondisi. Neighborhood block uses LINTAS_BUKU advice.
    """
    block = (
        f"[{index}]\n"
        f"  Konsep A: \"{info_a['name']}\"\n"
        f"  Mata Pelajaran A: \"{info_a.get('grade') or '(tidak diketahui)'}\"\n"
        f"  Deskripsi A: \"{info_a['description'] or info_a['name']}\"\n"
        f"  Materi Pokok A: {info_a.get('materi_pokok_ref') or '(tidak ada)'}\n"
        f"  Konsep B: \"{info_b['name']}\"\n"
        f"  Mata Pelajaran B: \"{info_b.get('grade') or '(tidak diketahui)'}\"\n"
        f"  Deskripsi B: \"{info_b['description'] or info_b['name']}\"\n"
        f"  Materi Pokok B: {info_b.get('materi_pokok_ref') or '(tidak ada)'}"
    )

    neighborhood = _build_neighborhood_block(
        name_a=info_a["name"],
        bab_a=info_a.get("bab"),
        out_edges_a=info_a.get("out_edges"),
        name_b=info_b["name"],
        bab_b=info_b.get("bab"),
        out_edges_b=info_b.get("out_edges"),
        shared_neighbors=shared_neighbors,
        vocab_hint="lintas_buku",
    )
    if neighborhood:
        indented = "\n".join("  " + line for line in neighborhood.splitlines())
        block += "\n" + indented
    return block


@_classify_retry
def _classify_batch_raw(
    pair_infos: list[tuple[dict, dict]],
    llm_model: str,
    pair_neighborhoods: list[dict] | None = None,
    adapter: SchemaAdapter | None = None,
):
    """Classify a batch of pairs in one LLM call. Retries on transient errors.

    Returns ``result.items`` from whichever batch output class the adapter's
    vocab selects (``ClassificationBatch`` for Soros, ``LintasBukuClassificationBatch``
    for Yhoga). Each item carries ``pair_index``, ``rel_type``, ``confidence``,
    and (LINTAS_BUKU only) ``description``.
    """
    llm = get_llm(llm_model)
    prompt, batch_cls = _batch_classifier_artifacts(adapter)
    program = LLMTextCompletionProgram.from_defaults(
        llm=llm,
        output_cls=batch_cls,
        prompt_template_str=prompt,
    )
    block_formatter = (
        _format_lintas_buku_pair_block
        if _is_lintas_buku_vocab(adapter)
        else _format_pair_block
    )
    blocks = []
    for i, (a, b) in enumerate(pair_infos):
        shared = (
            (pair_neighborhoods[i] or {}).get("shared_neighbors")
            if pair_neighborhoods else None
        )
        blocks.append(block_formatter(i, a, b, shared_neighbors=shared))
    pairs_block = "\n\n".join(blocks)
    result = program(
        pairs_block=pairs_block,
        n_pairs=len(pair_infos),
    )
    return result.items


def classify_pairs_batch(
    pair_infos: list[tuple[dict, dict]],
    llm_model: str | None = None,
    pair_neighborhoods: list[dict] | None = None,
    adapter: SchemaAdapter | None = None,
) -> list[tuple[str, float, str]]:
    """Classify a batch of pairs and return aligned [(rel_type, confidence, description), ...].

    `pair_infos[i]` is `(info_a, info_b)` where each info has keys:
    name, description, plus schema-specific fields (Soros: formula/variables/kondisi;
    Yhoga: grade/materi_pokok_ref), and optionally bab + out_edges (KONTEKS GRAF).

    `pair_neighborhoods[i]` is an optional dict with shared_neighbors per pair.

    The LLM returns items keyed by pair_index; we realign to the input order.
    Missing indices fall back to ("none", 0.0, ""). The third element is the
    LLM's one-sentence rationale (LINTAS_BUKU only; "" for Soros).
    """
    if not pair_infos:
        return []
    model = llm_model or DEFAULT_CHAT_MODEL
    items = _classify_batch_raw(pair_infos, model, pair_neighborhoods, adapter=adapter)

    by_index: dict[int, tuple[str, float, str]] = {
        it.pair_index: (
            it.rel_type,
            float(it.confidence),
            getattr(it, "description", "") or "",
        )
        for it in items
    }
    out: list[tuple[str, float, str]] = []
    for i in range(len(pair_infos)):
        out.append(by_index.get(i, ("none", 0.0, "")))
    return out


def _classification_cache_text(
    source: str,
    target: str,
    fingerprint: str = "",
    schema_name: str = "",
) -> str:
    """Canonical cache key input — order-invariant so (A,B) and (B,A) share a key.

    When `fingerprint` is set (a hash of the neighborhood payload fed to the
    classifier), it is folded into the key so that re-runs after the graph
    gains new typed edges produce a cache miss and re-classify.

    `schema_name` distinguishes Soros vs Yhoga classifications so the same
    pair name running under different vocabularies does not collide.
    """
    a, b = sorted([source, target])
    parts = [a, b]
    if schema_name:
        parts.append(f"schema:{schema_name}")
    if fingerprint:
        parts.append(f"ctx:{fingerprint}")
    return "||".join(parts)


def classify_similar_pairs(
    driver,
    pairs: list[dict],
    llm_model: str | None = None,
    progress_callback: Callable[[int, int, str], None] | None = None,
    use_cache: bool = True,
    batch_size: int = 1,
    adapter: SchemaAdapter | None = None,
) -> list[dict]:
    """Classify SIMILAR_TO pairs into typed relationships using LLM.

    For each pair, looks up descriptions and schema-specific fields (Soros:
    formula/variables/kondisi; Yhoga: grade/materi_pokok_ref) from Neo4j and
    calls the LLM to classify the relationship into the adapter's classifier
    vocabulary (Soros 3-type: isPrerequisiteOf/supports/analogousTo/none;
    Yhoga 5-type LINTAS_BUKU_*/none). Results include a confidence score in
    [0, 1] and, for Yhoga, a one-sentence Indonesian rationale (description).

    Classifications are cached on disk by (pair, schema, llm_model,
    prompt_version) so re-runs on the same pairs do not re-bill the LLM.
    Failures are NOT cached and surfaced in the final progress message so the
    user can rerun to retry.

    Args:
        driver: Neo4j driver (for fetching descriptions)
        pairs: List of {"source": str, "target": str, "similarity": float}
        llm_model: LLM model string
        progress_callback: Optional progress callback
        use_cache: Read/write classification cache (default True)
        batch_size: Number of pairs classified per LLM call. Default 1
            (per-pair mode, preserves old behavior). Values >1 batch pairs into
            one LLM request — much fewer round-trips but slightly more fragile
            per call (if the batch errors, the whole batch retries as a unit).
        adapter: SchemaAdapter selecting node labels, properties to read, and
            classifier vocabulary. Defaults to SorosAdapter().

    Returns list of dicts with source, target, similarity, rel_type, confidence,
    description (description is "" for Soros, one-sentence Indonesian text for Yhoga).
    """
    if adapter is None:
        adapter = SorosAdapter()
    total = len(pairs)
    results: list[dict | None] = [None] * total
    model = llm_model or DEFAULT_CHAT_MODEL
    batch_size = max(1, int(batch_size))

    names = list({p["source"] for p in pairs} | {p["target"] for p in pairs})
    node_map = adapter.get_classification_data(driver, names)

    # Neighborhood: parent (bab/chapter) + existing typed out-edges per concept.
    contexts = adapter.get_concept_context(
        driver, names, max_edges=NEIGHBORHOOD_MAX_EDGES
    )

    # Per-pair derived signals — computed once, reused for cache key and prompt.
    pair_ctx: list[dict] = []
    for pair in pairs:
        ctx_a = contexts.get(pair["source"], {"bab": None, "out_edges": []})
        ctx_b = contexts.get(pair["target"], {"bab": None, "out_edges": []})
        shared = _compute_shared_neighbors(ctx_a, ctx_b)
        pair_ctx.append(
            {
                "ctx_a": ctx_a,
                "ctx_b": ctx_b,
                "shared": shared,
                "fingerprint": _context_fingerprint(ctx_a, ctx_b, shared),
            }
        )

    cache_hits = 0
    uncached_indices: list[int] = []

    schema_name = getattr(adapter.spec, "name", "")

    for i, pair in enumerate(pairs):
        source, target = pair["source"], pair["target"]
        cached = None
        if use_cache:
            cache_text = _classification_cache_text(
                source,
                target,
                fingerprint=pair_ctx[i]["fingerprint"],
                schema_name=schema_name,
            )
            cached = cache_manager.get_classification(
                cache_text, model, CLASSIFICATION_PROMPT_VERSION
            )
        if cached is not None:
            results[i] = {
                **pair,
                "rel_type": cached.get("rel_type", "none"),
                "confidence": float(cached.get("confidence", 0.0)),
                "description": cached.get("description", ""),
            }
            cache_hits += 1
        else:
            uncached_indices.append(i)

    if progress_callback and cache_hits:
        progress_callback(
            cache_hits, total, f"Loaded {cache_hits}/{total} from cache"
        )

    def _info_for(name: str, ctx: dict | None = None) -> dict:
        info = node_map.get(name, {})
        ctx = ctx or {}
        return {
            "name": name,
            "description": info.get("description", ""),
            "formula": info.get("formula", []),
            "variables": info.get("variables", []),
            "kondisi": info.get("kondisi", []),
            "grade": info.get("grade", ""),
            "materi_pokok_ref": info.get("materi_pokok_ref", ""),
            "bab": ctx.get("bab"),
            "out_edges": ctx.get("out_edges", []),
        }

    failed_count = 0
    processed = cache_hits

    for batch_start in range(0, len(uncached_indices), batch_size):
        batch_idx = uncached_indices[batch_start : batch_start + batch_size]
        batch_pairs = [pairs[j] for j in batch_idx]
        pair_infos = [
            (
                _info_for(p["source"], pair_ctx[j]["ctx_a"]),
                _info_for(p["target"], pair_ctx[j]["ctx_b"]),
            )
            for j, p in zip(batch_idx, batch_pairs)
        ]
        batch_neighborhoods = [
            {"shared_neighbors": pair_ctx[j]["shared"]} for j in batch_idx
        ]

        if progress_callback:
            end = processed + len(batch_idx)
            progress_callback(
                processed,
                total,
                f"Classifying pairs {processed + 1}–{end}/{total}"
                + (f" (batch {len(batch_idx)})" if batch_size > 1 else ""),
            )

        try:
            if batch_size == 1:
                info_a, info_b = pair_infos[0]
                rel_type, confidence, description = classify_similar_pair(
                    name_a=info_a["name"],
                    desc_a=info_a["description"],
                    name_b=info_b["name"],
                    desc_b=info_b["description"],
                    llm_model=model,
                    formula_a=info_a["formula"],
                    variables_a=info_a["variables"],
                    kondisi_a=info_a["kondisi"],
                    formula_b=info_b["formula"],
                    variables_b=info_b["variables"],
                    kondisi_b=info_b["kondisi"],
                    bab_a=info_a.get("bab"),
                    out_edges_a=info_a.get("out_edges"),
                    bab_b=info_b.get("bab"),
                    out_edges_b=info_b.get("out_edges"),
                    shared_neighbors=batch_neighborhoods[0]["shared_neighbors"],
                    grade_a=info_a.get("grade"),
                    grade_b=info_b.get("grade"),
                    materi_pokok_a=info_a.get("materi_pokok_ref"),
                    materi_pokok_b=info_b.get("materi_pokok_ref"),
                    adapter=adapter,
                )
                outcomes = [(rel_type, confidence, description)]
            else:
                outcomes = classify_pairs_batch(
                    pair_infos,
                    llm_model=model,
                    pair_neighborhoods=batch_neighborhoods,
                    adapter=adapter,
                )
        except Exception as e:
            logger.warning(
                "Batch classify failed for pairs %d–%d (%d pairs): %s",
                batch_idx[0],
                batch_idx[-1],
                len(batch_idx),
                e,
            )
            # Don't poison the cache with failures — leave them uncached and
            # surfaced as failed so a rerun retries them.
            for j, pair in zip(batch_idx, batch_pairs):
                results[j] = {
                    **pair,
                    "rel_type": "none",
                    "confidence": 0.0,
                    "description": "",
                }
            failed_count += len(batch_idx)
            processed += len(batch_idx)
            continue

        for j, pair, (rel_type, confidence, description) in zip(
            batch_idx, batch_pairs, outcomes
        ):
            results[j] = {
                **pair,
                "rel_type": rel_type,
                "confidence": confidence,
                "description": description,
            }
            if use_cache:
                cache_text = _classification_cache_text(
                    pair["source"],
                    pair["target"],
                    fingerprint=pair_ctx[j]["fingerprint"],
                    schema_name=schema_name,
                )
                cache_manager.put_classification(
                    cache_text,
                    model,
                    CLASSIFICATION_PROMPT_VERSION,
                    {
                        "rel_type": rel_type,
                        "confidence": confidence,
                        "description": description,
                    },
                )

        processed += len(batch_idx)

    # Belt-and-suspenders: should never trigger given the loop above.
    final_results: list[dict] = []
    for i, r in enumerate(results):
        if r is None:
            final_results.append(
                {
                    **pairs[i],
                    "rel_type": "none",
                    "confidence": 0.0,
                    "description": "",
                }
            )
        else:
            final_results.append(r)

    if progress_callback:
        msg = f"Classified {total} pairs ({cache_hits} cached"
        if failed_count:
            msg += f", {failed_count} failed — rerun to retry"
        msg += ")"
        progress_callback(total, total, msg)

    logger.info(
        "Classification done: total=%d, cache_hits=%d, failed=%d, batch_size=%d, model=%s",
        total,
        cache_hits,
        failed_count,
        batch_size,
        model,
    )

    return final_results


# ── LINTAS_BUKU JSON staging dump ────────────────────────────────────────────


def _yhoga_instance_counts(driver) -> dict:
    """Snapshot node-label counts + total relationships for the dump header."""
    counts: dict = {}
    with driver.session() as session:
        for r in session.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS n"
        ):
            label = r["label"]
            if label:
                counts[label] = r["n"]
        total = session.run(
            "MATCH ()-[r]->() RETURN count(r) AS n"
        ).single()
        counts["total_relationships"] = total["n"] if total else 0
    return counts


def dump_lintas_buku_results(
    results: list[dict],
    output_path,
    driver=None,
    *,
    version: str = "ann-classifier-v1",
    source_state: str = "extraction-v2-reviewed",
    method: str = "ann + classifier",
    params: dict | None = None,
    captured_from: str = "Yhoga Neo4j Aura Free instance",
    overwrite: bool = False,
) -> int:
    """Write a friend-llm-shaped JSON dump of classified LINTAS_BUKU_* edges.

    Filters out ``rel_type == "none"`` results (they're not edges). Mirrors
    the schema in ``experiments/knowledge_graph_states/completion-experiments/
    friend-llm/lintas_buku_edges.json`` so the two completion experiments are
    text-diffable.

    **Refuses to overwrite an existing file by default** — pass ``overwrite=True``
    to opt in. This protects the canonical ``lintas_buku_edges.json`` pointer
    against silent clobbering during sweep workflows where the staging path is
    typically a per-iteration audit file (``lintas_buku_edges.tNN_kMM.json``).

    Args:
        results: Output of ``classify_similar_pairs`` (each dict carries
            source, target, rel_type, confidence, description, plus optional
            source_grade/target_grade from ``find_similar_pairs_ann``).
        output_path: Filesystem path (str or Path) to write JSON to.
        driver: Optional Neo4j driver for snapshotting instance counts at
            capture time. When None, the field is omitted.
        version: Methodology slug (default 'ann-classifier-v1').
        source_state: Ingestion state this completion ran against.
        method: Free-text method label.
        params: Hyperparameter dict (embed_model, chat_model, threshold,
            top_k, scope). Pass-through for audit.
        captured_from: Free-text source label.
        overwrite: If False (default), raises ``FileExistsError`` when the
            target path already exists. Set True to clobber.

    Returns the number of LINTAS_BUKU_* edges written.

    Raises:
        FileExistsError: ``output_path`` already exists and ``overwrite`` is False.
    """
    from datetime import date
    from pathlib import Path

    out_path = Path(output_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {out_path}. "
            f"Pass overwrite=True to clobber, or pick a different "
            f"staging filename (convention: lintas_buku_edges.tNN_kMM.json)."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    real_edges = [
        r for r in results
        if r.get("rel_type") and r["rel_type"] != "none"
    ]

    breakdown: dict[str, int] = {}
    edges_json: list[dict] = []
    for r in real_edges:
        rel_type = r["rel_type"]
        breakdown[rel_type] = breakdown.get(rel_type, 0) + 1
        suffix = (
            rel_type.removeprefix("LINTAS_BUKU_")
            if rel_type.startswith("LINTAS_BUKU_")
            else rel_type
        )
        edges_json.append(
            {
                "rel_type": rel_type,
                "source_label": "Concept",
                "source_name": r["source"],
                "source_grade": r.get("source_grade", ""),
                "target_label": "Concept",
                "target_name": r["target"],
                "target_grade": r.get("target_grade", ""),
                "properties": {
                    "description": r.get("description", ""),
                    "relation_type": suffix,
                    "confidence": float(r.get("confidence", 0.0)),
                    "method": version,
                },
            }
        )

    doc: dict = {
        "version": version,
        "source_state": source_state,
        "date_captured": date.today().isoformat(),
        "captured_from": captured_from,
        "method": method,
        "params": params or {},
        "edge_count": len(edges_json),
        "edge_type_breakdown": dict(
            sorted(breakdown.items(), key=lambda kv: -kv[1])
        ),
        "edges": edges_json,
    }

    if driver is not None:
        try:
            doc["instance_counts_at_capture"] = _yhoga_instance_counts(driver)
        except Exception as e:
            logger.warning("Could not snapshot instance counts: %s", e)

    out_path.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "Wrote %d LINTAS_BUKU_* edges across %d types to %s",
        len(edges_json),
        len(breakdown),
        out_path,
    )
    return len(edges_json)
