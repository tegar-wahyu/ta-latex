"""Schema adapters: the runtime that executes schema-specific Cypher.

The adapter abstracts the parts of the completion pipeline that depend on the
target KG ontology — node labels, vector index names, traversals to find a
concept's structural parent (Bab vs. Chapter), and the property names read for
embedding text and classifier context.

Two implementations live here:

- `SorosAdapter` targets the project-native ontology
  (`Konsep`/`SubKonsep`/`Bab`/`Document`/...) and delegates most work to the
  existing `src.graph` helpers. Backward-compat for callers that don't pass an
  adapter to `completion.py`.
- `YhogaAdapter` targets the curriculum ontology used by the `yhoga` Aura
  instance (`Concept`/`Subtopic`/`Chapter`/`Grade`). All its Cypher is inline
  and self-contained.

The completion module (`src.completion`) takes an adapter and never inspects
the spec directly — extending support to a third schema means adding one
`SchemaSpec` and one adapter class, with no changes to `completion.py`.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod

from neo4j import GraphDatabase

from src.connection import Neo4jConnection, resolve_connection
from src.schema_spec import SchemaSpec, SOROS_SCHEMA, YHOGA_SCHEMA

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Base
# ──────────────────────────────────────────────────────────────────────────────


class SchemaAdapter(ABC):
    """Schema-specific operations consumed by `src.completion`.

    Schema (ontology shape) and Neo4j connection (which DB instance) are
    independently configurable. A subclass instance can be constructed with an
    explicit ``connection`` to run e.g. the Yhoga ontology against the project's
    own Neo4j upstream, or vice versa. With no explicit connection, the adapter
    falls back to the ``NEO4J_TARGET`` env var if set, and otherwise to the
    spec's own env-var names — the legacy "yhoga schema implies yhoga DB"
    behavior, kept for backward compatibility.
    """

    spec: SchemaSpec
    connection: Neo4jConnection | None

    def __init__(self, connection: Neo4jConnection | None = None) -> None:
        # Explicit > NEO4J_TARGET env > spec defaults (None means "fall back").
        self.connection = resolve_connection(connection)

    # --- Connection ---------------------------------------------------------

    def get_driver(self):
        """Return a Neo4j driver bound to the active connection.

        If an explicit ``Neo4jConnection`` was passed in (or NEO4J_TARGET is
        set), that wins. Otherwise we resolve from the spec's env-var names —
        preserving prior single-DB-per-schema behavior.
        """
        if self.connection is not None:
            return GraphDatabase.driver(
                self.connection.uri,
                auth=(self.connection.user, self.connection.password),
            )
        uri = os.getenv(self.spec.neo4j_uri_env)
        user = os.getenv(self.spec.neo4j_user_env, "neo4j")
        pwd = os.getenv(self.spec.neo4j_password_env)
        if not uri or not pwd:
            raise ValueError(
                f"{self.spec.neo4j_uri_env} and {self.spec.neo4j_password_env} "
                f"must be set for schema '{self.spec.name}' (or set NEO4J_TARGET / "
                f"pass an explicit Neo4jConnection)"
            )
        return GraphDatabase.driver(uri, auth=(user, pwd))

    def connection_label(self) -> str:
        """Short label describing which DB this adapter is currently targeting."""
        if self.connection is not None:
            return self.connection.label
        # Reverse-lookup from env-var name for the spec default.
        if self.spec.neo4j_uri_env == "NEO4J_URI":
            return "default"
        if self.spec.neo4j_uri_env == "NEO4J_URI_YHOGA":
            return "yhoga"
        return self.spec.neo4j_uri_env

    # --- Embedding text -----------------------------------------------------

    def build_embed_text(self, node: dict) -> str:
        """Default: name + description + each extra field, comma-joined if list."""
        desc = node.get(self.spec.description_field) or node.get("description") or node["name"]
        parts = [f"{node['name']}. {desc}"]
        for field in self.spec.embed_extra_fields:
            val = node.get(field)
            if not val:
                continue
            if isinstance(val, list):
                if val:
                    parts.append(f"{field.capitalize()}: {', '.join(str(x) for x in val)}")
            else:
                parts.append(f"{field.capitalize()}: {val}")
        return " ".join(parts)

    # --- Vector index lifecycle --------------------------------------------

    @abstractmethod
    def create_vector_indexes(self, driver, dimensions: int) -> bool: ...

    @abstractmethod
    def drop_vector_indexes(self, driver) -> bool: ...

    @abstractmethod
    def get_index_info(self, driver) -> dict: ...

    @abstractmethod
    def count_nodes_with_embeddings(self, driver) -> dict: ...

    # --- Embedding storage / lookup ----------------------------------------

    @abstractmethod
    def store_node_embeddings(
        self, driver, nodes: list[dict], embeddings: list[list[float]], model: str
    ) -> int: ...

    @abstractmethod
    def get_nodes_with_current_embedding(
        self, driver, names: list[str], model: str
    ) -> set[str]: ...

    @abstractmethod
    def query_similar_nodes(
        self,
        driver,
        query_embedding: list[float],
        k: int,
        threshold: float,
        exclude_name: str | None,
    ) -> list[dict]: ...

    # --- Node / context fetch ----------------------------------------------

    @abstractmethod
    def get_nodes_for_completion(
        self,
        driver,
        document_name: str | None = None,
        include_cross_doc: bool = False,
    ) -> list[dict]:
        """Return concept nodes with all fields needed for embedding + classification.

        Each dict has at least: name, description, labels, plus every field
        listed in `spec.embed_extra_fields`.
        """

    @abstractmethod
    def get_classification_data(self, driver, names: list[str]) -> dict[str, dict]:
        """Return `{name: {description, ...embed_extra_fields}}` for the prompt."""

    @abstractmethod
    def get_concept_context(
        self, driver, names: list[str], max_edges: int = 5
    ) -> dict[str, dict]:
        """Return `{name: {parent: str|None, out_edges: [{rel_type, target, confidence}]}}`.

        `parent` is the structural container (Bab in soros, Chapter in yhoga) used
        in the KONTEKS GRAF prompt section. `out_edges` are existing typed
        out-relationships from `spec.existing_typed_rels`, sorted by confidence
        descending and capped at `max_edges`.
        """

    # --- UI helpers --------------------------------------------------------

    @abstractmethod
    def get_documents(self, driver) -> list[dict]:
        """Return the list of "documents" (soros: Document; yhoga: Grade) for
        the scope selector. Each dict has at least `name` and `topic_count`.
        """

    @abstractmethod
    def get_total_concept_count(self, driver) -> int:
        """Total concept-level nodes — used to gate the page on empty graphs."""

    @abstractmethod
    def get_similar_pairs(self, driver) -> list[dict]:
        """Existing SIMILAR_TO pairs (input to classification)."""

    # --- Persistence of completion output ----------------------------------

    @abstractmethod
    def save_similar_pair(
        self, driver, source: str, target: str, score: float
    ) -> None: ...

    @abstractmethod
    def save_typed_relationship(
        self,
        driver,
        source: str,
        target: str,
        rel_type: str,
        confidence: float | None,
    ) -> None: ...

    # --- Pair filters ------------------------------------------------------

    def get_pairs_with_existing_typed_edges(
        self, driver, pairs: list[dict]
    ) -> set[tuple[str, str]]:
        """Return the subset of candidate pairs that already have a non-SIMILAR_TO
        edge between source and target.

        Pair key in the returned set is the sorted ``(name_a, name_b)`` tuple,
        matching the dedup key used by ``find_similar_pairs_ann``. Default
        implementation returns ``set()``; adapters that key Concepts by more
        than name (e.g. Yhoga: name + grade) override this with a schema-aware
        Cypher batch.
        """
        return set()


# ──────────────────────────────────────────────────────────────────────────────
# Soros adapter — delegates to existing src.graph helpers
# ──────────────────────────────────────────────────────────────────────────────


class SorosAdapter(SchemaAdapter):
    spec = SOROS_SCHEMA

    def build_embed_text(self, node: dict) -> str:
        # Preserve the Indonesian labels the existing soros embeddings were
        # produced with — changing them would invalidate every cached vector.
        text = f"{node['name']}. {node.get('description', node['name'])}"
        if node.get("formula"):
            text += f" Rumus: {', '.join(node['formula'])}"
        if node.get("variables"):
            text += f" Variabel: {', '.join(node['variables'])}"
        if node.get("kondisi"):
            text += f" Kondisi: {', '.join(node['kondisi'])}"
        return text

    def create_vector_indexes(self, driver, dimensions):
        from src.graph import create_vector_index

        return create_vector_index(driver, dimensions=dimensions)

    def drop_vector_indexes(self, driver):
        from src.graph import drop_vector_index

        return drop_vector_index(driver)

    def get_index_info(self, driver):
        from src.graph import get_index_info

        return get_index_info(driver)

    def count_nodes_with_embeddings(self, driver):
        from src.graph import count_nodes_with_embeddings

        return count_nodes_with_embeddings(driver)

    def store_node_embeddings(self, driver, nodes, embeddings, model):
        from src.graph import store_node_embeddings

        return store_node_embeddings(driver, nodes, embeddings, model)

    def get_nodes_with_current_embedding(self, driver, names, model):
        from src.graph import get_nodes_with_current_embedding

        return get_nodes_with_current_embedding(driver, names, model)

    def query_similar_nodes(self, driver, query_embedding, k, threshold, exclude_name):
        from src.graph import query_similar_nodes

        return query_similar_nodes(
            driver,
            query_embedding=query_embedding,
            k=k,
            threshold=threshold,
            exclude_name=exclude_name,
        )

    def get_nodes_for_completion(self, driver, document_name=None, include_cross_doc=False):
        from src.graph import get_nodes_with_descriptions

        return get_nodes_with_descriptions(
            driver,
            document_name=document_name,
            include_cross_doc=include_cross_doc,
        )

    def get_classification_data(self, driver, names):
        if not names:
            return {}
        with driver.session() as session:
            result = session.run(
                """
                MATCH (n) WHERE n.name IN $names AND (n:Konsep OR n:SubKonsep)
                RETURN n.name AS name,
                       n.description AS description,
                       coalesce(n.formula, []) AS formula,
                       coalesce(n.variables, []) AS variables,
                       coalesce(n.kondisi, []) AS kondisi
                """,
                names=names,
            )
            out: dict[str, dict] = {}
            for r in result:
                out[r["name"]] = {
                    "description": r["description"] or "",
                    "formula": list(r["formula"]) if r["formula"] else [],
                    "variables": list(r["variables"]) if r["variables"] else [],
                    "kondisi": list(r["kondisi"]) if r["kondisi"] else [],
                }
            return out

    def get_concept_context(self, driver, names, max_edges=5):
        from src.graph import get_concept_context

        return get_concept_context(driver, names, max_edges=max_edges)

    def save_similar_pair(self, driver, source, target, score):
        src, tgt = sorted([source, target])
        with driver.session() as session:
            session.run(
                "MATCH (a:Konsep|SubKonsep {name: $src}), "
                "(b:Konsep|SubKonsep {name: $tgt}) "
                "MERGE (a)-[r:SIMILAR_TO]->(b) "
                "SET r.score = $score",
                src=src,
                tgt=tgt,
                score=score,
            )

    def save_typed_relationship(self, driver, source, target, rel_type, confidence):
        from src.graph import create_typed_relationship

        create_typed_relationship(
            driver, source=source, target=target, rel_type=rel_type, confidence=confidence
        )

    def get_documents(self, driver):
        from src.graph import get_all_documents

        return get_all_documents(driver)

    def get_total_concept_count(self, driver):
        from src.graph import get_graph_stats

        stats = get_graph_stats(driver)
        return stats.get("konsep", stats.get("topics", 0)) or 0

    def get_similar_pairs(self, driver):
        from src.graph import get_similar_relationships

        return get_similar_relationships(driver)


# ──────────────────────────────────────────────────────────────────────────────
# Yhoga adapter — Concept / Subtopic / Chapter / Grade ontology
# ──────────────────────────────────────────────────────────────────────────────


class YhogaAdapter(SchemaAdapter):
    spec = YHOGA_SCHEMA

    def build_embed_text(self, node: dict) -> str:
        text = f"{node['name']}. {node.get('description') or node['name']}"
        if node.get("materi_pokok_ref"):
            text += f" Materi Pokok: {node['materi_pokok_ref']}"
        if node.get("grade"):
            text += f" Mata Pelajaran: {node['grade']}"
        return text

    def create_vector_indexes(self, driver, dimensions):
        try:
            with driver.session() as session:
                for label, idx_name in self.spec.index_names.items():
                    existing = session.run(
                        "SHOW INDEXES YIELD name, options "
                        "WHERE name = $name "
                        "RETURN options.indexConfig.`vector.dimensions` AS dim",
                        name=idx_name,
                    ).single()
                    if existing is not None and existing["dim"] != dimensions:
                        logger.warning(
                            "Dropping stale %s (was %d-dim, need %d-dim)",
                            idx_name,
                            existing["dim"],
                            dimensions,
                        )
                        session.run(f"DROP INDEX {idx_name} IF EXISTS")
                    session.run(
                        f"""
                        CREATE VECTOR INDEX {idx_name} IF NOT EXISTS
                        FOR (n:{label}) ON n.embedding
                        OPTIONS {{
                            indexConfig: {{
                                `vector.dimensions`: {dimensions},
                                `vector.similarity_function`: 'cosine'
                            }}
                        }}
                        """
                    )
            logger.info("Yhoga vector indexes ready at %d dim", dimensions)
            return True
        except Exception as e:
            logger.error("Yhoga create_vector_indexes failed: %s", e)
            return False

    def drop_vector_indexes(self, driver):
        try:
            with driver.session() as session:
                for idx_name in self.spec.index_names.values():
                    session.run(f"DROP INDEX {idx_name} IF EXISTS")
            return True
        except Exception as e:
            logger.error("Yhoga drop_vector_indexes failed: %s", e)
            return False

    def get_index_info(self, driver):
        try:
            with driver.session() as session:
                idx_names = list(self.spec.index_names.values())
                result = session.run(
                    "SHOW INDEXES YIELD name, type, state, indexProvider "
                    "WHERE name IN $names "
                    "RETURN name, type, state, indexProvider",
                    names=idx_names,
                )
                indexes = [dict(r) for r in result]
                if indexes:
                    return {
                        "exists": True,
                        "indexes": indexes,
                        "state": indexes[0].get("state", "UNKNOWN"),
                    }
                return {"exists": False}
        except Exception as e:
            logger.error("Yhoga get_index_info failed: %s", e)
            return {"exists": False, "error": str(e)}

    def count_nodes_with_embeddings(self, driver):
        # Only :Concept participates in completion for yhoga (Subtopic is purely
        # structural and has no description property).
        with driver.session() as session:
            result = session.run(
                """
                MATCH (n:Concept)
                WITH count(n) AS total,
                     sum(CASE WHEN n.embedding IS NOT NULL THEN 1 ELSE 0 END) AS with_embedding,
                     collect(DISTINCT n.embedding_model) AS models
                RETURN total, with_embedding, models
                """
            )
            r = result.single()
            if r:
                return {
                    "total": r["total"],
                    "with_embedding": r["with_embedding"],
                    "models": [m for m in r["models"] if m],
                }
            return {"total": 0, "with_embedding": 0, "models": []}

    def store_node_embeddings(self, driver, nodes, embeddings, model):
        updated = 0
        with driver.session() as session:
            for node, emb in zip(nodes, embeddings):
                session.run(
                    """
                    MATCH (n:Concept {name: $name})
                    SET n.embedding = $embedding, n.embedding_model = $model
                    """,
                    name=node["name"],
                    embedding=emb,
                    model=model,
                )
                updated += 1
        logger.info("Stored %d yhoga Concept embeddings (model: %s)", updated, model)
        return updated

    def get_nodes_with_current_embedding(self, driver, names, model):
        if not names:
            return set()
        with driver.session() as session:
            result = session.run(
                """
                MATCH (n:Concept)
                WHERE n.name IN $names
                  AND n.embedding IS NOT NULL
                  AND n.embedding_model = $model
                RETURN n.name AS name
                """,
                names=names,
                model=model,
            )
            return {r["name"] for r in result}

    def query_similar_nodes(self, driver, query_embedding, k, threshold, exclude_name):
        idx_name = self.spec.index_names["Concept"]
        with driver.session() as session:
            result = session.run(
                f"""
                CALL db.index.vector.queryNodes('{idx_name}', $k, $embedding)
                YIELD node, score
                WHERE score >= $threshold
                  AND ($exclude_name IS NULL OR node.name <> $exclude_name)
                RETURN node.name AS name, node.description AS description,
                       score, labels(node) AS labels
                ORDER BY score DESC
                """,
                embedding=query_embedding,
                k=k,
                threshold=threshold,
                exclude_name=exclude_name,
            )
            return [dict(r) for r in result]

    def get_nodes_for_completion(self, driver, document_name=None, include_cross_doc=False):
        # Yhoga has no Document concept — `document_name` is interpreted as a
        # Grade name when provided. `include_cross_doc=True` drops the filter.
        with driver.session() as session:
            if document_name and not include_cross_doc:
                result = session.run(
                    """
                    MATCH (g:Grade {name: $grade})-[:HAS_CHAPTER]->
                          (:Chapter)-[:HAS_SUBTOPIC]->(:Subtopic)-[:HAS_CONCEPT]->(n:Concept)
                    RETURN n.name AS name,
                           coalesce(n.description, '') AS description,
                           labels(n) AS labels,
                           coalesce(n.materi_pokok_ref, '') AS materi_pokok_ref,
                           coalesce(n.grade, g.name) AS grade,
                           g.name AS doc_name
                    ORDER BY n.name
                    """,
                    grade=document_name,
                )
            else:
                result = session.run(
                    """
                    MATCH (n:Concept)
                    OPTIONAL MATCH (g:Grade)-[:HAS_CHAPTER]->
                          (:Chapter)-[:HAS_SUBTOPIC]->(:Subtopic)-[:HAS_CONCEPT]->(n)
                    WITH n, head(collect(DISTINCT g.name)) AS grade_name
                    RETURN n.name AS name,
                           coalesce(n.description, '') AS description,
                           labels(n) AS labels,
                           coalesce(n.materi_pokok_ref, '') AS materi_pokok_ref,
                           coalesce(n.grade, grade_name, '') AS grade,
                           grade_name AS doc_name
                    ORDER BY n.name
                    """
                )
            nodes = []
            seen = set()
            for r in result:
                name = r["name"]
                if name in seen:
                    continue
                seen.add(name)
                nodes.append(
                    {
                        "name": name,
                        "description": r["description"] or name,
                        "labels": r["labels"],
                        "document": r.get("doc_name"),
                        "materi_pokok_ref": r["materi_pokok_ref"],
                        "grade": r["grade"],
                    }
                )
            return nodes

    def get_classification_data(self, driver, names):
        if not names:
            return {}
        with driver.session() as session:
            result = session.run(
                """
                MATCH (n:Concept) WHERE n.name IN $names
                RETURN n.name AS name,
                       coalesce(n.description, '') AS description,
                       coalesce(n.materi_pokok_ref, '') AS materi_pokok_ref,
                       coalesce(n.grade, '') AS grade
                """,
                names=names,
            )
            out: dict[str, dict] = {}
            for r in result:
                out[r["name"]] = {
                    "description": r["description"] or "",
                    "materi_pokok_ref": r["materi_pokok_ref"],
                    "grade": r["grade"],
                    # Keep the soros-style fields present-but-empty so the
                    # shared classifier prompt slots accept them without branches.
                    "formula": [],
                    "variables": [],
                    "kondisi": [],
                }
            return out

    def get_concept_context(self, driver, names, max_edges=5):
        if not names:
            return {}
        rel_pattern = "|".join(self.spec.existing_typed_rels)
        contexts: dict[str, dict] = {}
        with driver.session() as session:
            result = session.run(
                f"""
                MATCH (n:Concept) WHERE n.name IN $names
                OPTIONAL MATCH (n)-[r:{rel_pattern}]->(t)
                WITH n, collect(DISTINCT {{
                        rel_type: type(r),
                        target: coalesce(t.name, ''),
                        confidence: r.confidence
                     }}) AS edges
                OPTIONAL MATCH (ch:Chapter)-[:HAS_SUBTOPIC|HAS_CONCEPT*1..2]->(n)
                WITH n, edges, head(collect(DISTINCT ch.name)) AS parent
                RETURN n.name AS name, parent,
                       [e IN edges WHERE e.target <> ''] AS out_edges
                """,
                names=names,
            )
            for r in result:
                edges = sorted(
                    r["out_edges"],
                    key=lambda e: (e.get("confidence") or 0.0),
                    reverse=True,
                )[:max_edges]
                contexts[r["name"]] = {
                    "bab": r["parent"],  # key kept as 'bab' so the prompt builder is shared
                    "out_edges": edges,
                }
        for name in names:
            contexts.setdefault(name, {"bab": None, "out_edges": []})
        return contexts

    def save_similar_pair(self, driver, source, target, score):
        src, tgt = sorted([source, target])
        with driver.session() as session:
            session.run(
                """
                MATCH (a:Concept {name: $src}), (b:Concept {name: $tgt})
                MERGE (a)-[r:SIMILAR_TO]->(b)
                SET r.score = $score
                """,
                src=src,
                tgt=tgt,
                score=score,
            )

    # Closed cross-book vocabulary per docs/yhoga-ontology.ttl:151-185.
    LINTAS_BUKU_TYPES = frozenset(
        {
            "LINTAS_BUKU_SAMA_DENGAN",
            "LINTAS_BUKU_APLIKASI_DARI",
            "LINTAS_BUKU_PRASYARAT_UNTUK",
            "LINTAS_BUKU_MEMPERDALAM",
            "LINTAS_BUKU_BERKAITAN_DENGAN",
        }
    )

    def save_typed_relationship(
        self,
        driver,
        source,
        target,
        rel_type,
        confidence,
        description: str = "",
        method: str = "ann-classifier-v1",
        source_grade: str | None = None,
        target_grade: str | None = None,
    ):
        """MERGE a LINTAS_BUKU_<type> edge between two Concepts on Yhoga.

        Matches Concepts by ``(name, grade)`` when grades are provided; this is
        important because Yhoga keys Concepts as ``(name, grade)`` and a name
        can in principle appear under multiple grades. Falls back to name-only
        match when grades are omitted.

        Properties set on the edge:
            description: 1-sentence LLM rationale (LINTAS_BUKU only)
            confidence: float in [0, 1]
            method: identifier of the completion run (default 'ann-classifier-v1')
        """
        if rel_type not in self.LINTAS_BUKU_TYPES:
            raise ValueError(
                f"rel_type must be one of {sorted(self.LINTAS_BUKU_TYPES)}; "
                f"got {rel_type!r}"
            )
        if source_grade and target_grade:
            match_clause = (
                "MATCH (a:Concept {name: $src, grade: $src_grade}), "
                "      (b:Concept {name: $tgt, grade: $tgt_grade}) "
            )
            params = {
                "src": source,
                "tgt": target,
                "src_grade": source_grade,
                "tgt_grade": target_grade,
                "confidence": confidence,
                "description": description,
                "method": method,
            }
        else:
            match_clause = (
                "MATCH (a:Concept {name: $src}), (b:Concept {name: $tgt}) "
            )
            params = {
                "src": source,
                "tgt": target,
                "confidence": confidence,
                "description": description,
                "method": method,
            }
        with driver.session() as session:
            session.run(
                f"""
                {match_clause}
                MERGE (a)-[r:{rel_type}]->(b)
                SET r.confidence = $confidence,
                    r.description = $description,
                    r.method = $method
                """,
                **params,
            )

    def get_pairs_with_existing_typed_edges(self, driver, pairs):
        """Return sorted-tuple keys of pairs that already have any non-SIMILAR_TO
        edge between source and target.

        Matches Concepts by ``(name, grade)`` when grade info is present on the
        pair dict (it should be — ``find_similar_pairs_ann`` attaches
        ``source_grade``/``target_grade`` for Yhoga). Falls back to name-only
        match when grade is missing.
        """
        if not pairs:
            return set()
        payload = [
            {
                "src": p["source"],
                "src_grade": p.get("source_grade") or "",
                "tgt": p["target"],
                "tgt_grade": p.get("target_grade") or "",
            }
            for p in pairs
        ]
        with driver.session() as session:
            result = session.run(
                """
                UNWIND $rows AS row
                MATCH (a:Concept {name: row.src})
                MATCH (b:Concept {name: row.tgt})
                WHERE (row.src_grade = '' OR a.grade = row.src_grade)
                  AND (row.tgt_grade = '' OR b.grade = row.tgt_grade)
                OPTIONAL MATCH (a)-[r]-(b)
                WHERE type(r) <> 'SIMILAR_TO'
                WITH row, count(r) AS existing
                WHERE existing > 0
                RETURN row.src AS src, row.tgt AS tgt
                """,
                rows=payload,
            )
            return {
                tuple(sorted([r["src"], r["tgt"]])) for r in result
            }

    def get_documents(self, driver):
        # Yhoga's "document" analog is :Grade. Count Concepts reachable from each.
        with driver.session() as session:
            result = session.run(
                """
                MATCH (g:Grade)
                OPTIONAL MATCH (g)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SUBTOPIC]->
                              (:Subtopic)-[:HAS_CONCEPT]->(c:Concept)
                RETURN g.name AS name, g.type AS kelas,
                       count(DISTINCT c) AS topic_count,
                       NULL AS uploaded_at
                ORDER BY g.name
                """
            )
            return [dict(r) for r in result]

    def get_total_concept_count(self, driver):
        with driver.session() as session:
            r = session.run("MATCH (n:Concept) RETURN count(n) AS n").single()
            return r["n"] if r else 0

    def get_similar_pairs(self, driver):
        with driver.session() as session:
            result = session.run(
                """
                MATCH (a:Concept)-[r:SIMILAR_TO]->(b:Concept)
                RETURN a.name AS source, b.name AS target, r.score AS score,
                       coalesce(a.grade, '') AS source_grade,
                       coalesce(b.grade, '') AS target_grade
                ORDER BY r.score DESC
                """
            )
            return [dict(r) for r in result]


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────


_REGISTRY: dict[str, type[SchemaAdapter]] = {
    "soros": SorosAdapter,
    "yhoga": YhogaAdapter,
}


def get_adapter(
    name: str = "soros",
    connection: Neo4jConnection | None = None,
) -> SchemaAdapter:
    """Return the adapter for `name` (currently 'soros' or 'yhoga').

    Args:
        name: Schema name — 'soros' or 'yhoga'. Picks ontology shape, label
            set, vocab.
        connection: Optional explicit Neo4j connection. If None, the adapter
            consults NEO4J_TARGET, then falls back to the spec's env-var names.
            Pass `Neo4jConnection.default()` or `Neo4jConnection.yhoga()` to
            force a specific upstream.

    Examples:
        # Default behavior — yhoga schema on yhoga DB (legacy)
        adapter = get_adapter("yhoga")

        # Yhoga ontology on the project's own NEO4J_URI
        from src.connection import Neo4jConnection
        adapter = get_adapter("yhoga", connection=Neo4jConnection.default())

        # Or via env var (one-line app-wide override)
        # $ export NEO4J_TARGET=default
        adapter = get_adapter("yhoga")  # picks up default DB despite yhoga schema
    """
    cls = _REGISTRY.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown schema '{name}'. Known: {sorted(_REGISTRY)}"
        )
    return cls(connection=connection)
