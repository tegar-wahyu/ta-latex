"""Schema specifications for KG ontologies the completion pipeline can target.

Each `SchemaSpec` is a static, frozen description of a graph: node labels, the
properties to embed/classify on, the env-var keys for the Neo4j connection,
and the existing typed relationship types whose neighborhoods feed the
classifier prompt. The runtime adapter (`src/schema_adapter.py`) is the thing
that actually executes Cypher; this module just declares *what* a schema is.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SchemaSpec:
    """Static, schema-shape-only configuration for a target KG."""

    name: str
    # Node labels relevant for completion (similarity + classification).
    # First label is treated as the primary "concept" label.
    node_labels: tuple[str, ...]
    # Vector index name per node label.
    index_names: dict[str, str]
    # Property holding the human description of a concept.
    description_field: str
    # Extra fields folded into the embedding text and classifier prompt.
    embed_extra_fields: tuple[str, ...]
    # Env var keys for the Neo4j connection.
    neo4j_uri_env: str
    neo4j_user_env: str
    neo4j_password_env: str
    # Existing typed rel types whose neighborhoods are surfaced to the classifier.
    existing_typed_rels: tuple[str, ...]
    # Closed vocabulary the LLM classifier emits for newly-discovered edges.
    # First N-1 entries are real rel types; last entry is reserved for "none".
    classifier_vocab: tuple[str, ...] = (
        "isPrerequisiteOf",
        "supports",
        "analogousTo",
        "none",
    )
    # Discovery edge name written by the similarity step.
    similar_to_rel: str = "SIMILAR_TO"


SOROS_SCHEMA = SchemaSpec(
    name="soros",
    node_labels=("Konsep", "SubKonsep"),
    index_names={
        "Konsep": "konsep_embedding_idx",
        "SubKonsep": "subkonsep_embedding_idx",
    },
    description_field="description",
    embed_extra_fields=("formula", "variables", "kondisi"),
    neo4j_uri_env="NEO4J_URI",
    neo4j_user_env="NEO4J_USERNAME",
    neo4j_password_env="NEO4J_PASSWORD",
    existing_typed_rels=("isPrerequisiteOf", "supports", "analogousTo"),
)


YHOGA_SCHEMA = SchemaSpec(
    name="yhoga",
    node_labels=("Concept",),
    index_names={"Concept": "concept_embedding_idx"},
    description_field="description",
    embed_extra_fields=("materi_pokok_ref", "grade"),
    neo4j_uri_env="NEO4J_URI_YHOGA",
    neo4j_user_env="NEO4J_USERNAME_YHOGA",
    neo4j_password_env="NEO4J_PASSWORD_YHOGA",
    existing_typed_rels=(
        "BERINTERAKSI_DENGAN",
        "MEMPENGARUHI",
        "BAGIAN_DARI",
        "BERGANTUNG_PADA",
        "MENYEBABKAN",
        "MEMUNGKINKAN",
        "MENDEFINISIKAN",
        "MEMDEFINISIKAN",
        "TERDIRI_DARI",
        "MENGHASILKAN",
        "DIRUMUSKAN_SEBAGAI",
        "PRASYARAT",
        "MEMPERSIAPKAN",
    ),
    # 5-type closed cross-book vocab from docs/yhoga-ontology.ttl:151-185.
    # Plus "none" so the classifier can decline an edge.
    classifier_vocab=(
        "LINTAS_BUKU_SAMA_DENGAN",
        "LINTAS_BUKU_APLIKASI_DARI",
        "LINTAS_BUKU_PRASYARAT_UNTUK",
        "LINTAS_BUKU_MEMPERDALAM",
        "LINTAS_BUKU_BERKAITAN_DENGAN",
        "none",
    ),
)


SCHEMAS: dict[str, SchemaSpec] = {
    "soros": SOROS_SCHEMA,
    "yhoga": YHOGA_SCHEMA,
}
