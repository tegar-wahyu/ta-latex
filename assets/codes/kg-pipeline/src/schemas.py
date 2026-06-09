"""Pydantic models for structured LLM extraction."""

from typing import Literal

from pydantic import BaseModel

# =============================================================================
# Relation Types - Aligned with docs/ontology.ttl
# =============================================================================

# The 7 ontology-defined semantic relations (authoritative)
RELATION_TYPES = {
    "BAGIAN_DARI",  # A is part of B (hierarchy)
    "MENYEBABKAN",  # A causes B (causation)
    "BERGANTUNG_PADA",  # A depends on B (dependency)
    "MENDEFINISIKAN",  # A defines B (definitional)
    "MEMPENGARUHI",  # A influences B (general influence)
    "BERINTERAKSI_DENGAN",  # A interacts with B (interaction)
    "DIFORMULASIKAN_SEBAGAI",  # A is formulated as B (formal/mathematical)
}

# Mapping from LLM-generated types to canonical ontology types
# Used to normalize relations before inserting into Neo4j
RELATION_TYPE_MAPPING = {
    # Influence variants → MEMPENGARUHI
    "MEMUNGKINKAN": "MEMPENGARUHI",
    "MENGATUR": "MEMPENGARUHI",
    "MENENTUKAN": "MEMPENGARUHI",
    "MENGONTROL": "MEMPENGARUHI",
    "MENGARAHKAN": "MEMPENGARUHI",
    # Hierarchy variants → BAGIAN_DARI
    "TERDIRI_DARI": "BAGIAN_DARI",  # Note: direction may need swap
    "MENGANDUNG": "BAGIAN_DARI",
    "MEMILIKI": "BAGIAN_DARI",
    # Interaction variants → BERINTERAKSI_DENGAN
    "BEREAKSI_DENGAN": "BERINTERAKSI_DENGAN",
    "BERKAITAN_DENGAN": "BERINTERAKSI_DENGAN",
    "BERHUBUNGAN_DENGAN": "BERINTERAKSI_DENGAN",
    "BERASOSIASI_DENGAN": "BERINTERAKSI_DENGAN",
    # Causation variants → MENYEBABKAN
    "MENIMBULKAN": "MENYEBABKAN",
    # Product/reaction variants → MENYEBABKAN
    "MENGHASILKAN": "MENYEBABKAN",
    "MEMPRODUKSI": "MENYEBABKAN",
    "MENGHASILKAN_PRODUK": "MENYEBABKAN",
    # Process step variants → BAGIAN_DARI
    "TAHAP_DARI": "BAGIAN_DARI",
    "LANGKAH_DARI": "BAGIAN_DARI",
    "TAHAPAN_DARI": "BAGIAN_DARI",
    "LANGKAH_DALAM": "BAGIAN_DARI",
    # Dependency variants → BERGANTUNG_PADA
    "MEMBUTUHKAN": "BERGANTUNG_PADA",
    "MENGGUNAKAN": "BERGANTUNG_PADA",
    "MEMERLUKAN": "BERGANTUNG_PADA",
    # Definitional variants → MENDEFINISIKAN
    "MENJELASKAN": "MENDEFINISIKAN",
    "MENGURAIIKAN": "MENDEFINISIKAN",
    "MENGGAMBARKAN": "MENDEFINISIKAN",
    # Formal variants → DIFORMULASIKAN_SEBAGAI
    "DIUKUR_DENGAN": "DIFORMULASIKAN_SEBAGAI",
    "DIREPRESENTASIKAN_SEBAGAI": "DIFORMULASIKAN_SEBAGAI",
}

# Backward-compat alias
PREFERRED_RELATION_TYPES = list(RELATION_TYPES)

# Type alias for documentation
RelationType = str


def normalize_relation_type(rel_type: str) -> str | None:
    """Normalize a relation type to one of the 7 ontology-defined types.

    Args:
        rel_type: The relation type string from LLM output

    Returns:
        One of the 7 canonical types, or None if the relation should be skipped
    """
    # Uppercase and strip whitespace
    rel_type = rel_type.upper().strip()

    # Already canonical?
    if rel_type in RELATION_TYPES:
        return rel_type

    # Map to canonical type
    return RELATION_TYPE_MAPPING.get(rel_type)


class SubKonsep(BaseModel):
    name: str
    description: str
    formula: list[str] = []  # e.g., ["F = m × a", "a = F/m"]
    variables: list[str] = []  # e.g., ["F (gaya, N)", "m (massa, kg)"]
    kondisi: list[str] = []  # e.g., ["suhu konstan", "gas ideal"]


# Backward-compat alias
SubTopic = SubKonsep


class Konsep(BaseModel):
    name: str
    description: str
    sub_konsep: list[SubKonsep] = []
    formula: list[str] = []  # e.g., ["F = m × a", "a = F/m"]
    variables: list[str] = []  # e.g., ["F (gaya, N)", "m (massa, kg)", "a (percepatan, m/s²)"]
    kondisi: list[str] = []  # e.g., ["suhu konstan", "gas ideal", "tanpa gesekan"]


# Backward-compat alias
Topic = Konsep


class MataPelajaran(BaseModel):
    """Indonesian curriculum subject (Mata Pelajaran)."""

    name: str  # e.g., "Fisika", "Biologi", "Kimia"
    phase: str = ""  # "E" (Kelas X) or "F" (Kelas XI-XII)


# Backward-compat alias
Subject = MataPelajaran

# Common subjects for Indonesian high school curriculum
SUBJECT_CHOICES = [
    "Fisika",
    "Kimia",
    "Biologi",
    "Matematika",
    "Bahasa Indonesia",
    "Bahasa Inggris",
    "Sejarah",
    "Geografi",
    "Ekonomi",
    "Sosiologi",
    "Informatika",
    "Other",
]

# Kurikulum Merdeka phases
PHASE_CHOICES = {
    "E": "Fase E (Kelas X)",
    "F": "Fase F (Kelas XI-XII)",
}

# Textbook class levels
KELAS_CHOICES = ["X", "XI", "XII"]


class KonsepExtraction(BaseModel):
    mata_pelajaran: MataPelajaran | None = None
    bab_name: str | None = None  # Chapter/section name detected from text
    konsep: list[Konsep]


# Backward-compat alias
TopicExtraction = KonsepExtraction


# =============================================================================
# Per-Bab Extraction Schema (ToC-enforced with inline relations)
# =============================================================================


class KonsepRelation(BaseModel):
    """A typed relation from one konsep to another."""

    type: str  # Flexible: allows any relation type the LLM generates
    target: str  # Name of target konsep
    description: str = ""  # Brief explanation of why this relation exists


class KonsepWithRelations(BaseModel):
    """Konsep with inline relations for per-Bab extraction."""

    name: str
    description: str
    relations: list[KonsepRelation] = []
    formula: list[str] = []  # e.g., ["F = m × a", "a = F/m"]
    variables: list[str] = []  # e.g., ["F (gaya, N)", "m (massa, kg)"]
    kondisi: list[str] = []  # e.g., ["suhu konstan", "gas ideal"]


class SubBabExtraction(BaseModel):
    """Extraction result for one SubBab (enforced to match ToC name)."""

    name: str  # Must match SubBab name from ToC exactly
    konsep: list[KonsepWithRelations] = []


class BabExtraction(BaseModel):
    """Per-Bab LLM output schema for ToC-enforced extraction."""

    bab_summary: str = ""  # 2-3 sentence academic summary
    sub_bab: list[SubBabExtraction]
