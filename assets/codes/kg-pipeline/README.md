# KG Pipeline — Extraction and Completion Artifacts

The current thesis pipeline is split across separate artifacts. Do not treat
`TA_KG_PIPELINE.ipynb` as the canonical end-to-end flow; it is an older
all-in-one notebook kept for traceability.

> **Legacy artifact:** `TA_KG_PIPELINE.ipynb` is deprecated and should not be
> used to reproduce the final thesis results. It still contains draft-era
> extraction-time `cross_book_links` / `infer_cross_links` logic. Use
> `TA_KG_EXTRACTION.ipynb` plus the `src/` completion runtime instead.

Canonical sources:

1. **Extraction / KG JSON awal** — `TA_KG_EXTRACTION.ipynb`
2. **Expert review + final consensus** — `assets/codes/final-consensus/`
3. **Consensus ingest / metrics** — consensus Neo4j ingest notebook and exports
4. **Cross-book completion** — `src/completion.py`

`TA_KG_EXTRACTION.ipynb` builds the initial per-book Class XII Knowledge Graph
JSON for Biologi, Fisika, and Kimia:

1. **Pass-1 extraction** — load textbook PDFs, extract table of contents,
   chapters, glossary, and `materi_pokok`, then ask Gemini to extract concepts
   and intra-book relationships.
2. **Save KG JSON awal** — write one JSON graph per subject to `outputs/*.json`.
3. **No cross-book completion / no Neo4j ingest here** — those stages are handled
   by later pipeline artifacts.

> This copy is prepared for thesis appendix/deposit use. Credentials are not
> embedded in the notebook; put them in `.env`.

## Prerequisites

- Python 3.10+ (tested with Python 3.13)
- VS Code with the Python and Jupyter extensions, or Jupyter Lab
- Neo4j Aura / Neo4j 5 with vector indexes enabled
- Google Gemini API key for the canonical run

## Setup

### 1. Prepare Textbook PDFs

PDF files are not included in this repository. Put them in `textbook/` with
these exact names:

```text
textbook/
├── Biologi_BS_KLS_XII_Rev.pdf
├── Biologi_BG_KLS_XII.pdf
├── Fisika_BS_KLS_XII.pdf
├── Fisika_BG_KLS_XII.pdf
├── Kimia_BS_KLS_XII.pdf
└── Kimia_BG_KLS_XII.pdf
```

`BS` means Buku Siswa; `BG` means Buku Guru.

### 2. Create the Environment

```bash
python -m venv venv

# Windows PowerShell
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

The first install can take a while because it pulls PDF, LlamaIndex, embedding,
Neo4j, and notebook dependencies.

### 3. Register the Jupyter Kernel

```bash
python -m ipykernel install --user --name kg-pipeline --display-name "Python (kg-pipeline)"
```

### 4. Configure `.env`

Copy the template and fill in the values:

```bash
cp .env.example .env
```

The notebook extraction/ingest cells currently use these legacy names:

```env
GEMINI_API_KEY=
NEO4J_URI=
NEO4J_USER=
NEO4J_PASS=
NEO4J_DB=
```

The bundled `src/` completion runtime also supports the newer app-style names:

```env
GOOGLE_API_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
ZAI_API_KEY=
HUGGINGFACE_API_KEY=

NEO4J_URI=
NEO4J_USERNAME=
NEO4J_PASSWORD=

NEO4J_URI_YHOGA=
NEO4J_USERNAME_YHOGA=
NEO4J_PASSWORD_YHOGA=
NEO4J_TARGET=
```

For the notebook as written, either duplicate the Neo4j values across both
name sets or update the notebook config cell to one convention. `NEO4J_TARGET`
is optional; set it to `default`/`soros` or `yhoga` when you want a schema
adapter to run against a specific configured Neo4j instance.

### 5. Run the Notebook

Open `TA_KG_EXTRACTION.ipynb`, select the **Python (kg-pipeline)** kernel, and
run the notebook from top to bottom:

1. Confirm the config cell reports all required PDFs as `OK`.
2. Run PDF loading, chunking, TOC/glossary/materi-pokok extraction.
3. Run LLM extraction.
4. Review the generated KG JSON files in `outputs/`.

## Current Pipeline Details

### Ontology Artifact

The formal ontology is stored at `assets/data/onthology.ttl`. It defines the
schema contract used by the pipeline:

- structural classes: `Grade`, `Chapter`, `Subtopic`, `Concept`, and
  `ConceptTarget`;
- structural edges: `HAS_CHAPTER`, `HAS_SUBTOPIC`, `HAS_CONCEPT`, plus
  chapter-level `NEXT_CHAPTER`, `PRASYARAT`, and `MEMPERSIAPKAN`;
- the closed 15-type within-book relation vocabulary used during extraction;
- the closed 5-type `LINTAS_BUKU_*` vocabulary used during completion.

The notebooks and runtime do not dynamically parse the TTL at execution time.
Instead, they use static constants derived from it:

- `TA_KG_EXTRACTION.ipynb`: `INTRA_RELATION_TYPES` for extraction prompts;
- `src/schema_spec.py`: `YHOGA_SCHEMA.existing_typed_rels` and
  `YHOGA_SCHEMA.classifier_vocab`;
- `src/schema_adapter.py` and `src/completion.py`: Neo4j label/index mappings
  and the `LINTAS_BUKU_*` classifier/writeback logic.

So the TTL functions as the formal documentation and consistency contract, while
the code uses mirrored vocabularies for speed and reproducibility.

### Pass-1: PDF Extraction

- Loads Class XII Biologi, Fisika, and Kimia student/teacher PDFs.
- Extracts table of contents, chapter structure, glossary, and `materi_pokok`.
- Chunks text with `chunk_size = 800` and `chunk_overlap = 200`.
- Uses `gemini-2.5-flash-lite` for concept and intra-book relationship
  extraction.
- Uses the ontology's 15-type within-book relation vocabulary
  (`INTRA_RELATION_TYPES`) for concept-to-concept relations.
- Produces only per-book / intra-subject KG JSON. It does not emit
  `cross_book_links`, does not run `infer_cross_links`, and does not write to
  Neo4j.
- Deduplicates extracted concepts and prepares them for Neo4j ingest.

### Neo4j Ingest

- Creates the curriculum ontology graph:
  `(:Grade)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_SUBTOPIC]->(:Subtopic)-[:HAS_CONCEPT]->(:Concept)`.
- Writes intra-book relationships before completion.
- Reuses the notebook `driver` for the Pass-2 handoff.

### Pass-2: Cross-Book Completion

Pass-2 does not reimplement completion logic inside the notebook. It imports the
bundled project package under `./src` and calls:

- `src.schema_adapter.get_adapter("yhoga")`
- `src.completion.build_embed_text`
- `src.completion.get_embeddings`
- `src.completion.find_similar_pairs_ann`
- `src.completion.classify_similar_pairs`
- `src.completion.dump_lintas_buku_results`

Canonical notebook parameters:

```python
EMBED_MODEL = "gemini/gemini-embedding-001"
CHAT_MODEL = "gemini/gemini-2.5-flash"
THRESHOLD = 0.70
TOP_K = 20
BATCH = 10
```

Completion stages:

1. Build schema-aware embedding text from `Concept` nodes.
2. Cache/fetch embeddings.
3. Store current embeddings in Neo4j and create/update vector indexes.
4. Retrieve candidates with Neo4j ANN search.
5. Keep only cross-`Grade` pairs and skip pairs that already have typed edges.
6. Classify candidates with a graph-aware prompt that includes each concept's
   existing neighborhood.
7. Write accepted `LINTAS_BUKU_*` edges to Neo4j with `confidence`,
   `description`, and `method`.
8. Dump an audit JSON with model/threshold metadata.

The Yhoga completion vocabulary is closed:

- `LINTAS_BUKU_SAMA_DENGAN`
- `LINTAS_BUKU_APLIKASI_DARI`
- `LINTAS_BUKU_PRASYARAT_UNTUK`
- `LINTAS_BUKU_MEMPERDALAM`
- `LINTAS_BUKU_BERKAITAN_DENGAN`
- `none`

## `src/` Runtime Notes

The `src/` folder is the reusable completion runtime copied beside the notebook
for replication.

- `schema_spec.py` declares supported ontology shapes: `soros` and `yhoga`.
- `schema_adapter.py` maps each ontology to labels, vector indexes, Cypher, and
  classifier vocabulary.
- `connection.py` decouples schema choice from Neo4j target selection.
- `completion.py` performs embedding, ANN candidate retrieval, graph-aware LLM
  classification, typed-edge persistence, and audit dumping.
- `cache.py` stores extraction, embedding, and classification caches under
  `data/cache`, `data/embedding_cache`, and `data/classification_cache`.

Default app-style models in `src/config.py` are:

- Chat: `gemini/gemini-2.5-flash`
- Embedding: `gemini/gemini-embedding-001`

The runtime can also route through LiteLLM-compatible OpenAI, Anthropic, Z.ai,
and Hugging Face model strings when the corresponding keys are configured.

## Outputs

Typical generated artifacts:

- `outputs/*.json` — extraction and completion audit outputs from the notebook.
- `data/cache/` — extraction cache and manifest.
- `data/embedding_cache/` — embedding cache keyed by text and model.
- `data/classification_cache/` — LLM classification cache keyed by pair,
  schema, graph-neighborhood hash, model, and prompt version.
- Neo4j graph with `SIMILAR_TO` discovery edges and accepted `LINTAS_BUKU_*`
  relationships.

Do not commit local `venv/`, `textbook/`, `outputs/`, or `data/*cache/`
directories.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `MISSING textbook/...pdf` | The PDF is absent or named differently. Use the exact filenames listed above. |
| `Chapters: 0` or `Glossary: 0` | PDF parsing failed. Reinstall dependencies, restart the kernel, and rerun from the top. |
| Kernel does not appear in VS Code | Rerun the `ipykernel install` command or select the `venv` interpreter manually. |
| `GEMINI_API_KEY` is `None` | `.env` is missing, not loaded, or uses only `GOOGLE_API_KEY`. Add `GEMINI_API_KEY` for the notebook cells. |
| `NEO4J_URI` / `NEO4J_PASS` error in notebook ingest | Fill the legacy notebook names: `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASS`, and `NEO4J_DB`. |
| `NEO4J_USERNAME` / `NEO4J_PASSWORD` error in `src/` | Fill the app-style names used by `src/connection.py`, or duplicate values from `NEO4J_USER` / `NEO4J_PASS`. |
| Vector index dimension mismatch | The runtime drops stale indexes when dimensions differ, then recreates them for the selected embedding model. |
| Completion reuses old classifications | Clear `data/classification_cache/` or bump `CLASSIFICATION_PROMPT_VERSION` in `src/completion.py` after prompt/schema changes. |

## Related Pipeline

For expert validation, final consensus, and validation metrics after Pass-1,
continue with [`../final-consensus/`](../final-consensus/).
