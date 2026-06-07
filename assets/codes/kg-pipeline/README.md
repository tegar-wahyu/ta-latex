# KG Pipeline — Ekstraksi + Cross-Book Completion (Biologi / Fisika / Kimia Kelas XII)

Notebook `TA_KG_PIPELINE.ipynb` membangun Knowledge Graph lintas-buku dari PDF buku teks
Kemdikbud secara end-to-end: ekstraksi konsep & relasi intra-buku (Pass-1), *completion*
lintas-buku berbasis ANN + LLM (Pass-2), lalu *ingest* ke Neo4j.

> Salinan ini disiapkan untuk lampiran/deposit skripsi. Kredensial **tidak** ditanam di
> notebook — diisi lewat file `.env` (lihat `.env.example`). Kode sumber lengkap & riwayat:
> https://github.com/Kemendickbud/final-kg

## Prasyarat
- **Python 3.10+** (diuji dengan 3.13)
- **VS Code** + ekstensi *Python* & *Jupyter* (atau Jupyter Lab)

## Setup

### 1. Siapkan PDF buku
File PDF **TIDAK** disertakan (ukurannya melebihi limit GitHub). Taruh di folder `textbook/`
dengan nama PERSIS seperti ini:
```
textbook/
├── Biologi_BS_KLS_XII_Rev.pdf
├── Biologi_BG_KLS_XII.pdf
├── Fisika_BS_KLS_XII.pdf
├── Fisika_BG_KLS_XII.pdf
├── Kimia_BS_KLS_XII.pdf
└── Kimia_BG_KLS_XII.pdf
```
> `BS` = Buku Siswa, `BG` = Buku Guru.

### 2. Virtual environment + dependency
```bash
python -m venv venv

# Windows (PowerShell):
venv\Scripts\Activate.ps1
# macOS / Linux:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```
> Unduhan pertama agak lama karena menarik PyTorch & model embedding.

### 3. Daftarkan kernel untuk Jupyter
```bash
python -m ipykernel install --user --name kg-pipeline --display-name "Python (kg-pipeline)"
```

### 4. Isi kredensial lewat `.env`
Salin templat lalu isi nilainya (kredensial **tidak** lagi ditulis di dalam notebook):
```bash
cp .env.example .env
```
- `GEMINI_API_KEY` — API key Google Gemini (https://aistudio.google.com/apikey)
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASS`, `NEO4J_DB` — kredensial Neo4j (hanya untuk tahap *ingest*)

Notebook memuat `.env` otomatis via `python-dotenv`. Bagian Pass-1 preprocessing (load PDF,
chapter, glossary) **tidak butuh** API key.

### 5. Jalankan
Buka `TA_KG_PIPELINE.ipynb` → pilih kernel **"Python (kg-pipeline)"** → **Run All** (jalankan
dari atas berurutan). Cek output cell config — semua PDF harus `OK`, bukan `MISSING`.

## Parameter utama (sesuai metodologi skripsi)
- Chunking: `chunk_size = 800`, `chunk_overlap = 200`
- LLM: `gemini-2.5-flash` (ekstraksi & klasifikasi relasi)
- Embedding: `paraphrase-multilingual-mpnet-base-v2`
- Completion lintas-buku: `COS_THRESHOLD = 0.75`, `TOP_K = 15`,
  ambang confidence per-tipe `{LINTAS_BUKU_BERKAITAN_DENGAN: 0.85, default: 0.7}`

## Troubleshooting
| Gejala | Penyebab & solusi |
|---|---|
| `MISSING textbook/...pdf` | PDF belum ada / nama salah. Pastikan ada di `textbook/` dengan nama persis (langkah 1). |
| `Chapters: 0`, `Glossary: 0` | PDF tidak ter-parse. Pastikan `pip install -r requirements.txt` sukses (butuh `llama-index-readers-file` + `pypdf` + `PyMuPDF`), lalu **Restart Kernel** dan Run All. |
| Kernel `venv` tidak muncul di VS Code | Ulangi langkah 3, atau pilih **Enter interpreter path** → `venv` Python. |
| `GEMINI_API_KEY` bernilai `None` | `.env` belum dibuat/terisi, atau `python-dotenv` belum terinstall (`pip install -r requirements.txt`). |

## Catatan
- `PROJECT_DIR` dideteksi otomatis dari folder notebook (`Path.cwd()`), jadi tidak perlu edit path manual.
- Folder `venv/`, `outputs/`, dan `textbook/` sebaiknya tidak ikut di-commit.
- Catatan replikasi: sebelum memakai keluaran Pass-1 sebagai basis Pass-2 atau analisis akhir,
  sebaiknya validasi dulu relasi intra-buku hasil Pass-1. Untuk alur validasi pakar,
  konsensus akhir, dan metrik validasi, lanjutkan ke
  [`../final-consensus/`](../final-consensus/).
