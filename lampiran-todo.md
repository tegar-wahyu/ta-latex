# Lampiran yang Perlu Ditambahkan

Daftar lampiran yang disinggung/dibutuhkan di Bab 1–5 tetapi belum dibuat.
Sudah ada: **Lampiran A** (Heuristik Deteksi TOC), **B** (Arsitektur KG Review App),
**C** (Visualisasi KG per Mata Pelajaran).

| # | Lampiran | Isi | Sumber data | Prioritas |
|---|---|---|---|---|
| 1 | **Templat Prompt** | Prompt lengkap untuk ekstraksi (8 blok), \textit{completion} (klasifikasi relasi), dan \textit{LLM-as-a-judge} | notebook implementasi | **Tinggi** |
| 2 | **Daftar Relasi Lintas-Buku & Hasil Validasi Fase 2** | 117 \textit{edge} (LB001–LB117): konsep A–tipe–konsep B, \textit{confidence}, validitas/tipe/arah | `completion/`, `canonical/03_completion/` | **Sedang–Tinggi** |
| 3 | **Kueri Cypher** | Cypher konstruksi graf (\textit{passes} \texttt{MERGE}) + perhitungan metrik (ADC, Density, Modularity, TRR, DC, ESR) | `neo4j-dashboards/`, `canonical/` | Sedang |
| 4 | **Daftar Buku Teks / Korpus** | 3 e-book Kemendikbudristek (Fisika/Kimia/Biologi Fase F Kelas XII): judul, edisi/tahun, URL | `buku.kemdikbud.go.id` | Sedang |
| 5 | **Profil & Kualifikasi Pakar** | Enam pakar (dua per mapel), anonim: latar pendidikan/bidang | — | Sedang |
| 6 | **Daftar Missing Triples** | Relasi yang ditandai pakar sebagai terlewat, per mapel (15 Fisika, 29 Kimia, …) | data validasi Fase 1 | Rendah–Sedang (bisa digabung ke #2) |
| 7 | **Tautan Repositori Kode & Dataset KG** | Repo notebook/kode + ekspor KG terbuka (Neo4j) | repo/DOI | Rendah/opsional |

## Catatan
- Tidak perlu lampiran (sudah di dalam bab): skema ontologi OWL (Bab 3 §Skema
  Ontologi), rumus & definisi metrik (Bab 2), tabel Parameter Sistem & Definisi
  Operasional (Bab 3), instrumen & metrik Fase 2 (Bab 3 subsec:metrik-fase2),
  rubrik 4 kategori penilaian (Bab 2).
- Hanya Bab 4 yang sudah me-`\ref` lampiran (A & B); lampiran baru perlu
  ditautkan dari bab terkait (mis. #1/#3 dari Bab 4; #2/#5/#6 dari Bab 3/Bab 5).
