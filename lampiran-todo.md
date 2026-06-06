# Lampiran yang Perlu Ditambahkan

Status pelacakan lampiran yang disinggung/dibutuhkan di Bab 1–5.
Sudah ada sebelumnya: **Lampiran A** (Heuristik Deteksi TOC), **B** (Arsitektur
KG Review App), **C** (Visualisasi KG per Mata Pelajaran).
Ditambahkan via PR #7 (`feat/lampiran-appendices`): **Lampiran D** (Templat
Prompt), **E** (Relasi Lintas-Buku & Validasi Fase 2), **F** (Kueri Cypher),
**G** (Daftar Missing Triples).

| # | Lampiran | Isi | Sumber data | Prioritas | Status |
|---|---|---|---|---|---|
| 1 | **Templat Prompt** | Prompt lengkap untuk ekstraksi (8 blok), \textit{completion} (klasifikasi relasi), dan \textit{LLM-as-a-judge} | notebook implementasi | **Tinggi** | ✅ Selesai — **Lampiran D** (PR #7) |
| 2 | **Daftar Relasi Lintas-Buku & Hasil Validasi Fase 2** | 117 \textit{edge} (LB001–LB117): konsep A–tipe–konsep B, \textit{confidence}, validitas/tipe/arah | `completion/`, `canonical/03_completion/` | **Sedang–Tinggi** | ✅ Selesai — **Lampiran E** (PR #7) |
| 3 | **Kueri Cypher** | Cypher konstruksi graf (\textit{passes} \texttt{MERGE}) + perhitungan metrik (ADC, Density, Modularity, TRR, DC, ESR) | `neo4j-dashboards/`, `canonical/` | Sedang | ✅ Selesai — **Lampiran F** (PR #7) |
| 4 | **Daftar Buku Teks / Korpus** | 3 e-book Kemendikbudristek (Fisika/Kimia/Biologi Fase F Kelas XII): judul, edisi/tahun, URL | `buku.kemdikbud.go.id` | Sedang | ⬜ Belum — butuh judul/edisi/URL final |
| 5 | **Profil & Kualifikasi Pakar** | Enam pakar (dua per mapel), anonim: latar pendidikan/bidang | — | Sedang | ⬜ Belum — butuh data dari penulis (tidak ada di repo) |
| 6 | **Daftar Missing Triples** | Relasi yang ditandai pakar sebagai terlewat, per mapel (15 Fisika, 29 Kimia, 0 Biologi) | data validasi Fase 1 | Rendah–Sedang | ✅ Selesai — **Lampiran G** (PR #7); dibuat sebagai lampiran tersendiri, bukan digabung ke #2 |
| 7 | **Tautan Repositori Kode & Dataset KG** | Repo notebook/kode + ekspor KG terbuka (Neo4j) | repo/DOI | Rendah/opsional | ⬜ Belum — butuh URL repo/DOI |

## Catatan
- Tidak perlu lampiran (sudah di dalam bab): skema ontologi OWL (Bab 3 §Skema
  Ontologi), rumus & definisi metrik (Bab 2), tabel Parameter Sistem & Definisi
  Operasional (Bab 3), instrumen & metrik Fase 2 (Bab 3 subsec:metrik-fase2),
  rubrik 4 kategori penilaian (Bab 2).
- Tautan `\ref` lampiran baru (sudah dipasang via PR #7):
  - **D** (Templat Prompt) ← Bab 4 (struktur prompt ekstraksi + klasifikasi relasi).
  - **E** (Relasi Lintas-Buku) ← Bab 5 §Hasil Validasi Pakar.
  - **F** (Kueri Cypher) ← Bab 4 (konstruksi tiga-\textit{pass}) + Bab 5 (metrik struktural).
  - **G** (Missing Triples) ← Bab 5 §5.2 Hasil Validasi Pakar (bukan Bab 3 metodologi).
- Sisa yang belum (butuh masukan eksternal): #4 (judul/URL buku), #5 (profil pakar),
  #7 (URL repo/DOI).
- Catatan data: Bab 5 menyebut "125 relasi lintas-buku", sedangkan validasi Fase 2
  mencakup 117 (LB001–LB117); perlu kalimat penyelaras 117-dari-125 di Bab 5.
