# LLM-Assisted Knowledge Graph Completion untuk Pemetaan Interkoneksi Materi Sains (Fisika, Kimia, Biologi) pada Kurikulum Merdeka SMA Kelas XII

Fadrian Yhoga Pratama (2206819395), Soros Febriano (2206083445), Tegar Wahyu
Khisbulloh (2206082032) — Program Studi Ilmu Komputer, Fakultas Ilmu Komputer,
Universitas Indonesia (Sarjana, 2026).

Pembimbing: Dr. Siti Aminah, S.Kom., M.Kom.

----------

## Selayang Pandang

Struktur Kurikulum Merdeka memisahkan rumpun IPA pada jenjang SMA menjadi
Fisika, Kimia, dan Biologi sehingga berisiko menimbulkan *siloed learning* —
terputusnya koneksi konseptual antarmata pelajaran. Penelitian ini
mengembangkan *pipeline* **LLM-Assisted Knowledge Graph Completion** (KGC) untuk
membangun dan melengkapi graf pengetahuan sains lintas-disiplin Fase F Kelas XII
berbahasa Indonesia, berbasis buku teks resmi Kemendikbudristek.

Mengikuti kerangka *Design Science Research*, sistem mengekstraksi entitas dan
relasi intra-buku menggunakan LLM dengan *prompt few-shot* terpandu skema
ontologi OWL, menginstansiasikannya ke basis data graf Neo4j, lalu memprediksi
relasi implisit lintas-disiplin melalui pencarian kandidat *Approximate Nearest
Neighbor* (HNSW) atas *embedding* deskripsi konsep yang dilanjutkan klasifikasi
relasi oleh LLM. Kualitas graf dievaluasi secara *hybrid*: metrik struktural dan
validasi enam pakar domain, dengan *LLM-as-a-judge* sebagai penengah
ketidaksepakatan.

Temuan utama: penambahan 125 relasi lintas-buku menurunkan *Modularity* dari
0,811 menjadi 0,373 dan menaikkan *Average Degree Centrality* sebesar 98,4%;
ekstraksi intra-mapel mencapai F1-Score 92,6%; relasi lintas-disiplin tervalidasi
valid (99,1%) dan tepat tipe (98,3%), namun ketepatan arah relasi hanya 75,9%.

## Cara Kompilasi

Berkas master adalah `thesis.tex` (`\documentclass{report}`, memuat seluruh bab
dan lampiran melalui `\subfile`).

### Menggunakan Overleaf (disarankan)

1. Unduh repositori ini sebagai zip.
2. Unggah ke Overleaf, atur *Main document* ke `thesis.tex`.
3. Atur *Compiler* ke **pdfLaTeX** dan *TeX Live* versi terbaru (proyek juga
   kompatibel dengan XeLaTeX).

### Kompilasi lokal

Bibliografi memakai biblatex dengan *backend* Biber:

```bash
pdflatex thesis.tex
biber thesis
pdflatex thesis.tex
pdflatex thesis.tex
```

Tiap bab dan lampiran dapat dikompilasi mandiri (kelas `subfiles` mewarisi
preamble `thesis.tex`), mis. `pdflatex src/01-body/bab-5.tex`.

## Struktur Repositori

```
thesis.tex                 # berkas master
src/00-frontMatter/        # sampul, halaman judul, pernyataan, kata pengantar, abstrak
src/01-body/               # bab-1 s.d. bab-6 (+ diagram TikZ: pipeline, ontologi, llm-judge)
src/99-backMatter/         # lampiran-1 s.d. lampiran-8
config/references.bib      # basis data pustaka (biblatex)
assets/codes/kg-pipeline/  # notebook pipeline ekstraksi, KGC, dan ingest awal
assets/codes/final-consensus/
                           # notebook konsensus pakar, metrik, dan ingest KG konsensus
assets/data/               # data kanonik, completion, dan dashboard Neo4j
assets/kg-from-neo4j/      # ekspor KG dari Neo4j (node/edge pre-KGC dan post-KGC)
assets/pics/               # gambar & diagram, termasuk KG, UI, HNSW, dan LLM-as-judge
assets/signs/              # tanda tangan penulis
```

## Milestone Pekerjaan

Penelitian dilaksanakan selama enam bulan (Januari–Juni 2026):

| Fase | Kegiatan | Periode |
|------|----------|---------|
| 1 | Studi literatur dan desain sistem | Januari–Februari 2026 |
| 2 | Implementasi *pipeline* ekstraksi | Februari–Maret 2026 |
| 3 | Konstruksi *Knowledge Graph* | Maret–April 2026 |
| 4 | Evaluasi dan validasi pakar #1 | Maret–April 2026 |
| 5 | *Knowledge Graph Completion* | April 2026 |
| 6 | Evaluasi dan validasi pakar #2 | Mei 2026 |
| 7 | Analisis hasil dan penulisan | Mei–Juni 2026 |

## Referensi dan Data

- **Korpus**: tiga Buku Siswa (Fisika, Kimia, Biologi Fase F Kelas XII) sebagai
  sumber ekstraksi, beserta Buku Panduan Guru pasangannya, terbitan Pusat
  Perbukuan Kemendikbudristek (cetakan pertama, 2022). Rincian bibliografis
  lengkap tercantum pada Lampiran 7 dan diperoleh dari
  <https://buku.kemdikbud.go.id>.
- **Artefak**: purwarupa *Knowledge Graph* pada Neo4j beserta *pipeline*
  ekstraksi–konstruksi–*completion* yang terdokumentasi pada Bab 4. Notebook
  replikasi pipeline tersedia di `assets/codes/kg-pipeline/`; catatan
  replikasi di dalam folder tersebut juga mengarahkan pembaca untuk
  memvalidasi keluaran Pass-1 melalui alur `assets/codes/final-consensus/`
  sebelum memakai hasilnya sebagai basis Pass-2 atau analisis akhir.
- **Konsensus dan metrik validasi**: `assets/codes/final-consensus/` memuat
  notebook untuk membentuk *gold standard* final per mata pelajaran, menghitung
  Gwet's AC1 serta metrik Precision/Recall/F1, dan mengingest KG konsensus ke
  Neo4j. Folder ini juga menyimpan `*_gold_standard.json`, berkas validasi pakar,
  checkpoint LLM-as-a-judge, dan README khusus untuk navigasi.
- **Ekspor KG Neo4j**: ekspor graf dari Neo4j tersedia di
  `assets/kg-from-neo4j/`, mencakup
  `01-pre-kgc-nodes.json`, `01-pre-kgc-edges.json`, `02-post-kgc-nodes.json`,
  dan `02-post-kgc-edges.json`.
- **Data pendukung**: `assets/data/` memuat data kanonik, keluaran completion,
  hasil survei completion, serta konfigurasi dashboard Neo4j pre/post-KGC.
- **Visual**: `assets/pics/` hanya memuat gambar yang dipakai dalam dokumen atau
  README, termasuk visualisasi KG, antarmuka validasi, diagram HNSW/KICGPT,
  `param-sweep.png`, dan `llm-as-a-judge-pipeline.png`.
- **Pustaka ilmiah**: seluruh rujukan tercatat pada `config/references.bib`.

## Atribusi dan Lisensi

Penyiapan LaTeX skripsi ini mengikuti standar Tugas Akhir Universitas Indonesia
(2017) dan mengadaptasi templat
[latex-ta-ui](https://gitlab.com/ichlaffterlalu/latex-ta-ui) karya Andreas
Febrian, Erik Dominikus, Azhar Kurnia, dan Ichlasul Affan (Fasilkom UI).

Hak cipta atas isi skripsi dimiliki oleh para penulis dan Universitas Indonesia.
