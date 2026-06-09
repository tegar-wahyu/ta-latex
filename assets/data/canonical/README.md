# Paper Data (Main Artifacts)

Data utama yang digunakan paper, satu artefak acuan per tahap pipeline.

## Alur artefak input/output

Alur utama penelitian mengikuti urutan berikut:

1. **KG construction -> KG JSON**
   - Input: PDF Buku Siswa dan Buku Guru Kelas XII.
   - Proses: ingestion, ekstraksi konsep, dan relasi dalam-buku.
   - Output: `01_extraction/*.json`, yaitu KG awal per mata pelajaran.

2. **Ingest Neo4j awal (opsional) -> KG Review App**
   - Input: KG JSON awal.
   - Proses: ingest ke Neo4j bila diperlukan untuk inspeksi dan visualisasi.
   - Output: graf yang dapat ditampilkan pada KG Review App.

3. **Review pakar -> data validasi**
   - Input: graf awal yang ditampilkan pada KG Review App.
   - Proses: dua pakar per mata pelajaran menilai relasi dalam-buku dan
     mengusulkan missing triples.
   - Output: `02_consensus/validations/expert-*.json`.

4. **Final consensus -> KG konsensus**
   - Input: KG awal, berkas validasi pakar, dan konteks buku teks.
   - Proses: ajudikasi disagreement dan missing triples menggunakan
     LLM-as-a-judge berbasis konteks sumber.
   - Output: `02_consensus/*.consensus.repaired.json` dan
     `02_consensus/*_gold_standard.json`.

5. **Ingest Neo4j konsensus dan metrik Fase 1**
   - Input: KG konsensus, gold standard, dan penilaian awal pakar dari KG
     Review App.
   - Proses: ingest ke Neo4j dan perhitungan metrik struktural serta metrik
     validasi intra-mapel. Precision, recall, dan F1 dihitung dari perbandingan
     label awal pakar terhadap label final gold standard/konsensus; Gwet's AC1
     dihitung dari dua penilaian asli pakar.
   - Output: graf konsensus Neo4j, laporan metrik, precision, recall, F1, dan
     Gwet's AC1.

6. **Completion atas KG konsensus**
   - Input: KG konsensus (`{mapel}.consensus.json` /
     `{mapel}.consensus.repaired.json`).
   - Proses: completion berbasis embedding, ANN, dan LLM classifier.
   - Output: `03_completion/lintas_buku_edges.t070_k20.json`, berisi 125 edge
     `LINTAS_BUKU_*`.

7. **Evaluasi pasca-completion**
   - Input: KG konsensus + edge completion.
   - Proses: perhitungan metrik struktural pre/post-KGC dan diskusi panel pakar
     lintas-disiplin atas hasil completion.
   - Output: metrik post-KGC dan validasi Fase 2 atas 117 relasi yang masuk
     survei utama.

## Tahapan

| Tahap | Folder | Sumber | Model / Parameter | Jumlah |
|---|---|---|---|---|
| Step 1 — Ekstraksi | `01_extraction/` | final-kg/extracted | Gemini 2.5 Flash Lite; chunk 800/200 | konsep: Biologi 85, Fisika 134, Kimia 144 |
| Konsensus | `02_consensus/` | final-kg/final-consensus/kg_repaired | multi-judge + repair (PR #1) | 3 mapel (repaired) + gold standards + 6 validasi pakar |
| Step 2 — Completion | `03_completion/` | ann-classifier-v1-KG-TEGAR-FIXED | Gemini 2.5 Flash; embed gemini-embedding-001; threshold 0,70 / top_k 20 | 125 edge (PRASYARAT_UNTUK 57, MEMPERDALAM 31, BERKAITAN_DENGAN 21, APLIKASI_DARI 15, SAMA_DENGAN 1) |
| Final | `04_final/` | Neo4j final (diunggah penulis) | konsensus + 125 LINTAS_BUKU | lihat 04_final/README.md |

## Catatan provenans
- Konsensus utama = varian **repaired** (kg_repaired), bukan (2.5-pro) / (mock).
- Completion config final = **0,70 / k20** (125 edge); 8 edge confidence < 0,5 dikeluarkan dari survei (117 dinilai).
- Validasi pakar lintas-buku (Fase 2): diskusi panel atas 117 relasi (validitas 99,1% [116/117] / tipe 98,3% [115/117] / arah 75,9% [88/116]). Sumber otoritatif = `../completion/completion-survey.json`. Berkas `03_completion/lintas_buku_survey_responses.csv` adalah pilot awal 1 penilai (100/117) dan **bukan acuan** untuk angka final.
- Versi lain (extraction-v1..v4, consensus 2.5-pro/mock) bukan artefak acuan dan diarsipkan terpisah.
