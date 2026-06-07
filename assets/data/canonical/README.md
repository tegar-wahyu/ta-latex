# Paper Data (Canonical)

Data kanonik yang digunakan paper, satu artefak otoritatif per tahap pipeline.

## Tahapan

| Tahap | Folder | Sumber | Model / Parameter | Jumlah |
|---|---|---|---|---|
| Step 1 — Ekstraksi | `01_extraction/` | final-kg/extracted | Gemini 2.5 Flash Lite; chunk 800/200 | konsep: Biologi 85, Fisika 134, Kimia 144 |
| Konsensus | `02_consensus/` | final-kg/final-consensus/kg_repaired | multi-judge + repair (PR #1) | 3 mapel (repaired) + gold standards + 6 validasi pakar |
| Step 2 — Completion | `03_completion/` | ann-classifier-v1-KG-TEGAR-FIXED | Gemini 2.5 Flash; embed gemini-embedding-001; threshold 0,70 / top_k 20 | 125 edge (PRASYARAT_UNTUK 57, MEMPERDALAM 31, BERKAITAN_DENGAN 21, APLIKASI_DARI 15, SAMA_DENGAN 1) |
| Final | `04_final/` | yhoga Neo4j (diunggah penulis) | konsensus + 125 LINTAS_BUKU | lihat 04_final/README.md |

## Catatan provenans
- Konsensus kanonik = varian **repaired** (kg_repaired), bukan (2.5-pro) / (mock).
- Completion config final = **0,70 / k20** (125 edge); 8 edge confidence < 0,5 dikeluarkan dari survei (117 dinilai).
- Validasi pakar lintas-buku (Fase 2): diskusi panel atas 117 relasi (validitas 99,1% [116/117] / tipe 98,3% [115/117] / arah 75,9% [88/116]). Sumber otoritatif = `../completion/completion-survey.json`. Berkas `03_completion/lintas_buku_survey_responses.csv` adalah pilot awal 1 penilai (100/117) dan **TIDAK kanonik** untuk angka final.
- Versi lain (extraction-v1..v4, consensus 2.5-pro/mock) BUKAN kanonik — diarsipkan terpisah.
