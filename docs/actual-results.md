# Actual Results

## Metrics (KG vs KGC)

### Topologi

| Metric | KG | KGC |
|---|---:|---:|
| ADC | 2.4123006833712983 | 2.6970387243735794 |
| ADC (Concept to Concept) | 0.7383720930232559 | 1.465116279069767 |
| Density | 0.001076344158926029 | 0.0021357380161366873 |
| Modularity | 0.8106482512809089 | 0.3731508875739645 |

### Ekstraksi

| Metric | KG | KGC |
|---|---:|---:|
| Description Completeness | 0.9307479224376731 | 0.9307479224376731 |
| Empty SubKonsep Rate | 0.0 | 0.0 |

### Hubungan

| Metric | KG | KGC |
|---|---:|---:|
| Typed Relation Ratio | 0.6968698517298187 | 0.7486338797814208 |

---

## Human-AI Evaluation

### Phase 1: KG

#### Gwet's AC1

| Subject | Items compared | Observed agreement | Chance agreement | Gwet's AC1 |
|---|---:|---:|---:|---:|
| Biologi | 151 | 0.934 | 0.040 | 0.931 |
| Fisika | 240 | 0.771 | 0.069 | 0.754 |
| Kimia | 166 | 0.651 | 0.112 | 0.607 |

#### Reviewed existing triples (summary)
| Subject | Actor | Reviewed | Correct | Partial | Wrong | Missing label | Missing Triples | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Biologi | Expert 1 | 151 | 140 | 6 | 2 | 3 | 0 | 0.952 | 0.965 | 0.958 | # Pendidikan Biologi UPI 2022
| Biologi | Expert 2 | 151 | 143 | 6 | 0 | 2 | 0 | 0.970 | 0.970 | 0.970 | # Biologi UI 2022
| Biologi | Final consensus | 151 | 144 | 4 | 1 | 2 | 0 | 0.970 | 0.977 | 0.973 |
| Fisika | Expert 1 | 240 | 239 | 0 | 0 | 1 | 15 | 0.997 | 0.938 | 0.967 | # Pendidikan Fisika UPI 2023
| Fisika | Expert 2 | 240 | 186 | 40 | 3 | 11 | 0 | 0.870 | 0.881 | 0.875 | # Fisika UGM 2022
| Fisika | Final consensus | 235 | 215 | 18 | 2 | 0 | 10 | 0.953 | 0.922 | 0.937 |
| Kimia | Expert 1 | 166 | 132 | 7 | 25 | 2 | 0 | 0.819 | 0.965 | 0.886 | # Pendidikan Kimia UPI 2022
| Kimia | Expert 2 | 166 | 136 | 6 | 5 | 19 | 29 | 0.866 | 0.757 | 0.808 | # Kimia ITB 2022
| Kimia | Final consensus | 158 | 135 | 6 | 15 | 2 | 19 | 0.877 | 0.855 | 0.866 |
| Overall | Expert 1 | 557 | 511 | 13 | 27 | 6 | 15 | 0.932 | 0.952 | 0.942 |
| Overall | Expert 2 | 557 | 465 | 52 | 8 | 32 | 29 | 0.896 | 0.863 | 0.879 |
| Overall | Final consensus | 544 | 494 | 28 | 18 | 4 | 29 | 0.936 | 0.917 | 0.926 |

### Phase 2: KGC (cross-book relations)

Source: `completion/completion-survey.json` (panel-consensus answers) +
`completion/lcompletion-edges.csv` (117 surveyed edges).
Final config: threshold = 0.70, top-k = 20. Of 125 generated edges, 8 with
confidence < 0.5 (all BERKAITAN_DENGAN) were excluded → 117 surveyed.

#### Per-aspect validation (n = 117)

| Aspect | Result | Rate |
|---|---|---:|
| Validity (relasi valid?) | 116 Ya / 1 Tidak | 99.1% |
| Type correctness (tipe benar?) | 115 Benar / 2 Salah | 98.3% |
| Direction (arah benar?) | 88 Benar / 28 Terbalik / 1 NA | 75.9% (88/116) |

- Invalid edge: LB005
- Wrong-type edges: LB008, LB074
- NA edge: LB085 (MEMPERDALAM, panel could not judge direction — lacked context)
- Direction rate excludes the 1 NA from the denominator (88 / (88+28) = 116).

#### Relation-type distribution (surveyed, n = 117)

| Type | Edges | Share |
|---|---:|---:|
| PRASYARAT_UNTUK | 57 | 48.7% |
| MEMPERDALAM | 31 | 26.5% |
| APLIKASI_DARI | 15 | 12.8% |
| BERKAITAN_DENGAN | 13 | 11.1% |
| SAMA_DENGAN | 1 | 0.9% |

(Note: the bab-5 §5.3 distribution percentages — 45.6/24.8/16.8/12.0/0.8 — are
computed over the 125 generated edges, not the 117 surveyed.)

#### Direction accuracy by relation type

| Type | Benar | Terbalik | NA | Accuracy |
|---|---:|---:|---:|---:|
| PRASYARAT_UNTUK | 45 | 12 | 0 | 78.9% |
| MEMPERDALAM | 18 | 12 | 1 | 60.0% |
| APLIKASI_DARI | 11 | 4 | 0 | 73.3% |
| BERKAITAN_DENGAN (symmetric) | 13 | 0 | 0 | 100% |
| SAMA_DENGAN (symmetric) | 1 | 0 | 0 | 100% |

#### Reversed edges by subject pair (28 total)

| Pair | Reversed |
|---|---:|
| Biologi ↔ Kimia | 26 |
| Fisika ↔ Kimia | 2 |

#### Cross-book bridges by subject pair (surveyed, n = 117)

| Pair | Edges | Share |
|---|---:|---:|
| Biologi ↔ Kimia | 91 | 77.8% |
| Fisika ↔ Kimia | 21 | 17.9% |
| Biologi ↔ Fisika | 5 | 4.3% |

#### Parameter sweep (edge counts; not yet pakar-validated except final)

| Config (threshold / top-k) | Edges |
|---|---:|
| 0.70 / 20 (final, surveyed) | 125 |
| 0.75 / 15 | 72 |
| 0.80 / 10 | 42 |
| 0.85 / 5 | 14 |

#### Confidence distribution (125 generated edges)

Source: `completion/lintas_buku_edges.t070_k20.json` (has per-edge `confidence`).

| Band | Edges | Notes |
|---|---:|---|
| confidence = 1.0 | 20 (16%) | high-certainty |
| confidence < 0.5 | 8 | all BERKAITAN_DENGAN → excluded from survey (→117) |
| confidence < 0.7 | 9 | the 8 above + one at 0.6 (also BERKAITAN_DENGAN) |
| confidence = 0.7 | 13 | borderline |

Verified false-positive examples (low confidence, all BERKAITAN_DENGAN unless noted):
- Gerbang Logika Dasar [Fisika] ↔ Notasi Sel Standar [Kimia] — conf 0.2
- Klasifikasi Semikonduktor [Fisika] ↔ Polimer Anorganik [Kimia] — conf 0.6
- Efek Fotolistrik [Fisika] ↔ Fotosintesis [Biologi] — conf 0.7, type **MEMPERDALAM**
  (note: conf 0.7 is *not* < 0.7, so this is a borderline case, not one of the 9)
