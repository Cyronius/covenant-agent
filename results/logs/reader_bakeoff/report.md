# Reader bake-off: right tool among its rivals, zero-shot

Chance is 1 / (number of candidates): about 40-50% here.

| reader or model | params | lookalike | decoy, decidable | decoy, undecidable | decoy, unlabelled |
|---|---|---|---|---|---|
| chance | — | 50.0% | 41.3% | 41.0% | 41.2% |
| all-MiniLM-L6-v2 | 22.7M | 76.4% (n=212) | 86.6% (n=359) | 49.9% (n=587) | 67.1% (n=310) |
| all-MiniLM-L6-v2 +name | 22.7M | 77.4% (n=212) | 89.4% (n=359) | 47.2% (n=587) | 63.9% (n=310) |
| bge-small-en-v1.5 | 33.4M | 75.0% (n=212) | 91.1% (n=359) | 46.7% (n=587) | 66.1% (n=310) |
| bge-small-en-v1.5 +name | 33.4M | 76.9% (n=212) | 92.8% (n=359) | 48.7% (n=587) | 66.8% (n=310) |
| ternlight-base | 15.4M | 68.4% (n=212) | 90.5% (n=359) | 48.6% (n=587) | 67.1% (n=310) |
| ternlight-base +name | 15.4M | 72.2% (n=212) | 91.1% (n=359) | 54.0% (n=587) | 68.1% (n=310) |
| ternlight-mini | 9.5M | 70.3% (n=212) | 88.9% (n=359) | 52.0% (n=587) | 67.1% (n=310) |
| ternlight-mini +name | 9.5M | 72.6% (n=212) | 90.0% (n=359) | 53.7% (n=587) | 66.1% (n=310) |
| 27B, direct (generator) | — | 97.2% (committed 177/212) | 98.4% (committed 319/359) | 85.0% (committed 454/587) | 92.6% (committed 269/310) |
| 27B, reasons first (generator) | — | 100.0% (committed 82/96) | 100.0% (committed 47/56) | 89.8% (committed 98/135) | 91.5% (committed 82/99) |
| A0 options-off (generator) | — | 88.4% (committed 207/212) | 44.9% (committed 323/359) | 56.6% (committed 519/587) | 55.8% (committed 278/310) |
| SPt options-off (generator) | — | 93.2% (committed 190/212) | 38.1% (committed 294/359) | 75.2% (committed 500/587) | 64.7% (committed 255/310) |
| SPt 50% S6 (generator) | — | 80.0% (committed 160/212) | 91.4% (committed 324/359) | 86.7% (committed 368/587) | 77.2% (committed 237/310) |
| A0 25% decoys+twin (generator) | — | 62.1% (committed 169/212) | 59.3% (committed 324/359) | 76.0% (committed 329/587) | 69.2% (committed 237/310) |
| A0 25% flip decoys (generator) | — | 82.6% (committed 207/212) | 54.8% (committed 334/359) | 55.7% (committed 510/587) | 62.7% (committed 260/310) |
| A0+reader options-off (generator) | — | 88.6% (committed 210/212) | 85.0% (committed 321/359) | 60.2% (committed 525/587) | 70.2% (committed 275/310) |
| A0+reader 25% decoys+twin (generator) | — | 71.5% (committed 186/212) | 91.4% (committed 337/359) | 76.9% (committed 402/587) | 76.2% (committed 260/310) |
| A0+reader 25% flip decoys (generator) | — | 91.4% (committed 209/212) | 91.0% (committed 343/359) | 58.1% (committed 532/587) | 72.4% (committed 283/310) |
