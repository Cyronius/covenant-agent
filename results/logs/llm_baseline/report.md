
## plain exam (300 tasks)

| model | n | goal | compiles | called: unique | called: lookalike | called: decoy, decidable | called: decoy, undecidable | called: decoy, unlabelled | swapped: lookalike | swapped: decoy, decidable | swapped: decoy, undecidable | swapped: decoy, unlabelled | wrong picks at lookalike and decoy calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 27B, direct (Cerebras) | 300 | 50.7% | 81.0% | 89.1% (n=229) | 77.5% (n=298) | — | — | 97.6% (n=41) | 1.7% | — | — | 0.0% | another tool 51%, no call there 25%, the lookalike 21%, a generic helper 3% |
| 27B, reasons first (Cerebras) | 150 | 74.0% | 93.3% | 87.0% (n=115) | 71.8% (n=142) | — | — | 87.0% (n=23) | 0.0% | — | — | 0.0% | no call there 60%, the lookalike 21%, another tool 19% |
| A0 options-off | 300 | 74.3% | 86.0% | 93.9% (n=229) | 88.9% (n=298) | — | — | 92.7% (n=41) | 8.1% | — | — | 0.0% | the lookalike 72%, another tool 28% |
| A0 options-off + backoff | 300 | 82.3% | 95.3% | 93.9% (n=229) | 98.0% (n=298) | — | — | 92.7% (n=41) | 0.3% | — | — | 0.0% | another tool 89%, the lookalike 11% |
| SPt options-off | 300 | 66.0% | 81.0% | 87.3% (n=229) | 86.2% (n=298) | — | — | 90.2% (n=41) | 4.4% | — | — | 2.4% | another tool 51%, the lookalike 33%, a generic helper 9%, no call there 7% |
| SPt 25% S6 | 300 | 53.3% | 61.7% | 91.3% (n=229) | 63.4% (n=298) | — | — | 85.4% (n=41) | 9.1% | — | — | 0.0% | the lookalike 37%, another tool 33%, a generic helper 30% |
| SPt 50% S6 | 300 | 53.7% | 60.7% | 92.6% (n=229) | 61.1% (n=298) | — | — | 82.9% (n=41) | 10.7% | — | — | 0.0% | another tool 46%, the lookalike 37%, a generic helper 17%, no call there 1% |
| A0 25% decoys+twin (R19) | 300 | 48.0% | 55.0% | 88.2% (n=229) | 55.4% (n=298) | — | — | 78.0% (n=41) | 21.5% | — | — | 0.0% | the lookalike 58%, a generic helper 20%, another tool 20%, no call there 2% |
| A0 25% flip decoys | 300 | 70.3% | 80.3% | 92.6% (n=229) | 85.9% (n=298) | — | — | 95.1% (n=41) | 12.1% | — | — | 0.0% | the lookalike 84%, another tool 16% |
| A0+reader options-off | 300 | 75.3% | 86.7% | 93.9% (n=229) | 89.6% (n=298) | — | — | 90.2% (n=41) | 8.1% | — | — | 0.0% | the lookalike 80%, another tool 11%, a generic helper 6%, no call there 3% |
| A0+reader options-off + backoff | 300 | 81.7% | 97.0% | 93.9% (n=229) | 98.7% (n=298) | — | — | 90.2% (n=41) | 0.3% | — | — | 0.0% | another tool 50%, the lookalike 25%, a generic helper 12%, no call there 12% |
| A0+reader 25% decoys+twin | 300 | 55.3% | 65.0% | 89.5% (n=229) | 66.1% (n=298) | — | — | 85.4% (n=41) | 17.8% | — | — | 0.0% | the lookalike 61%, another tool 20%, a generic helper 19%, no call there 1% |
| A0+reader 25% flip decoys | 300 | 76.0% | 89.3% | 93.9% (n=229) | 91.9% (n=298) | — | — | 92.7% (n=41) | 6.0% | — | — | 0.0% | the lookalike 81%, another tool 15%, no call there 4% |
| A0+reader 25% flip decoys + backoff | 300 | 81.0% | 95.3% | 93.9% (n=229) | 97.7% (n=298) | — | — | 92.7% (n=41) | 1.7% | — | — | 0.0% | the lookalike 60%, another tool 30%, no call there 10% |

## decoy exam (300 tasks)

| model | n | goal | compiles | called: unique | called: lookalike | called: decoy, decidable | called: decoy, undecidable | called: decoy, unlabelled | swapped: lookalike | swapped: decoy, decidable | swapped: decoy, undecidable | swapped: decoy, unlabelled | wrong picks at lookalike and decoy calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 27B, direct (Cerebras) | 300 | 42.3% | 82.0% | 95.5% (n=22) | — | 89.6% (n=115) | 63.1% (n=282) | 76.5% (n=149) | — | 3.5% | 12.8% | 11.4% | its decoy 34%, no call there 31%, another tool 26%, the lookalike 8%, a generic helper 1% |
| 27B, reasons first (Cerebras) | 150 | 62.0% | 96.7% | 92.3% (n=13) | — | 85.7% (n=56) | 65.2% (n=135) | 73.7% (n=76) | — | 0.0% | 7.4% | 9.2% | no call there 56%, its decoy 20%, another tool 16%, the lookalike 8% |
| A0 options-off | 300 | 29.3% | 86.3% | 45.5% (n=22) | — | 42.6% (n=115) | 48.9% (n=282) | 40.9% (n=149) | — | 49.6% | 39.7% | 49.7% | its decoy 79%, the lookalike 12%, another tool 8%, no call there 1% |
| SPt options-off | 300 | 38.7% | 79.3% | 68.2% (n=22) | — | 41.7% (n=115) | 61.7% (n=282) | 51.0% (n=149) | — | 44.3% | 23.8% | 34.2% | its decoy 65%, another tool 16%, the lookalike 11%, no call there 8%, a generic helper 0% |
| SPt 25% S6 | 300 | 42.7% | 63.0% | 95.5% (n=22) | — | 76.5% (n=115) | 56.0% (n=282) | 65.8% (n=149) | — | 16.5% | 10.3% | 17.4% | its decoy 31%, another tool 28%, the lookalike 21%, a generic helper 14%, no call there 5% |
| SPt 50% S6 | 300 | 44.0% | 61.3% | 90.9% (n=22) | — | 84.3% (n=115) | 54.3% (n=282) | 63.1% (n=149) | — | 7.8% | 8.9% | 18.1% | another tool 37%, its decoy 27%, the lookalike 24%, a generic helper 7%, no call there 5% |
| A0 25% decoys+twin (R19) | 300 | 34.0% | 57.0% | 81.8% (n=22) | — | 54.8% (n=115) | 45.0% (n=282) | 54.4% (n=149) | — | 35.7% | 14.2% | 22.8% | its decoy 39%, the lookalike 31%, another tool 16%, a generic helper 11%, no call there 4% |
| A0 25% flip decoys | 300 | 31.3% | 79.0% | 45.5% (n=22) | — | 55.7% (n=115) | 47.5% (n=282) | 48.3% (n=149) | — | 39.1% | 38.3% | 38.9% | its decoy 73%, the lookalike 18%, another tool 7%, no call there 2% |
| A0+reader options-off | 300 | 43.3% | 85.7% | 50.0% (n=22) | — | 71.3% (n=115) | 53.5% (n=282) | 57.7% (n=149) | — | 20.0% | 36.9% | 35.6% | its decoy 77%, the lookalike 14%, another tool 7%, no call there 2%, a generic helper 0% |
| A0+reader 25% decoys+twin | 300 | 43.3% | 69.0% | 77.3% (n=22) | — | 84.3% (n=115) | 52.5% (n=282) | 62.4% (n=149) | — | 11.3% | 16.7% | 22.8% | its decoy 41%, the lookalike 34%, another tool 15%, a generic helper 8%, no call there 2% |
| A0+reader 25% flip decoys | 300 | 45.0% | 88.0% | 50.0% (n=22) | — | 81.7% (n=115) | 52.8% (n=282) | 62.4% (n=149) | — | 13.9% | 38.3% | 29.5% | its decoy 76%, the lookalike 14%, another tool 7%, no call there 3% |

## flip exam (300 tasks)

| model | n | goal | compiles | called: unique | called: lookalike | called: decoy, decidable | called: decoy, undecidable | called: decoy, unlabelled | swapped: lookalike | swapped: decoy, decidable | swapped: decoy, undecidable | swapped: decoy, unlabelled | wrong picks at lookalike and decoy calls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 27B, direct (Cerebras) | 300 | 46.0% | 83.0% | — | — | 93.4% (n=244) | 68.5% (n=305) | 80.8% (n=120) | — | 0.4% | 10.5% | 2.5% | another tool 41%, its decoy 23%, no call there 18%, the lookalike 17%, a generic helper 1% |
| A0 options-off | 300 | 17.3% | 86.7% | — | — | 41.8% (n=244) | 52.8% (n=305) | 46.7% (n=120) | — | 49.6% | 37.0% | 40.8% | its decoy 80%, another tool 10%, the lookalike 9%, no call there 1% |
| SPt options-off | 300 | 17.0% | 80.0% | — | — | 28.7% (n=244) | 68.9% (n=305) | 43.3% (n=120) | — | 53.7% | 18.7% | 31.7% | its decoy 64%, another tool 22%, the lookalike 8%, no call there 6%, a generic helper 0% |
| SPt 25% S6 | 300 | 30.3% | 55.0% | — | — | 70.5% (n=244) | 60.0% (n=305) | 47.5% (n=120) | — | 22.1% | 6.2% | 25.0% | its decoy 35%, another tool 26%, the lookalike 23%, a generic helper 10%, no call there 7% |
| SPt 50% S6 | 300 | 32.0% | 51.7% | — | — | 82.4% (n=244) | 56.4% (n=305) | 45.8% (n=120) | — | 7.8% | 7.9% | 22.5% | another tool 38%, its decoy 27%, the lookalike 20%, no call there 8%, a generic helper 7% |
| A0 25% decoys+twin (R19) | 300 | 21.0% | 49.3% | — | — | 54.1% (n=244) | 43.0% (n=305) | 44.2% (n=120) | — | 37.3% | 12.8% | 32.5% | its decoy 45%, the lookalike 29%, another tool 13%, a generic helper 9%, no call there 3% |
| A0 25% flip decoys | 300 | 21.0% | 81.7% | — | — | 51.2% (n=244) | 51.1% (n=305) | 47.5% (n=120) | — | 43.4% | 38.7% | 32.5% | its decoy 77%, the lookalike 13%, another tool 6%, no call there 4% |
| A0+reader options-off | 300 | 37.7% | 84.3% | — | — | 78.7% (n=244) | 55.7% (n=305) | 60.0% (n=120) | — | 10.2% | 34.4% | 24.2% | its decoy 66%, the lookalike 14%, another tool 14%, no call there 6% |
| A0+reader 25% decoys+twin | 300 | 37.7% | 63.0% | — | — | 87.3% (n=244) | 54.4% (n=305) | 60.8% (n=120) | — | 6.6% | 15.1% | 23.3% | the lookalike 40%, its decoy 38%, another tool 15%, a generic helper 5%, no call there 1% |
| A0+reader 25% flip decoys | 300 | 39.0% | 87.0% | — | — | 90.2% (n=244) | 53.4% (n=305) | 62.5% (n=120) | — | 6.1% | 37.7% | 28.3% | its decoy 74%, the lookalike 18%, another tool 7%, no call there 0% |
