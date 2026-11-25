# Evaluation during pretraining

Settings (fp32, PyTorch reference kernels):

- Multiple choice: first 1000 items of each set, 0-shot; acc_norm for HellaSwag and ARC-Challenge, acc for the others
- MMLU: 10 items per subject (570 in all), 5-shot, answer letter
- GSM8K: 100 items, 5-shot, greedy, flexible answer match
- HumanEval: all 164 problems, greedy pass@1
- MBPP (`mbpp_*.json`): 500 test problems, 3-shot, greedy pass@1

Numbers are percentages. Validation loss is the latest training-time validation before that step. The learning rate starts to decay at step 38148, when the data mix also switches to the annealing mix.

| Step | Tokens (B) | HellaSwag | ARC-E | ARC-C | PIQA | Winogrande | MMLU | GSM8K | HumanEval | Weighted val loss |
|---|---|---|---|---|---|---|---|---|---|---|
| 17700 | 9.3 | 44.6 | 58.9 | 29.2 | 67.8 | 54.3 | 26.1 | 2.0 | 11.6 | 2.1082 |
| 20000 | 10.5 | 45.5 | 58.1 | 28.6 | 67.5 | 52.2 | 26.7 | 2.0 | 13.4 | 2.0974 |
| 25000 | 13.1 | 46.2 | 59.9 | 27.3 | 66.6 | 52.5 | 25.8 | 1.0 | 13.4 | 2.0770 |
| 30000 | 15.7 | 46.0 | 58.9 | 30.0 | 68.2 | 55.2 | 28.9 | 7.0 | 17.1 | 2.0675 |
| 35000 | 18.4 | 45.6 | 62.0 | 30.3 | 68.2 | 54.1 | 25.8 | 4.0 | 15.2 | 2.0509 |
| 40000 | 21.0 | 48.4 | 62.7 | 30.7 | 69.3 | 55.3 | 26.8 | 9.0 | 18.9 | 1.9807 |
| 45000 | 23.6 | 49.5 | 63.9 | 30.4 | 70.6 | 57.2 | 26.1 | 14.0 | 28.0 | 1.8802 |
| 47684 | 25.0 | 49.8 | 65.7 | 31.4 | 71.7 | 58.1 | 23.5 | 14.0 | 31.1 | 1.8488 |
