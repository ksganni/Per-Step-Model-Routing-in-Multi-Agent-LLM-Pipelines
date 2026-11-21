# Summary tables

Generated 2026-11-21 15:25 UTC by `compute_metrics.py` from experiments/sql/all-flash.csv, experiments/sql/cheap-decomposer.csv, experiments/sql/cheap-refiner.csv, experiments/sql/cheap-selector.csv, experiments/qa_results.csv (sha256 prefixes: f3f8f0f7a0f0, dcb9b6ac5110, c16599b8bb16, 900b604d93c3, 2dd3406d00e7). 95% CIs: Wilson for proportions, Bonett-Price for paired accuracy differences, percentile bootstrap (10000 resamples, seed 42) for means and paired F1 and cost differences. Non-inferiority margin 0.05; no verdict below 20 paired questions.

## Per-config quality and cost

| pipeline | config | swap | tier | n | accuracy | accuracy 95% CI | mean F1 | F1 95% CI | EM | cost/q ($) | cost/q 95% CI |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qa | qa-all-strong | all-strong | strong | 107 | 0.710 | [0.618, 0.788] | 0.794 | [0.730, 0.853] | 0.607 | 0.02718 | [0.02529, 0.02921] |
| qa | qa-cheap-composer | composer → Flash-Lite | cheap | 98 | 0.714 | [0.618, 0.794] | 0.798 | [0.730, 0.861] | 0.643 | 0.02411 | [0.02229, 0.02612] |
| qa | qa-qwen-composer | composer → Qwen | local | 83 | 0.675 | [0.568, 0.766] | 0.770 | [0.693, 0.842] | 0.602 | 0.02427 | [0.02214, 0.02659] |
| qa | qa-cheap-decomposer | decomposer → Flash-Lite | cheap | 71 | 0.676 | [0.561, 0.773] | 0.799 | [0.722, 0.869] | 0.634 | 0.02003 | [0.01837, 0.02183] |
| qa | qa-qwen-decomposer | decomposer → Qwen | local | 23 | 0.565 | [0.368, 0.744] | 0.674 | [0.493, 0.841] | 0.565 | 0.02139 | [0.01825, 0.02520] |
| qa | qa-cheap-formatter | formatter → Flash-Lite | cheap | 93 | 0.710 | [0.611, 0.792] | 0.813 | [0.751, 0.872] | 0.634 | 0.02309 | [0.02120, 0.02515] |
| qa | qa-qwen-formatter | formatter → Qwen | local | 85 | 0.612 | [0.505, 0.708] | 0.719 | [0.636, 0.797] | 0.541 | 0.02337 | [0.02131, 0.02563] |
| qa | qa-cheap-hop-reader | hop_reader → Flash-Lite | cheap | 90 | 0.689 | [0.587, 0.775] | 0.783 | [0.711, 0.850] | 0.611 | 0.01712 | [0.01573, 0.01867] |
| qa | qa-qwen-hop-reader | hop_reader → Qwen | local | 43 | 0.651 | [0.502, 0.776] | 0.781 | [0.672, 0.876] | 0.651 | 0.01726 | [0.01516, 0.01965] |
| sql | all-flash | all-strong | strong | 14 | 0.786 | [0.524, 0.924] |  |  |  | 0.03752 | [0.03284, 0.04213] |
| sql | cheap-decomposer | decomposer → Flash-Lite | cheap | 14 | 0.643 | [0.388, 0.837] |  |  |  | 0.02227 | [0.01589, 0.03082] |
| sql | cheap-refiner | refiner → Flash-Lite | cheap | 14 | 0.786 | [0.524, 0.924] |  |  |  | 0.03752 | [0.03287, 0.04213] |
| sql | cheap-selector | selector → Flash-Lite | cheap | 14 | 0.643 | [0.388, 0.837] |  |  |  | 0.02653 | [0.01962, 0.03595] |

## Paired tests against all-strong

| pipeline | config | swap | n paired | Δ accuracy | Δ acc 95% CI | b/c | McNemar p | McNemar p (Holm) | Δ F1 | Δ F1 95% CI | bootstrap p | bootstrap p (Holm) | cost ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qa | qa-cheap-composer | composer → Flash-Lite | 98 | 0.000 | [-0.048, 0.048] | 2/2 | 1.000 | 1.000 | -0.007 | [-0.038, 0.018] | 0.611 | 1.000 | 0.88 |
| qa | qa-qwen-composer | composer → Qwen | 83 | -0.012 | [-0.081, 0.057] | 4/3 | 1.000 | 1.000 | -0.024 | [-0.071, 0.016] | 0.284 | 1.000 | 0.89 |
| qa | qa-cheap-decomposer | decomposer → Flash-Lite | 71 | -0.014 | [-0.074, 0.046] | 2/1 | 1.000 | 1.000 | -0.001 | [-0.029, 0.025] | 0.929 | 1.000 | 0.72 |
| qa | qa-qwen-decomposer | decomposer → Qwen | 23 | -0.130 | [-0.289, 0.049] | 3/0 | 0.250 | 1.000 | -0.150 | [-0.290, -0.043] | 0.022 | 0.153 | 0.77 |
| qa | qa-cheap-formatter | formatter → Flash-Lite | 93 | 0.000 | [-0.051, 0.051] | 2/2 | 1.000 | 1.000 | 0.006 | [-0.016, 0.030] | 0.619 | 1.000 | 0.86 |
| qa | qa-qwen-formatter | formatter → Qwen | 85 | -0.071 | [-0.131, -0.007] | 6/0 | 0.031 | 0.250 | -0.070 | [-0.129, -0.022] | 0.012 | 0.094 | 0.86 |
| qa | qa-cheap-hop-reader | hop_reader → Flash-Lite | 90 | -0.022 | [-0.074, 0.030] | 3/1 | 0.625 | 1.000 | -0.024 | [-0.064, 0.010] | 0.208 | 1.000 | 0.64 |
| qa | qa-qwen-hop-reader | hop_reader → Qwen | 43 | -0.047 | [-0.150, 0.061] | 3/1 | 0.625 | 1.000 | -0.031 | [-0.107, 0.036] | 0.399 | 1.000 | 0.61 |
| sql | cheap-decomposer | decomposer → Flash-Lite | 14 | -0.143 | [-0.362, 0.112] | 2/0 | 0.500 | 1.000 |  |  |  |  | 0.59 |
| sql | cheap-refiner | refiner → Flash-Lite | 14 | 0.000 | [-0.173, 0.173] | 0/0 | 1.000 | 1.000 |  |  |  |  | 1.00 |
| sql | cheap-selector | selector → Flash-Lite | 14 | -0.143 | [-0.362, 0.112] | 2/0 | 0.500 | 1.000 |  |  |  |  | 0.71 |

## Error propagation by step and model tier

| pipeline | step | model | tier | config | baseline right | became wrong | right→wrong rate | 95% CI | baseline wrong | wrong→right rate | error_propagated rate | 95% CI | n judged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qa | composer | Flash-Lite | cheap | qa-cheap-composer | 70 | 2 | 0.029 | [0.008, 0.098] | 28 | 0.071 |  |  | 0 |
| qa | composer | Qwen | local | qa-qwen-composer | 57 | 4 | 0.070 | [0.028, 0.167] | 26 | 0.115 |  |  | 0 |
| qa | decomposer | Flash-Lite | cheap | qa-cheap-decomposer | 49 | 2 | 0.041 | [0.011, 0.137] | 22 | 0.045 |  |  | 0 |
| qa | decomposer | Qwen | local | qa-qwen-decomposer | 16 | 3 | 0.188 | [0.066, 0.430] | 7 | 0.000 |  |  | 0 |
| qa | formatter | Flash-Lite | cheap | qa-cheap-formatter | 66 | 2 | 0.030 | [0.008, 0.104] | 27 | 0.074 |  |  | 0 |
| qa | formatter | Qwen | local | qa-qwen-formatter | 58 | 6 | 0.103 | [0.048, 0.208] | 27 | 0.000 |  |  | 0 |
| qa | hop_reader | Flash-Lite | cheap | qa-cheap-hop-reader | 64 | 3 | 0.047 | [0.016, 0.129] | 26 | 0.038 |  |  | 0 |
| qa | hop_reader | Qwen | local | qa-qwen-hop-reader | 30 | 3 | 0.100 | [0.035, 0.256] | 13 | 0.077 |  |  | 0 |
| sql | decomposer | Flash-Lite | cheap | cheap-decomposer | 11 | 2 | 0.182 | [0.051, 0.477] | 3 | 0.000 | 0.143 | [0.040, 0.399] | 14 |
| sql | refiner | Flash-Lite | cheap | cheap-refiner | 11 | 0 | 0.000 | [0.000, 0.259] | 3 | 0.000 | 0.000 | [0.000, 0.215] | 14 |
| sql | selector | Flash-Lite | cheap | cheap-selector | 11 | 2 | 0.182 | [0.051, 0.477] | 3 | 0.000 | 0.143 | [0.040, 0.399] | 14 |

## Cost-quality Pareto frontier

| pipeline | config | quality metric | n | quality | cost/q ($) | on frontier | dominated by | quality (common ids) | cost/q (common ids) | on frontier (common ids) |
|---|---|---|---|---|---|---|---|---|---|---|
| qa | qa-all-strong | mean_f1 | 107 | 0.794 | 0.02718 | no | qa-cheap-composer, qa-cheap-decomposer, qa-cheap-formatter | 0.824 | 0.02765 | no |
| qa | qa-cheap-composer | mean_f1 | 98 | 0.798 | 0.02411 | no | qa-cheap-decomposer, qa-cheap-formatter | 0.812 | 0.02511 | no |
| qa | qa-qwen-composer | mean_f1 | 83 | 0.770 | 0.02427 | no | qa-cheap-composer, qa-cheap-decomposer, qa-cheap-formatter, qa-cheap-hop-reader, qa-qwen-hop-reader | 0.806 | 0.02492 | no |
| qa | qa-cheap-decomposer | mean_f1 | 71 | 0.799 | 0.02003 | yes |  | 0.812 | 0.01876 | yes |
| qa | qa-qwen-decomposer | mean_f1 | 23 | 0.674 | 0.02139 | no | qa-cheap-decomposer, qa-cheap-hop-reader, qa-qwen-hop-reader | 0.674 | 0.02139 | no |
| qa | qa-cheap-formatter | mean_f1 | 93 | 0.813 | 0.02309 | yes |  | 0.824 | 0.02451 | yes |
| qa | qa-qwen-formatter | mean_f1 | 85 | 0.719 | 0.02337 | no | qa-cheap-decomposer, qa-cheap-formatter, qa-cheap-hop-reader, qa-qwen-hop-reader | 0.780 | 0.02438 | no |
| qa | qa-cheap-hop-reader | mean_f1 | 90 | 0.783 | 0.01712 | yes |  | 0.746 | 0.01820 | no |
| qa | qa-qwen-hop-reader | mean_f1 | 43 | 0.781 | 0.01726 | no | qa-cheap-hop-reader | 0.746 | 0.01719 | yes |
| sql | all-flash | accuracy | 14 | 0.786 | 0.03752 | yes |  |  |  |  |
| sql | cheap-decomposer | accuracy | 14 | 0.643 | 0.02227 | yes |  |  |  |  |
| sql | cheap-refiner | accuracy | 14 | 0.786 | 0.03752 | yes |  |  |  |  |
| sql | cheap-selector | accuracy | 14 | 0.643 | 0.02653 | no | cheap-decomposer |  |  |  |

## Step findings

| pipeline | step | model | n paired | metric | Δ vs all-strong | 95% CI | p (Holm) | cost ratio | right→wrong | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| qa | composer | Flash-Lite | 98 | mean_f1 | -0.007 | [-0.038, 0.018] | 1.000 | 0.88 | 0.029 | non-inferior |
| qa | composer | Qwen | 83 | mean_f1 | -0.024 | [-0.071, 0.016] | 1.000 | 0.89 | 0.070 | inconclusive |
| qa | decomposer | Flash-Lite | 71 | mean_f1 | -0.001 | [-0.029, 0.025] | 1.000 | 0.72 | 0.041 | non-inferior |
| qa | decomposer | Qwen | 23 | mean_f1 | -0.150 | [-0.290, -0.043] | 0.153 | 0.77 | 0.188 | worse (unadjusted) |
| qa | formatter | Flash-Lite | 93 | mean_f1 | 0.006 | [-0.016, 0.030] | 1.000 | 0.86 | 0.030 | non-inferior |
| qa | formatter | Qwen | 85 | mean_f1 | -0.070 | [-0.129, -0.022] | 0.094 | 0.86 | 0.103 | worse (unadjusted) |
| qa | hop_reader | Flash-Lite | 90 | mean_f1 | -0.024 | [-0.064, 0.010] | 1.000 | 0.64 | 0.047 | inconclusive |
| qa | hop_reader | Qwen | 43 | mean_f1 | -0.031 | [-0.107, 0.036] | 1.000 | 0.61 | 0.100 | inconclusive |
| sql | decomposer | Flash-Lite | 14 | accuracy | -0.143 | [-0.362, 0.112] | 1.000 | 0.59 | 0.182 | too few |
| sql | refiner | Flash-Lite | 14 | accuracy | 0.000 | [-0.173, 0.173] | 1.000 | 1.00 | 0.000 | too few |
| sql | selector | Flash-Lite | 14 | accuracy | -0.143 | [-0.362, 0.112] | 1.000 | 0.71 | 0.182 | too few |

## Data quality

- qa: the common-ids frontier is based on the 23 question_ids every config answered
- sql: fewer than 20 questions for all-flash, cheap-decomposer, cheap-refiner, cheap-selector; treat those intervals as indicative only
- sql: only 14 question_ids shared by every config (< 20); common-ids frontier not computed
