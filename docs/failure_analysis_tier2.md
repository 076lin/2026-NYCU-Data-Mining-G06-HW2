# 7 Results

## 7.x Why Tier 2 Wins and Loses

1. **Best ranking, worst F1@τ: the threshold is calibrated on GoldenEye.** On Friday, Tier 2 ranks flows best of the three tiers (AUROC 0.921 vs. 0.762 for Tier 1 and 0.858 for Tier 3). Yet its F1@τ is the lowest (0.200), because it flags only 11.7% of attacks. The cause is the validation set: 82% of its attacks are DoS GoldenEye, which lies far from normal traffic. Max-F1 therefore picks a strict τ (validation FPR 0.26%, F1 0.653). On Friday the same τ sits above almost every new attack. Compared post hoc at Tier 1's FPR (20.5%), Tier 2 catches 99.4% of attacks against Tier 1's 44.3%. Its hindsight-optimal F1 is 0.91. A Shapley decomposition of the validation-to-Friday F1 change attributes −0.447 to the change in attack families and only +0.131 to the rise in prevalence. Prevalence helps Tier 2 little because its false-positive rate is already low.

2. **Unseen attacks look like normal traffic.** PortScan and Bot fall inside the dense normal region of a PCA fitted on normal training flows. Their nearest training flow is benign for 99% and 100% of samples. The typical PortScan score (2.29) sits at the 90th percentile of benign Friday flows, far below τ = 5.09. Tier 2 still ranks these families above most benign flows (AUROC 0.899 for PortScan, 0.752 for Bot), but not above the validation threshold. Recall@τ is 0.2% for PortScan, 0.8% for Bot and 26.2% for DDoS.

3. **Most Friday false positives are label errors.** Attack flows whose direction was swapped are labelled BENIGN (2, Engelen et al.). On Friday, 32,314 BENIGN flows run between the victim web server (192.168.10.50:80) and the attacker NAT address (172.16.0.1), 97% of them inside the DDoS window. Tier 2 flags 42% of these flows, and they make up 94.5% of its false positives. Excluding them (post hoc) has these effects:
   - Tier 2's FPR falls from 3.48% to 0.20%.
   - Its precision rises from 0.71 to 0.98, and its AP from 0.80 to 0.96.
   - The "normal traffic" term of the F1 decomposition moves from −0.137 to +0.006.

   Tier 2's only real loss on Friday is therefore the change in attack families. Tier 1 and Tier 3 are barely affected (1.1% and 3.5% of their false positives).

# 8 Analysis

## 8.x Tier 2 Error Analysis

Cases come from seed 20261104 (τ = 5.0943) and are chosen by fixed rules, not by eye. The kNN score s(x) is the Euclidean distance to the 5th nearest normal reference flow. Its square splits exactly into per-feature terms, so each feature's share of the distance is exact (Fig. 9).

1. **Timing features drive the decision, not attack semantics.** The first principal component of normal traffic (71% of variance) is Idle/Active time, so Euclidean distance is dominated by timing features. Timing is the largest term in five of the six cases. DDoS flows #2723650 (TP, s = 6.46) and #2687946 (FN, s = 5.0942, short of τ by 1.1e-4) are the same kind of flow: 4-packet half-open HTTP requests with no server response. Whether each crosses τ is decided by Active Mean/Min/Max (≈68–88% of the distance), not by the missing response. An analyst cannot trust such an explanation.

2. **Single-flow view.** PortScan and Bot are anomalous only in context: one source probing many ports, or periodic beaconing to one host. Each single flow looks like a small benign flow.
   - PortScan flow #2454412 (455 µs, 3 packets, s = 2.29 < τ) is a complete inversion, not a near miss. Its distance is spread over flag and response-size features, with no dominant term.
   - Bot C&C flow #2286836 (s = 1.69) differs from its nearest normal neighbour almost only in Flow IAT Min (95% of the distance).

3. **Rare is not malicious.** Benign flow #2132365, a protocol-0 flow between two internal hosts, scores s = 12.58 (2.5τ), the highest benign score on Friday. 63% of its distance comes from Idle Std. A one-class model flags any legitimate traffic that the training week never contained.

**First fix.** Add per-source windowed features, such as distinct destination ports and connection periodicity within 60 s. These target the failure mode in case 2. Relabel victim-to-attacker response flows before evaluation.

## Supporting Tables

Unless marked otherwise, numbers are 3-seed means (20261104–06) with the original labels. Post hoc means the number uses Friday labels and is not used for model selection.

**T1. Tier 2 on validation and test (§7.x-1)**

| | Recall@τ | FPR@τ | F1@τ | AP | AUROC |
|---|---:|---:|---:|---:|---:|
| Validation (prevalence 1.6%) | 0.561 | 0.26% | 0.653 | 0.449 | 0.906 |
| Friday (prevalence 41.1%) | 0.117 | 3.48% | 0.200 | 0.798 | 0.921 |

**T2. Test recall at matched FPR (§7.x-1; post hoc)**

The Tier 1 and Tier 3 numbers come from the Tier 3 analysis (S1, S2).

| | At Tier 1's FPR (20.5%) | At Tier 3's FPR (5.6%) | Best F1, hindsight τ |
|---|---:|---:|---:|
| Tier 1 | 0.443 | 0.000 | |
| Tier 2 kNN | **0.994** | **0.364** | 0.91 |
| Tier 3 XGBoost | 0.712 | 0.297 | |

**T3. Shapley decomposition of the F1 change, validation → Friday (§7.x-1, §7.x-3)**

| | Val F1 | Attack mix | Normal traffic | Prevalence | Friday F1 |
|---|---:|---:|---:|---:|---:|
| Original labels | 0.653 | −0.447 | −0.137 | +0.131 | 0.200 |
| Suspect flows excluded (post hoc) | 0.653 | −0.491 | +0.006 | +0.040 | 0.209 |

**T4. Recall and AUROC by family (§7.x-2)**

The AUROC for each family is computed against Friday benign flows.

| Family | Share of test attacks | Nearest training flow benign | Recall@τ | AUROC |
|---|---:|---:|---:|---:|
| PortScan | 55.0% | 99% | 0.002 | 0.899 |
| DDoS | 44.3% | 85% | 0.262 | 0.952 |
| Bot | 0.7% | 100% | 0.008 | 0.752 |

**T5. Effect of the 32,314 suspect flows (§7.x-3; post hoc)**

| | FPs that are suspect flows | FPR: original → excluded | Precision: original → excluded | AP: original → excluded |
|---|---:|---|---|---|
| Tier 2 kNN | 94.5% | 3.48% → 0.20% | 0.71 → 0.98 | 0.80 → 0.96 |

**T6. Cases (§8.x; seed 20261104, τ = 5.0943)**

Durations are in µs. The "Top feature" column gives the feature with the largest share of s².

| Case | Row ID | Label | Duration | Fwd / Bwd pkts | Bwd bytes | s(x) | Top feature (share of s²) |
|---|---:|---|---:|---:|---:|---:|---|
| FN | 2454412 | PortScan | 455 | 2 / 1 | 2 | 2.292 | PSH Flag Count (19%) |
| FN | 2286836 | Bot | 1,001,534 | 3 / 3 | 18 | 1.686 | Flow IAT Min (95%) |
| FN | 2687946 | DDoS | 10,093,717 | 4 / 0 | 0 | 5.0942 | Active Mean/Min/Max (≈68–88%) |
| TP | 2723650 | DDoS | 5,021,793 | 4 / 0 | 0 | 6.462 | Active Mean/Min/Max (≈68–88%) |
| FP | 2132365 | BENIGN | 45,670,096 | 6 / 0 | 0 | 12.580 | Idle Std (63%) |
| FP | 2285373 | BENIGN | 13,413,432 | 3 / 2 | 0 | 5.0957 | Bwd IAT Min (66%) |

### References

2. Engelen, Rimmer, and Joosen. Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study. IEEE SPW 2021.
