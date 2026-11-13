# 7 Results

## 7.x Why Tier 3 Wins and Loses

1. **Better ranking than Tier 1, lower F1 than Tier 1: a difference in operating point.** Tier 3 ranks flows better than Tier 1 (AP 0.782 vs. 0.577; AUROC 0.858 vs. 0.762), yet its F1@τ (0.431 vs. 0.510) and its recall@τ (0.297 vs. 0.443) are lower. The cause is the threshold each method picks on val. Tier 1's τ flags 20.5% of benign test flows, against 5.6% for Tier 3. At a test prevalence of 41%, F1 rewards flagging more flows: flagging every flow already gives F1 = 0.582. When the two tiers are compared at the same FPR (oracle thresholds, post hoc), the order reverses. At Tier 1's FPR, Tier 3 reaches recall 0.712 against 0.443; at Tier 3's FPR, Tier 1 reaches 0. The remaining gap at τ comes from DDoS, which makes up 44% of test attacks. Tier 1 catches 99.0% of DDoS flows, because their Bwd Packet Length Max (median 5,755 B) is far from benign (82 B). Tier 3 catches only 66.1%.

2. **Losing to Tier 2 on unseen attacks: a closed world.** Supervision covers only the five training attack types, so Tier 3 catches an unseen attack only if it resembles one of them (1, Sommer and Paxson). Its AP falls from 1.000 on a holdout of seen types to 0.119 on val and 0.782 on test. For each of the 9 unseen types, we computed the share of flows whose nearest training neighbour is an attack. This share predicts Tier 3's recall@τ (Spearman ρ = 0.996). Web attacks reach shares of 0.87–0.93 and are the only types on which Tier 3 beats Tier 2 (AUROC 0.97 vs. 0.68–0.69). PortScan, Bot and Heartbleed reach shares of at most 0.002, and there Tier 3 trails Tier 2 (AUROC 0.790 vs. 0.899, 0.716 vs. 0.761, 0.736 vs. 0.998).

3. **Label errors tilt the comparison towards Tier 3.** Attack flows whose direction was swapped are labelled BENIGN (2, Engelen et al.). In every split, at least 99% of the BENIGN flows sent from the victim's port 80 to the attacker fall inside an attack window. Tier 3 learns 6,583 such training flows as benign, so in test it flags only 3.5% of the 32,314 such flows, compared with 5.7% of other benign flows. Excluding these flows moves Tier 3's AP only from 0.782 to 0.799. Tier 2's AP, by contrast, rises from 0.80 to 0.96, because these flows are most of its false positives.

# 8 Analysis

## 8.x Tier 3 Error Analysis

1. **Shortcut feature.** Every training attack type has an empty response: the median Bwd Packet Length Min is 0 B, against 48 B for benign flows. Tier 3 therefore learns the shortcut "no response payload ⇒ attack" (3, Geirhos et al.). Bwd Packet Length Min is its top-gain feature. Its mean SHAP value (4, Lundberg and Lee) is −2.74 for PortScan and −1.48 for Bot, both of which send 6-B replies. This feature alone contributes −2.76 to PortScan flow #2466594, which scores s = 6.2e-4 < τ = 5.87e-3 (seed 20261101) and is missed (FN). Conversely, benign flow #2355969 receives no response and scores s = 0.9998 (FP).

2. **Single-flow view.** The evidence for PortScan and Bot is spread over many flows, so each single flow looks like a tiny benign flow. 99.1% of PortScan flows carry at most one packet per direction, and so do 20.6% of benign flows. Bot C&C flow #2191085 scores s = 2.8e-5, a complete inversion (FN).

3. **False alarms on attacked services.** Benign traffic to services that were attacked during training resembles those attacks. The FPR on port 22, where SSH-Patator appeared in training, is 21.7%, against 5.6% overall.

## Supporting Tables

These tables contain only the numbers quoted in the text. Test numbers use the original labels unless marked post hoc. Tier 2 kNN numbers come from the re-run (seeds 20261101–03).

**S1. Overall test results (§7.x-1)**

Test prevalence is 41.1% (288,923 attacks among 703,245 flows).

| | AP | AUROC | Recall@τ | FPR@τ | F1@τ |
|---|---:|---:|---:|---:|---:|
| Tier 1 | 0.577 | 0.762 | 0.443 | 20.5% | 0.510 |
| Tier 3 XGBoost | 0.782 | 0.858 | 0.297 | 5.6% | 0.431 |
| Flag every flow as an attack | | | | | 0.582 |

**S2. Test recall at matched FPR (§7.x-1; oracle thresholds, post hoc)**

| | At Tier 1's FPR (20.5%) | At Tier 3's FPR (5.6%) |
|---|---:|---:|
| Tier 1 | 0.443 | 0.000 |
| Tier 3 XGBoost | 0.712 | |

**S3. DDoS (§7.x-1)**

| Share of test attacks | Recall@τ, Tier 1 | Recall@τ, Tier 3 | Median Bwd Pkt Len Max, DDoS | Median Bwd Pkt Len Max, benign (test) |
|---:|---:|---:|---:|---:|
| 44.3% | 0.990 | 0.661 | 5,755 B | 82 B |

**S4. Tier 3 AP on seen and unseen attacks (§7.x-2)**

| Holdout (seen types) | Val (unseen) | Test (unseen) |
|---:|---:|---:|
| 1.000 | 0.119 | 0.782 |

**S5. Similarity to training attacks and AUROC (§7.x-2)**

NN-attack is the share of a type's flows whose nearest training neighbour is an attack. Over all 9 unseen types, its Spearman ρ with Tier 3's recall@τ is 0.996.

| Attack type | NN-attack | AUROC, Tier 2 | AUROC, Tier 3 |
|---|---:|---:|---:|
| Web Brute Force (val) | 0.871 | 0.686 | 0.966 |
| Web XSS (val) | 0.933 | 0.682 | 0.967 |
| PortScan (test) | 0.002 | 0.899 | 0.790 |
| Bot (test) | 0.000 | 0.761 | 0.716 |
| Heartbleed (val) | 0.000 | 0.998 | 0.736 |

**S6. BENIGN flows from the victim's port 80 to the attacker (§7.x-3)**

| Split | Flows | Inside an attack window |
|---|---:|---:|
| Train | 6,583 | 6,583 (100%) |
| Val | 2,155 | 2,143 (99.4%) |
| Test | 31,354 | 31,354 (100%) |

**S7. Effect of the 32,314 suspect test flows (§7.x-3; post hoc)**

| | FPR@τ on suspect flows | FPR@τ on other benign | Share of FPs that are suspect | AP: original → suspect excluded |
|---|---:|---:|---:|---|
| Tier 2 kNN | | | 94.2% | 0.80 → 0.96 |
| Tier 3 XGBoost | 3.5% | 5.7% | | 0.782 → 0.799 |

**S8. Median Bwd Packet Length Min and tiny-flow share (§8.x-1, §8.x-2)**

A tiny flow carries at most one packet in each direction.

| Class | Median Bwd Pkt Len Min | Tiny flows |
|---|---:|---:|
| Each of the 5 training attack types | 0 B | |
| Benign (train) | 48 B | |
| PortScan (test) | 6 B | 99.1% |
| Bot (test) | 6 B | |
| Benign (test) | | 20.6% |

**S9. Tier 3 mean SHAP of Bwd Packet Length Min (§8.x-1; seed 20261101)**

| PortScan | Bot |
|---:|---:|
| −2.74 | −1.48 |

**S10. Tier 3 FPR by destination port (§8.x-3)**

| Destination port | FPR@τ |
|---|---:|
| 22 (SSH) | 21.7% |
| All | 5.6% |

### References

1. Sommer and Paxson. Outside the Closed World: On Using Machine Learning for Network Intrusion Detection. IEEE S&P 2010.
2. Engelen, Rimmer, and Joosen. Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study. IEEE SPW 2021.
3. Geirhos et al. Shortcut Learning in Deep Neural Networks. Nature Machine Intelligence 2020.
4. Lundberg and Lee. A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.
