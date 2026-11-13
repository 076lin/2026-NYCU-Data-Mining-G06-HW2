- **Tier 1 Baseline**
  - Method: Single-feature threshold.
  - Feature: `Bwd Packet Length Max`.
  - Feature selection: the highest mean KS distance across the five training attack types in the training-set EDA.

- **Validation Results**
    | Metric | Value |
  | --- | ---: |
  | Total Flows | 764,639 |
  | Benign Flows | 752,114 |
  | Attack Flows | 12,525 |
  | Attack Prevalence | 1.64% |
  | Precision | 5.45% |
  | Recall | 90.85% |
  | F1 | 0.1029 |
  | False Positive Rate | 26.23% |
  | ROC-AUC | 0.7867 |
  | TP | 11,379 |
  | FP | 197,263 |
  | TN | 554,851 |
  | FN | 1,146 |

  - Despite detecting 90.85% of attacks, the method generated approximately 17.34 false alarms per detected attack. The low attack prevalence and substantial benign false-positive rate jointly resulted in low precision and F1.

- **Test Results**
   | Metric | Value |
  | --- | ---: |
  | Total Friday Flows | 703,245 |
  | Benign Flows | 414,322 |
  | Attack Flows | 288,923 |
  | Attack Prevalence | 41.08% |
  | Precision | 60.06% |
  | Recall | 44.28% |
  | F1 | 0.5098 |
  | False Positive Rate | 20.54% |
  | ROC-AUC | 0.7622 |
  | TP | 127,937 |
  | FP | 85,096 |
  | TN | 329,226 |
  | FN | 160,986 |

  - Test precision and F1 were higher, while recall was lower. Attack prevalence increased from 1.64% to 41.08%, the benign false-positive rate decreased from 26.23% to 20.54%, and the attack types also changed.

- **Attack-Specific Test Performance**

  | Attack Type | Total Flows | Detected | Missed | Recall | ROC-AUC against Test Benign Flows |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | DDoS | 128,027 | 126,797 | 1,230 | 99.04% | 0.8431 |
  | PortScan | 158,930 | 1,138 | 157,792 | 0.72% | 0.7008 |
  | Bot | 1,966 | 2 | 1,964 | 0.10% | 0.4647 |


  - PortScan represented 55.01% of test attack flows and accounted for 98.02% of all test false negatives. It was the main contributor to the low overall test recall.
  - Bot detection was also poor, but Bot represented only 0.68% of test attack flows, so its contribution to the aggregate recall was smaller.

- **Shared False-Positive Pattern**
  - Of 85,096 false positives, 78,926 had `Bwd Packet Length Max = 0`, accounting for 92.75% of false positives.
  - A feature value of 0 produces a score of 2.411664, above the threshold of 1.717218, so these benign-labeled flows were flagged.
  - This shows a limitation of the score: being far from the benign median does not necessarily indicate an attack. A single-center deviation rule can flag legitimate traffic patterns that occur away from that center.
  - These observations do not establish why the underlying traffic had zero recorded backward packet length; that would require inspecting additional flow information or packets.

- **Shared False-Negative Pattern**
  - Of 158,930 PortScan flows, 157,687 had `Bwd Packet Length Max = 6`, accounting for 99.22% of PortScan flows.
  - The test set also contained 47,316 benign flows with the same feature value of 6.
  - Both groups receive exactly the same score of 1.427973 and are classified as normal at the fixed threshold.
  - This is direct evidence of overlapping inputs: a deterministic rule using only this feature cannot assign different labels to flows with identical feature values. Changing the threshold alone cannot separate these particular benign and attack flows.


- **Failure Cases (τ = 1.717218)**

  | # | Type | True Label | Bwd Packet Length Max (bytes) | s(x) | s(x) − τ | Likely Cause |
  | ---: | --- | --- | ---: | ---: | ---: | --- |
  | 1 | FP | BENIGN | 0 | 2.411664 | +0.694446 | Benign value far from reference center |
  | 2 | FN | PortScan | 6 | 1.427973 | −0.289245 | Overlapping benign and attack values |
  | 3 | FN | Bot | 128 | 0.045056 | −1.672163 | Attack value near benign reference center |
  | 4 | FP | BENIGN | 3,525 | 1.717361 | +0.000143 | Benign score just above threshold |

  **Decision rule:** Predict attack when s(x) > τ; otherwise, predict normal.

  - **Case 1:** The benign flow's zero value is far from the reference center, producing a score 0.694446 above the threshold. The same zero-value pattern accounts for 92.75% of test false positives.
  - **Case 2:** The PortScan flow has the same feature value of 6 bytes as 47,316 benign test flows, so this single-feature rule cannot distinguish them. Its score is 0.289245 below the threshold, resulting in a false negative.
  - **Case 3:** The Bot flow's value of 128 bytes is close to the benign reference center of approximately 117 bytes. Its score is 1.672163 below the threshold, making this a clear miss rather than a near-boundary error.
  - **Case 4:** This benign flow's score is only 0.000143 above the threshold. It is a near-boundary false positive that illustrates sensitivity to threshold placement.

- **Conclusion**
  - Class imbalance, particularly the validation set's low attack prevalence of 1.64%, combined with a substantial false-positive rate to produce low precision and F1.
  - The method showed stronger detection of DoS-related traffic, potentially because similar patterns were present in training, although this explanation remains a hypothesis.
  - Of all test false positives, 92.75% had `Bwd Packet Length Max = 0`, which produced a score above the threshold and caused these benign flows to be flagged as attacks.
  - Of all PortScan flows, 99.22% had `Bwd Packet Length Max = 6`, giving them the same below-threshold score and causing all flows in this group to be classified as normal, resulting in false negatives.
