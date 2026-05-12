# smiles_project
# Hallucination Detection — Solution

## Reproduction

```bash
pip install -r requirements.txt
python solution.py
```

Outputs: `results.json` (evaluation metrics) and `predictions.csv` (test-set labels).

Tested on:  Python 3.10+, CUDA T4.

---

## Results

5-fold  stratified CV, fixed 15% held-out test set (104 samples):

| Checkpoint | Accuracy | F1 | AUROC |
|---|---|---|---|
| Majority-class baseline | 70.19% | 82.49% | N/A |
| Probe (train) | 71.67% | 83.00% | 80.58% |
| Probe (val) | 71.97% | 83.23% | 64.56% |
| **Probe (test)** | **69.81%** | **81.93%** | **60.82%** |

Primary metric — **Test AUROC: 60.82%**

Feature dim: 8081 · Samples: 689 

---

## Approach

### Feature Extraction (`aggregation.py`)

The default baselinee uses only the last real token of the final transformer layer (896 dims). This discards almost all inter-layer signal.

**Multi-layer mean pooling + geometric features:**

1. **Last 8 transformer layers** (Qwen2.5-0.5B has 24; layers 17–24 used).
2. **Mean-pool over all real (non-padding) tokens** per layer - 896-dim vector per layer.
3. **Concatenate** all 8 layer vectors - 7168 dims. Each layer captures a different abstraction level.
4. **Last real token of the final layer** appended → +896 dims (terminal hidden state).
5. **Geometric features** baked into `aggregate()`:
   - Inter-layer cosine similarities between consecutive layer means (7 values) — representation drift correlates with hallucination.
   - Layer-wise L2 norms (8 values) — activation scale shifts signal distributional anomalies.
   - Log sequence length (1 value).
   - Mean token-level variance in the last layer (1 value).

**Total feature dimension: 8081.**

### Probe Classifier (`probe.py`)

With ~468 training samples and 8081-dimensional features, an MLP massively overfits (train AUROC 100%, test AUROC 56% in initial experiments). A linear probe is the standard approach in the mechanistic interpretability literature for exactly this reason.

**Pipeline:** `StandardScaler -> PCA(64) -> LogisticRegressionCV`

- **PCA(64)**: Reduces 8081 -> 64 dims, removing noise and collinear directions.
- **LogisticRegressionCV**: Selects the best L2 regularisation strength C from `[0.001, 0.01, 0.1, 1.0, 10.0]` via internal 3-fold CV scored by AUROC — the same metric used for evaluation.
- **`class_weight='balanced'`**: Automatically corrects for the 70% / 30% hallucinated / truthful imbalance without manual tuning.
- **Threshold tuning**: `fit_hyperparameters` searches  over the validation fold to maximise F1.

### Splitting Strategy (`splitting.py`)

- **Fixed 15% held-out test set** (stratified, 104 samples) - never touched during training or validation.
- **5-fold stratified cross-validation** on the remaining 85% - metrics averaged across folds for robust estimation on this small dataset.

---

## Experiments tried but not  included in final solution

- **Last-token only (baseline)**: 896 dims. Test AUROC ~56% with MLP. Poor because a single token from the last layer  discards almost all inter-layer signal.

- **MLP probe** (`Linear(128->256) -> ReLU -> Dropout(0.4) -> Linear(256->128) -> Dropout(0.4) -> Linear(128->1)`): Train AUROC 100%, test AUROC 56%. Catastrophic overfitting — 66K parameters on 468 samples. Replacing with logistic regression immediately improved test AUROC to ~60.8% and closed the train/test gap from 44 to 20 points.

- **PCA(128)**: More components slightly hurt test AUROC compared to PCA(64). The additional components capture noise directions that hurt LR generalisation on 468 samples.

- **Balanced-accuracy threshold tuning**: Overcorrected  for the class imbalance - pushed accuracy well  below baseline by predicting too many samples as truthful. F1-based threshold tuning is more stable.

- **Full 24-layer concatenation** (21504 dims): Marginally worse - early embedding layers add noise. Using last 8 layers is better.
