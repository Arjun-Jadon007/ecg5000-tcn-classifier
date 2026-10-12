# ECG-TCN: Arrhythmia Detection with a Temporal Convolutional Network

A PyTorch reimplementation of the ECG-TCN architecture for 5-class heartbeat classification on the ECG5000 dataset. The project goes past reproduction. It diagnoses a severe class-imbalance problem, mitigates it, tests generalization on real recordings from a different database (MIT-BIH), and ships an interactive Gradio demo.

> **Disclaimer:** This is a research and learning project. It is not a medical device and must not be used for clinical decisions.

## Highlights

- Reproduced the published ECG-TCN architecture. The parameter count is **14,859** against the paper's **14,883**.
- Diagnosed the class-imbalance problem with confusion matrices and balanced metrics, then mitigated it with class-weighted cross-entropy. SMOTE was also explored as a complementary approach.
- Evaluated with imbalance-aware metrics (balanced accuracy, macro F1) alongside plain accuracy, so the weak rare-class performance is visible.
- Tested domain shift on raw MIT-BIH signals using R-peak detection, segmentation, and resampling to the model's 140-sample input format.
- Wrapped the model in a Gradio demo that returns a predicted class and a softmax confidence score.

## Dataset

**ECG5000** is derived from the BIDMC Congestive Heart Failure Database (PhysioNet). Individual heartbeats were extracted from the raw recording (92,584 beats), 5,000 were randomly sampled, and each was resampled by interpolation to exactly **140 samples**.

- **Split:** 500 train / 4,500 test. The small training set is deliberate. It mimics scarce labeled hospital data and matches the original paper.
- **Classes and training-set counts:**

| Class | Label | Train examples |
|---|---|---|
| 1 | Normal | 292 |
| 2 | R-on-T Premature Ventricular Contraction | 177 |
| 3 | Premature Ventricular Contraction | 10 |
| 4 | Supraventricular Premature / Ectopic Beat | 19 |
| 5 | Unclassified beat | 2 |

The rarest class has only **2** training examples. That imbalance drives most of the results and limitations below.

Download ECG5000 from the [UCR Time Series Archive](https://www.timeseriesclassification.com/) and place `ECG5000_TRAIN.txt` and `ECG5000_TEST.txt` in `./data`. The first column of each file is the class label and the remaining 140 columns are the signal.

## Architecture

```
Input (1 x 140)
  -> Conv 1x1 (F1 = 2)
  -> TCN Block 1 (dilation 1)
  -> TCN Block 2 (dilation 2)
  -> TCN Block 3 (dilation 4)
  -> Flatten (11 x 140 = 1540)
  -> Fully Connected -> 5 logits
```

Each TCN block contains:

```
Dilated causal Conv1d (kernel 11) -> BatchNorm -> ReLU -> Dropout(0.3)
Dilated causal Conv1d (kernel 11) -> BatchNorm -> ReLU -> Dropout(0.3)
+ residual connection -> ReLU
```

- **Receptive field:** `1 + 2(K-1)(2^L - 1) = 141` for K = 11 and L = 3. This covers the whole 140-sample heartbeat.
- **Causal padding:** each output depends only on past and present samples, which suits streaming use.
- **Three deliberate modifications to the original TCN design:** BatchNorm instead of weight norm, BatchNorm on the residual path too, and standard dropout instead of spatial dropout.

## Training setup

| Setting | Value |
|---|---|
| Loss | `CrossEntropyLoss` with inverse-frequency class weights |
| Optimizer | Adam, lr = 0.001 |
| Batch size | 30 |
| Epochs | 20 |

Class weights are computed from the training label counts:

```python
class_counts = np.bincount(train_labels, minlength=5)
class_weights = 1.0 / np.maximum(class_counts, 1)
class_weights = class_weights / class_weights.sum() * 5
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32))
```

## Results

Evaluated on the 4,500-beat test set.

| Metric | Value |
|---|---|
| Accuracy | 90.56% |
| Balanced accuracy | 61.48% |
| Weighted F1 | 0.9149 |
| Macro F1 | 0.5605 |

**How to read these numbers.** Accuracy and weighted F1 are dominated by the Normal class. Balanced accuracy and macro F1 weight every class equally, so they show the real weakness: the model does well on common classes and poorly on the rare ones. That gap is why this repo reports both families of metrics.

## Domain-shift test (MIT-BIH)

To check generalization beyond ECG5000, raw MIT-BIH recordings go through a separate preprocessing pipeline:

1. Load the record with `wfdb`.
2. Detect R-peaks with `neurokit2`.
3. Cut a window around each R-peak to get individual heartbeats.
4. Resample each beat to 140 samples.
5. Feed the beats to the trained model and report the predicted class with its softmax confidence.

These recordings have no automated ground-truth check in this pipeline, so the confidence score is **not** a correctness guarantee.

## Demo

Run the Gradio app (`app.py`) to upload a single-beat CSV and get a predicted class and confidence. Sample beats from the ECG5000 test set are in `sample_data/`.

## Installation and usage

```bash
git clone https://github.com/YOUR-USERNAME/ecg-tcn-arrhythmia-detection.git
cd ecg-tcn-arrhythmia-detection
pip install -r requirements.txt
```

Train:

```bash
python train.py --data_dir ./data --epochs 20
```

Predict on a single beat:

```bash
python predict.py --input sample_data/sample_class0.csv
```

## Project structure

```
.
├── model.py          # TCN architecture
├── data.py           # ECG5000 loading and DataLoaders
├── train.py          # training loop, class-weighted loss, evaluation
├── predict.py        # single-beat inference with softmax confidence
├── app.py            # Gradio demo
├── requirements.txt
└── sample_data/      # example single-beat CSVs
```

## Limitations

- **Rare classes are the main weakness.** With 2 to 19 training examples per rare class, class weighting cannot add information that isn't in the data. SMOTE only interpolates between the few real examples that exist.
- **Domain shift.** Performance on MIT-BIH recordings depends heavily on R-peak detection quality and on how closely the preprocessing matches ECG5000's format.
- **Confidence is not correctness.** A softmax score is not calibrated and the model has no out-of-distribution detection.
- **Not deployed on embedded hardware.** The original paper targets the GAP8 ultra-low-power chip. This project validates the model in PyTorch only.

## Future work

- Calibration (for example temperature scaling) and out-of-distribution detection
- Larger and more balanced training data
- Quantization and deployment on edge hardware

## Reference

This project reimplements the architecture from the ECG-TCN paper (Ingolfsson et al.). Add the full citation here after checking it against the paper.

## Acknowledgements

Built as part of the *Application of AI and ML in Healthcare* training at MIIC, MNIT Jaipur.

## Author

Arjun Jadon, Diploma in Computer Science and Engineering, Thapar Polytechnic College, Patiala
[LinkedIn](https://www.linkedin.com/in/YOUR-PROFILE) · [GitHub](https://github.com/YOUR-USERNAME)
