# Technical report

## Scope

This project evaluates three computer-vision tasks on PASCAL VOC2012 under a common leakage-aware protocol: multi-label image classification, binary foreground/background semantic segmentation, and 20-class object detection.

The official training partitions are split into training and internal-validation subsets for model development. The official VOC validation partitions are retained as one-time final holdouts. Architecture choice, checkpoint selection, thresholds, and detection post-processing are fixed before final holdout inference.

## Data and reproducibility

| Task | Training | Internal validation | Final holdout |
|---|---:|---:|---:|
| Multi-label classification | 4,574 | 1,143 | 5,823 |
| Binary segmentation | 1,171 | 293 | 1,449 |
| Object detection | 4,574 | 1,143 | 5,823 |

The original run used random seed 42, deterministic TensorFlow operations, Python 3.12.13, TensorFlow 2.20.0, Keras 3.13.2, and two NVIDIA Tesla T4 GPUs on Kaggle.

### Input handling

- **Classification:** 180×180 RGB. Scratch models rescale to `[0,1]`; Xception models map raw `[0,255]` pixels to `[-1,1]` once inside the model.
- **Segmentation:** 128×128 RGB/masks. VOC labels 1–20 are merged to foreground, class 0 remains background, and void label 255 receives zero loss/metric weight.
- **Detection:** 256×256 RGB. Targets use an 8×8 grid and two object slots per cell.

## 1. Multi-label classification

### Scratch-CNN iteration

Three original architectures were tested.

| Model | Parameters | Best epoch | Val BCE ↓ | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|
| High-capacity baseline | 16,258,900 | 3 | 0.22095 | 0.77880 | 0.26261 |
| Regularised efficient CNN | 681,108 | 32 | **0.18454** | **0.85991** | **0.40043** |
| Residual separable CNN | 643,348 | 22 | 0.20043 | 0.82978 | 0.33187 |

The baseline was intentionally large and overfit rapidly. The selected scratch model replaced the large flattened head with global average pooling and used augmentation, L2 regularisation, batch normalisation, spatial/dense dropout, and depthwise-separable convolutions. It used 95.8% fewer parameters than the baseline while improving validation BCE and PR-AUC.

Residual shortcuts were a tested negative result: they did not improve the regularised architecture in this setting.

### Xception transfer learning

Three required transfer-learning strategies were compared: standalone classifiers on frozen pooled Xception features, a frozen image-level Xception classifier, and restricted last-block fine-tuning.

| Approach | Trainable parameters | Val BCE ↓ | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|
| Standalone frozen features | 542,548 | 0.10275 | 0.94731 | 0.79445 |
| Extended frozen Xception | 542,548 | **0.10021** | **0.95068** | 0.80487 |
| Last-block fine-tuned Xception | 5,284,180 | 0.10021 | 0.95047 | **0.80492** |

The predeclared selection criterion was BCE, so the simpler frozen image-level classifier was retained. Fine-tuning substantially increased trainable capacity without improving the selection metric.

### Locked holdout result

| Metric | Value |
|---|---:|
| BCE | 0.09539 |
| ROC-AUC | 0.95471 |
| PR-AUC | 0.81444 |
| Micro-F1 @ 0.50 | 0.75494 |
| Macro-F1 @ 0.50 | 0.74254 |
| Exact-match accuracy | 0.54783 |

The fixed 0.50 threshold gives higher micro precision (0.86634) than recall (0.66893), so the final classifier is conservative in positive predictions.

## 2. Binary semantic segmentation

A three-level encoder-decoder without skip connections was compared with a parameter-matched U-Net.

| Model | Parameters | Val IoU | Precision | Recall |
|---|---:|---:|---:|---:|
| Encoder-decoder baseline | 1,947,105 | 0.46599 | 0.57314 | 0.71368 |
| U-Net with skip connections | 1,929,825 | **0.51003** | **0.62670** | **0.73260** |

The U-Net improved validation IoU by 0.04404 with slightly fewer parameters, supporting the interpretation that same-scale skip connections recover spatial detail lost during downsampling.

### Locked holdout result

| Metric | Value |
|---|---:|
| Foreground IoU | 0.51007 |
| Dice/F1 | 0.67556 |
| Foreground precision | 0.62766 |
| Foreground recall | 0.73138 |
| Pixel accuracy | 0.81227 |
| Median per-image IoU | 0.52053 |

Per-image IoU remains heterogeneous. The model often captures the main object extent but smooths thin structures and boundaries, partly because masks are resized to 128×128 and all 20 semantic classes are merged into one foreground channel.

## 3. Object detection

The detector combines an ImageNet-pretrained Xception backbone with a two-layer separable-convolution head. It predicts two box slots per 8×8 cell. The custom loss combines objectness, box, and class terms; negative objectness is downweighted and class loss uses clipped inverse-frequency weights. VOC difficult objects are handled explicitly, and inference uses class-aware NMS.

Target encoding represents 99.405% of development easy boxes. Grid/slot collisions remain a limitation.

The detector is selected directly by internal-validation mAP@0.50 rather than the composite training loss. Frozen-backbone warm-up peaks at **0.21395 mAP@0.50**; restricted block-14 fine-tuning does not improve it.

### Locked holdout result

| Metric | Value |
|---|---:|
| mAP@0.50 | 0.19000 |
| Slot objectness precision @ 0.50 | 0.16975 |
| Slot objectness recall @ 0.50 | 0.59776 |
| Positive-slot class accuracy | 0.72942 |

Detection is the weakest task. High objectness recall with low precision indicates many false-positive proposals; small objects, crowded scenes, and the coarse grid are the principal observed failure modes.

## Validity safeguards

- official holdouts were excluded from architecture comparison, epoch selection, threshold tuning, and NMS tuning;
- development/holdout identifier overlap was checked to be zero;
- split prevalence was audited before modelling;
- preprocessing ranges and output shapes were asserted;
- best checkpoints were restored before evaluation;
- frozen/fine-tuned Xception boundaries were audited;
- void segmentation pixels and difficult detection objects were excluded according to VOC semantics;
- headline metrics come from preserved executed notebook outputs.

## Main limitations

1. one deterministic seed/split does not quantify between-run variability;
2. universal 0.50 classification/segmentation thresholds are simple but not class-specific optima;
3. reduced input resolution limits small-object and fine-boundary fidelity;
4. binary segmentation removes class identity and merges touching foreground objects;
5. the 8×8 two-slot detector still loses some boxes to representation collisions;
6. detector objectness/post-processing remains poorly calibrated;
7. model search was bounded by the available dual-T4 compute budget.

## Next steps

The most promising extensions are class-specific classification thresholds selected on internal validation, higher-resolution/boundary-aware segmentation, and a multi-scale or anchor-free detection head with geometric box augmentation and focal-style objectness treatment.
