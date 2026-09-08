# Reproducibility record

This page records the execution environment and locked evaluation settings used for the reported experiments. It complements the complete executed notebook and the technical report.

## Recorded environment

| Item | Recorded setting |
|---|---|
| Platform | Kaggle notebook |
| Python | 3.12.13 |
| TensorFlow | 2.20.0 |
| Keras | 3.13.2 |
| Accelerator | Two NVIDIA Tesla T4 GPUs |
| Random seed | 42 |
| Determinism | TensorFlow deterministic operations enabled |

The exact versions above are the versions recorded in the final academic run. Flexible version ranges for supporting libraries are kept in `requirements.txt`/`pyproject.toml` because their exact patch versions were not part of the submitted reproducibility record.

## Locked task settings

| Setting | Value |
|---|---:|
| Classification input / batch | 180 x 180 / 32 |
| Segmentation input / batch | 128 x 128 / 16 |
| Detection input / batch | 256 x 256 / 16 |
| Classification threshold | 0.50 |
| Segmentation threshold | 0.50 |
| Detection AP IoU | 0.50 |
| Detection NMS IoU | 0.50 |
| Detection AP score floor | 0.001 |
| Detection visual score threshold | 0.25 (visualisation only) |

## Data partition policy

For each task, architecture selection, checkpoint selection, thresholds, and post-processing choices were made from the development data only. The official VOC validation partition was retained as the final holdout and was evaluated after model selection.

| Task | Training | Internal validation | Final holdout |
|---|---:|---:|---:|
| Multi-label classification | 4,574 | 1,143 | 5,823 |
| Binary segmentation | 1,171 | 293 | 1,449 |
| Object detection | 4,574 | 1,143 | 5,823 |

## Dataset layout

Place the extracted dataset at `VOCdevkit/VOC2012/` or set `VOC_ROOT` to the VOC2012 directory.

```text
VOCdevkit/VOC2012/
├── Annotations/
├── ImageSets/
├── JPEGImages/
└── SegmentationClass/
```

The raw dataset is intentionally not committed to the repository.

## Environment setup

For the full experiment environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For package development and tests:

```bash
pip install -e ".[test]"
pytest -q
```

## Reviewing versus rerunning

`notebooks/full_experiment.ipynb` is the auditable record of the complete executed run, including training histories, checkpoint-selection logic, final metric tables, and qualitative predictions. Reviewing the preserved outputs does not require retraining.

Full retraining is GPU-intensive and can take substantially longer on CPU-only machines. The compact `notebooks/project_walkthrough.ipynb` and the reusable modules under `src/voc_multitask/` provide lighter entry points for inspection and reuse.
