# Tree-Structured Spectral-Temporal Fusion for Spoofed Speech Detection

This repository provides the PyTorch implementation of **TreeFus-ST**, a tree-structured spectral-temporal fusion method for spoofed speech detection.

TreeFus-ST is built on a RawGAT-ST-style spectral-temporal graph attention backbone. The main contribution is a tree-structured fusion layer that models hierarchical dependencies between spectral and temporal graph representations.

## Main idea

TreeFus-ST performs spectral-temporal fusion through three core operations:

1. **Norm-based tree construction**
   Spectral-temporal joint node features are ranked according to their L2 norms, and a deterministic binary tree is constructed based on the ranking.

2. **Parent aggregation**
   Each non-root node receives information from its parent node, allowing high-level structural cues to be propagated along the tree.

3. **Child aggregation**
   Each node aggregates information from its child nodes when available, enabling bottom-up contextual refinement.

The enhanced node representation is obtained by combining self information, parent information, and child information, and is then passed to downstream graph attention, pooling, and classification layers.

## Repository structure

```text
TreeFus-ST/
├── core_scripts/
├── package-stage-1/
├── tDCF_python/
├── .gitignore
├── data_utils.py
├── LICENSE
├── main.py
├── model.py
├── model_config_TreeFus-ST.yaml
├── RawBoost.py
├── README.md
└── requirements.txt
```

Main files:

* `main.py`: training and evaluation entry point.
* `model.py`: model definition, including the proposed `TreeFusSTLayer`.
* `data_utils.py`: dataset loading and protocol parsing.
* `RawBoost.py`: optional RawBoost data augmentation during training.
* `model_config_TreeFus-ST.yaml`: model configuration file.
* `tDCF_python/`: EER and min t-DCF evaluation scripts for ASVspoof2019 LA.
* `package-stage-1/`: official-style evaluation utilities for ASVspoof2021, if used.
* `core_scripts/`: utility scripts used by the training pipeline.

The following files or folders are **not included** in this repository:

* ASVspoof datasets
* model checkpoints
* score files
* log files
* cache files
* temporary experiment outputs

## Installation

Create a Python environment:

```bash
conda create -n treefus-st python=3.8
conda activate treefus-st
```

Install PyTorch according to your CUDA version from the official PyTorch website. Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Dataset preparation

The code supports ASVspoof2019 LA and ASVspoof2021 LA/DF-style protocols.

For ASVspoof2019 LA, the expected directory structure is:

```text
/path/to/ASVspoof2019/LA/
├── ASVspoof2019_LA_train/flac/
├── ASVspoof2019_LA_dev/flac/
├── ASVspoof2019_LA_eval/flac/
└── ASVspoof2019_LA_cm_protocols/
    ├── ASVspoof2019.LA.cm.train.trn.txt
    ├── ASVspoof2019.LA.cm.dev.trl.txt
    └── ASVspoof2019.LA.cm.eval.trl.txt
```

The dataset should be downloaded separately from the official ASVspoof data source. This repository does not contain any speech data.

## Training on ASVspoof2019 LA

Train TreeFus-ST without RawBoost:

```bash
python main.py \
  --database_path /path/to/ASVspoof2019/LA/ \
  --protocols_path /path/to/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ \
  --year 2019 \
  --task LA \
  --num_epochs 100 \
  --batch_size 8 \
  --lr 0.0001 \
  --weight_decay 0.0001 \
  --loss WCE \
  --track logical \
  --features Raw_GAT \
  --comment TreeFus-ST
```

Train TreeFus-ST with optional RawBoost augmentation:

```bash
python main.py \
  --database_path /path/to/ASVspoof2019/LA/ \
  --protocols_path /path/to/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ \
  --year 2019 \
  --task LA \
  --num_epochs 100 \
  --batch_size 8 \
  --lr 0.0001 \
  --weight_decay 0.0001 \
  --loss WCE \
  --track logical \
  --features Raw_GAT \
  --enable_rawboost \
  --rawboost_algo 4 \
  --comment TreeFus-ST_rawboost
```

## Evaluation on ASVspoof2019 LA

Generate a score file:

```bash
python main.py \
  --database_path /path/to/ASVspoof2019/LA/ \
  --protocols_path /path/to/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/ \
  --year 2019 \
  --task LA \
  --eval --is_eval \
  --model_path /path/to/checkpoint.pth \
  --eval_output score_TreeFus-ST.txt
```

Compute EER and min t-DCF:

```bash
python tDCF_python/evaluate_tDCF_asvspoof19_eval_LA.py \
  Eval \
  score_TreeFus-ST.txt
```

## Evaluation on ASVspoof2021 LA

For ASVspoof2021 LA, the generated score file follows the official two-column format:

```text
utterance_id score
```

Example command for ASVspoof2021 LA:

```bash
python main.py \
  --database_path /path/to/ASVspoof2021/LA/ \
  --protocols_path /path/to/LA-keys-full/keys/LA/CM/ \
  --year 2021 \
  --task LA \
  --eval --is_eval \
  --model_path /path/to/checkpoint.pth \
  --eval_output cm_scores_LA.txt
```

The official ASVspoof2021 evaluation package should be used to compute the final metrics.

## Notes

* `RawBoost.py` is only used as optional data augmentation during training and is not part of the proposed TreeFus-ST fusion module.
* The main contribution of this repository is the tree-structured spectral-temporal fusion module implemented in `TreeFusSTLayer`.
* Score files, logs, cache files, and datasets are not included in this repository.
* Please modify `database_path`, `protocols_path`, and `model_path` according to your local environment.

## Citation

If you use this code, please cite the corresponding paper:

```bibtex
@inproceedings{treefusst,
  title={Tree-Structured Spectral-Temporal Fusion for Spoofed Speech Detection},
  author={Anonymous},
  booktitle={To appear},
  year={2026}
}
```

## Acknowledgements

This code is developed based on a RawGAT-ST-style spectral-temporal graph attention framework. We also thank the ASVspoof organizers for providing benchmark datasets and evaluation protocols.
