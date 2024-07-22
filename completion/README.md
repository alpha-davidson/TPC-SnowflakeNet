# Point Cloud Completion

This repository contains the source code for the papers:

1. Snowflake Point Deconvolution for Point Cloud Completion and Generation with Skip-Transformer (TPAMI 2023)

2. SnowflakeNet: Point Cloud Completion by Snowflake Point Deconvolution with Skip-Transformer (ICCV 2021, Oral)

[<img src="../pics/completion.png" width="100%" alt="Intro pic" />](../pics/completion.png)

The code in this repository has been adapted by [Ben Wagner](https://github.com/bewagner1) for ALPhA usage.

## Datasets

#### For ALPhA use the following datasets were used:
##### Training
- Simulated 22Mg + alpha
- Simulated 16O + alpha
- A combination of the above was primarily used, with events filtered to a minimum of 128 unique points and sampled to 512 points for a complete event
##### Inferencing
- Test set from the simulated 22Mg + alpha and simulated 16O + alpha combined dataset
- Experimental 14C

## Getting Started

To use our code, make sure that the environment and PyTorch extensions are installed according to the instructions in the [main page](https://raw.githubusercontent.com/AllenXiangX/SnowflakeNet).


## Training

To train a point cloud completion model from scratch, run:

```
python train.py --config <config> --exp_name <training_name> <other_args>
```

For example:

```
python train.py --config ./configs/22Mg_16O_combo.yaml --exp_name 22Mg_16O_center_cutting
```

## Evaluation

To evaluate a pre-trained model, first specify the model_path in configuration file, then run:

```
python test.py --config <config> --model <path/to/model/checkpoint>
```

For example:

```
python test.py --config ./configs/22Mg_16O_combo.yaml --model ./exp/checkpoints/22Mg_16O_center_cutting/ckpts-best.pth
```

## Inferences

To use a pre-trained model to inference on a the test set, run:

```
python inference.py --config <config> <other_args>
```

For example

```
python inference.py --config ./configs/22Mg_16O_combo.yaml --model ./exp/checkpoints/22Mg_16O_center_cutting/ckpts-best.pth --normed
```

## Acknowledgements


This repo is based on: 
- [GRNet](https://github.com/hzxie/GRNet), 
- [PoinTr](https://github.com/yuxumin/PoinTr),

We thank the authors for their great job!