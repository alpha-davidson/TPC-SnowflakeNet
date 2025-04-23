# Fréchet Point Cloud Distance

## Getting Started

First, run `predict.py` while specifying what model to use (via command line arguments) and what simulated dataset to predict on (via config file) to get your predicted and ground truth point clouds. For example:
```
python3 predict.py --config ./configs/JustMgEMD.yaml --model ./exp/checkpoints/JustMg512In2048OutEMD/ckpt-best.pth --gt_save_path ../data/frechet/gts/JustMgO2048.npy --pred_save_path ../data/frechet/preds/JustMg512In2048OutEMD.npy
```

A different virtual environment is needed to evaluate predicted point clouds using Fréchet Point Cloud Distance rather than the one used to train a SnowflakeNet model. To create the new virtual environment see the `alpha-davidson/downstream-benchmarks/README.md`.

To train the classification model required to compute FPCD, again got to the `alpha-davidson/downstream-benchmarks` repo. Go into the `data_pipeline/Mg22_pipeline_class` folder and open `Mg22_pipeline_class.py`. Set `sample_size` to the number of points in your SnowflakeNet model output and run the file.

Then go back to this repository and edit `frechet.py` to the correct file paths.
Finally, simply run:
```
python3 frechet.py
```