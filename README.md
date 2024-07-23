# Cross-sensor super-resolution of irregularly sampled Sentinel-2 time series

This repository contains the implementation of our workshop paper on ["Cross-sensor super-resolution of irregularly sampled Sentinel-2 time series"](https://arxiv.org/abs/2404.16409). If you use our work, please cite the following:

```
@inproceedings{okabayashi_crosssensor_2024,
  title = {Cross-Sensor Super-Resolution of Irregularly Sampled {{Sentinel-2}} Time Series},
  booktitle = {{{EARTHVISION}} 2024 {{IEEE}}/{{CVF CVPR Workshop}}. {{Large Scale Computer Vision}} for {{Remote Sensing Imagery}}},
  author = {Okabayashi, Aimi and Audebert, Nicolas and Donike, Simon and Pelletier, Charlotte},
  date = {2024-06},
  location = {Seattle, United States},
  url = {https://hal.science/hal-04552850},
  urldate = {2024-05-07},
}
```

## Dataset

We use the [BreizhSR dataset](https://zenodo.org/records/11551220) available on Zenodo. BreizhSR is a dataset targetting super-resolution of (RGB bands of) Sentinel-2 images by providing time series colocated in space and time with SPOT-6/7 acquisitions. This dataset is composed of cloud free Sentinel-2 time series (visible bands at 10m resolution) and SPOT-6/7 pansharpened color images resampled 2.5m resolution. The study area is the region of Brittany (Breizh in the local language), located on the northwestern coast of France with an oceanic climate. The dataset covers about 35 000 km² with mostly agricultural areas (about 80 %). All acquisitions are from 2018 in the Brittany region of France.
See its webpage for additional details regarding the dataset and its collection process.

### Preprocessing

We provide two ways to load the BreizhSR dataset using PyTorch: using `rasterio` or using `torch`. The latter is **heavily** recommended, although it requires an offline preprocessing step to extract all image pairs as `torch.Tensor`. This is automated using the `data_gen/preprocess_data.py` script. You can use it by running the command:

```bash
python data_gen/preprocess_data.py path/to/where/you/stored/BreizhSR/
```

The resulting dataset will be stored by default in the `preprocessed/` subfolder. You can then load this dataset using `utils/dataloader.py`.

If this seems inconvenient for you, or you just want to access the raw data, you can use the `BreizhSRDataset` class from the `data_gen/dataset.py` module, which reads the original image rasters using `rasterio`.

## Train

### Single-image super-resolution (SISR)

```
# pretrain backbone model (RRDB or other SISR model)
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/rrdb/rrdb.yaml --exp_name sisr/rrdb_ckpt --reset

# train SRDiff conditioned by the backbone model
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/diffsr_sat.yaml --exp_name sisr/srdiff_rrdb_ckpt --hparams="rrdb_ckpt=checkpoints/sisr/rrdb_ckpt"
```

### Multi-image super-resolution (MISR)

```
# pretrain backbone model (HighRes-net L-TAE or other MISR model)
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/misr/misr_pretrain.yaml --exp_name misr/highresnet_ltae_ckpt --reset

# train SRDiff conditioned by the backbone model
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/diffsr_hr_ltae.yaml --exp_name misr/srdiff_highresnet_ltae_ckpt --hparams="rrdb_ckpt=checkpoints/misr/highresnet_ltae_ckpt"
```

## Evaluate

```
# backbone model
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config checkpoints/sisr/rrdb_ckpt/config.yaml --exp_name sisr/rrdb_ckpt --infer

# SRDiff
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config checkpoints/sisr/srdiff_rrdb_ckpt/config.yaml --exp_name sisr/srdiff_rrdb_ckpt --infer
```

## Credits

The implementation is based on the following works:
* [SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models](https://arxiv.org/abs/2104.14951) and its [GitHub repository](https://github.com/LeiaLi/SRDiff)
* [HighRes-net: Recursive Fusion for Multi-Frame Super-Resolution of Satellite Imagery](https://arxiv.org/abs/2002.06460)

