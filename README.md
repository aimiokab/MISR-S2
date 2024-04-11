# Cross-sensor super-resolution of irregularly sampled Sentinel-2 time series

This repository contains the implementation of our workshop paper on "Cross-sensor super-resolution of irregularly sampled Sentinel-2 time series".

## Train

### Single-image super-resolution (SISR)

```
# pretrain backbone model (RRDB or other SISR model)
CUDA_VISIBLE_DEVICES=0 python tasks/trainer.py --config configs/rrdb/sat_pretrain.yaml --exp_name sisr/rrdb_ckpt --reset

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

## License

## Credits

The implementation is based on the following works:
* [SRDiff: Single Image Super-Resolution with Diffusion Probabilistic Models](https://arxiv.org/abs/2104.14951)
* [HighRes-net: Recursive Fusion for Multi-Frame Super-Resolution of Satellite Imagery](https://arxiv.org/abs/2002.06460)

