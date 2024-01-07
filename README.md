# DreamDistribution

This repo contains the code implementation of the paper:

*DreamDistribution: Prompt Distribution Learning for Text-to-Image Diffusion Models*

[[project page](https://briannlongzhao.github.io/DreamDistribution)] [[arXiv](https://arxiv.org/abs/2312.14216)]

![](docs/static/images/teaser_1_crop.gif)

> Based on a reference image set, DreamDistribution finds a prompt distribution, which can then be used to generate new 2D/3D instances, capable of text-guided editing and more.

## Usage

### Preparation
```shell
pip install -r requirements.txt
```

### Training

Place your reference images in a directory, for example `sample_images/cathedral/`, then run the following:

```shell
accelerate launch train.py \
  --train_data_dir=sample_images/cathedral/ \
  --output_dir=output
```
A more comprehensive list of command arguments is shown in `train.sh`

### Generate

Assume your checkpoint is saved at `output/final-1000.pt`.

```shell
python generate.py \
  --weights_path=output/final-1000.pt \
  --output_dir=output_images \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  --n_images=50 \
  --bsz=4
```

Generate with text-edit:
```shell
python generate.py \
  --weights_path=output/final-1000.pt \
  --output_dir=output_images \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  --n_images=50 \
  --bsz=4 \
  --customize_prefix="a photo of" \
  --customize_suffix="at night"
```

Generate with scaled standard deviation:
```shell
python generate.py \
  --weights_path=output/final-1000.pt \
  --output_dir=output_images \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  --n_images=50 \
  --bsz=4 \
  --std_scale=2.0
```

Generate composition of multiple prompt distributions:
```shell
python generate.py \
  --weights_path output1/final-1000.pt output2/final-1000.pt \
  --output_dir=output_images \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1" \
  --n_images=50 \
  --bsz=4 \
  --std_scale 1.0 1.0 \
  --distribution_weight 0.5 0.5
```

## Acknowledgement

We borrow code from [Hugging Face diffusers](https://arxiv.org/abs/2312.14216) and [Learning to Prompt for Vision-Language Models](https://arxiv.org/abs/2109.01134) ([CoOp](https://github.com/KaiyangZhou/CoOp)). We thank the authors and the open source contributors for their work and contribution.
