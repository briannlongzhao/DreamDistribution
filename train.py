#!/usr/bin/env python
# coding=utf-8
# Original Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
#
# Changes made:  
# - Date: 11/01/2023
# - Author: Brian Nlong Zhao, Yuhang Xiao  
# - Description: Modified training pipeline for compatibility with DreamDistribution

import argparse
import time
import logging
import math
import os
import random
from pathlib import Path
import gc

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from collections import OrderedDict
from einops import repeat, rearrange
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, DistributedDataParallelKwargs
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from itertools import chain

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from prompt_learner import PromptLearner
from utils.imagenet_classes import wnid2classname_simple as class_dict
from utils.imagenet_classes import subset100 as subset

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.17.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {}



def log_validation(prompt_learner, vae, unet, args, accelerator, class_dict, weight_dtype, step):
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    
    if args.enable_xformers_memory_efficient_attention and str(pipeline.device) != "cpu":
        pipeline.enable_xformers_memory_efficient_attention()
        
    prompt_learner = accelerator.unwrap_model(prompt_learner)
    
    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)
    
    if args.use_classname:
        # Convert all validation class input into list of wnid
        class_idx = []
        if args.validation_class is None:
            bn = os.path.basename(args.train_data_dir)
            if bn[0] == 'n':  # Trained on one class subfolder
                wnids = [bn]
            else:  # Trained on multiple classes, smaple subset of classes
                wnids = random.choices(os.listdir(args.train_data_dir), k=args.num_validation_images)
        else:
            if args.validation_class[0][1:].isdigit():  # wnids
                wnids = args.validation_class
            else:  # class names
                wnids = []
                for item in args.validation_class:
                    try:
                        wnids.append(class_dict.values().index(item))
                    except Exception as e:
                        print(e)
                        continue
            if len(wnids) == 0:
                logger.info("No valid class provided, skip validation")
                return
        for wnid in wnids:
            try:
                class_idx.append(list(class_dict.keys()).index(wnid))
            except Exception as e:
                print(e)
                continue
        class_idx = class_idx * (args.num_validation_images // len(class_idx))
        logger.info(f"Running validation on {[list(class_dict.values())[idx] for idx in class_idx]}")

        images = []
        prompt_learner.eval()
        prompt_learner.fit()
        for i in class_idx:
            with torch.no_grad():
                prompt_embeds = prompt_learner.sample(torch.LongTensor([i]))
                with accelerator.autocast():
                    image = pipeline(
                        prompt_embeds=prompt_embeds,
                        num_inference_steps=50,
                        generator=generator
                    ).images[0]
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation", np_images, step, dataformats="NHWC")
            elif tracker.name == "wandb":
                tracker.log(
                    {
                        "validation": [
                            wandb.Image(
                                image,
                                caption=f"{class_idx[i]}: {accelerator.unwrap_model(prompt_learner).interpret(class_idx[i])}")
                            for i, image in enumerate(images)
                        ],
                    }
                )
            else:
                logger.warn(f"image logging not implemented for {tracker.name}")
    else:
        images = []
        prompt_learner.eval()
        with accelerator.autocast():
            prompt_learner.fit(prefix=args.customize_prefix, suffix=args.customize_suffix)
        prompt_embs, caption = prompt_learner.sample(0, interpret=True)
        caption = ''.join([word[0] for word in caption])
        caption = caption[:caption.find("<|endoftext|>")]
        caption = caption.replace("</w>", ' ')
        caption = caption.replace('"', "")  # server don't like ", will map to %2522
        caption = caption.replace('/', "")  # otherwise will split directory
        caption = caption.replace('.', "")
        caption = caption[:100].strip()
        for _ in range(args.num_validation_images):
            with torch.autocast("cuda"):
                image = pipeline(prompt_embeds=prompt_embs, num_inference_steps=50, generator=generator).images[0]
            images.append(image)

        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images("validation", np_images, step, dataformats="NHWC")
            elif tracker.name == "wandb":
                tracker.log(
                    {"validation": [wandb.Image(image, caption=f"{i}: {caption}") for i, image in enumerate(images)]}
                )
            else:
                logger.warn(f"image logging not implemented for {tracker.name}")
    del pipeline
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column",
        type=str,
        default="image",
        help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--n_cls",
        default=100,
        type=int,
        choices=[100, 1000],
        help="Train using 100 class subset"
    )
    parser.add_argument(
        "--validation_class",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of imagenet classes (wnid or class name) evaluated every `--validation_steps` and logged to `--report_to`."),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help=("Number of images to generate in a round of validation"),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=768,  # 768 for stabilityai/stable-diffusion-2, 512 for CompVis/stable-diffusion-v1-4
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        default=False,
        action="store_true",
        help=(
            "Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping."
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=1000,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=5.0,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
             "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile pytorch model"
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Whether or not to push the model to the Hub."
    )
    parser.add_argument(
        "--hub_token",
        type=str,
        default=None,
        help="The token to use to push to the Model Hub."
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' and `"wandb"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default=None,
        help="Load a checkpoint.pt weights file directly."
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        default=True,
        help="Whether or not to use xformers."
    )
    parser.add_argument(
        "--latent_noise_offset",
        type=float,
        default=0,
        help="The scale of latent noise offset."
    )
    parser.add_argument(  # Deprecated
        "--num_noise_token",
        type=int,
        default=0,
        help="The number of prompt noise token."
    )
    parser.add_argument(  # Deprecated
        "--prompt_noise_offset",
        type=float,
        default=0,
        help="The scale of prompt noise offset."
    )
    parser.add_argument(
        "--use_classname",
        action="store_true",
        help="use class names in prompt learner"
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help="Run validation every X steps, 0 to disable",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="DreamDistribution",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--n_ctx",
        type=int,
        default=4,
        help="Number of learnable words (tokens) in prompts"
    )
    parser.add_argument(
        "--cls_pos",
        type=str,
        default="end",
        choices=["end", "middle", "front"],
        help="Position of class name token in prompt"
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=32,
        help="Number of prompts per class in prompt distribution learning"
    )
    parser.add_argument(
        "--customize_prefix",
        type=str,
        default=None,
        help="Prefix added to learned context"
    )
    parser.add_argument(
        "--customize_suffix",
        type=str,
        default=None,
        help="Suffix added to learned context"
    )
    parser.add_argument(
        "--reparam_samples",
        type=int,
        default=3,
        help="Number of samples used in reparameterization"
    )
    parser.add_argument(
        "--ortho_loss_weight",
        type=float,
        default=0.001,
        help="Weight of orthogonality loss in prompt distribution learning"
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    # Sanity checks
    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError("Need either a dataset name or a training folder.")
    if not args.use_classname:
        args.n_cls = 1

    return args


def main():
    args = parse_args()

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    classnames = class_dict
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        # logging_dir=logging_dir,
        project_config=accelerator_project_config,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Load scheduler and diffusion model.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")

    # Initialize prompts learner
    prompt_learner = PromptLearner(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        classnames=list(classnames.values()),
        n_ctx=args.n_ctx,
        n_prompts=args.n_prompts,
        cls_pos=args.cls_pos,
        dtype=weight_dtype,
        use_classname=args.use_classname,
        customize_prefix=args.customize_prefix,
        customize_suffix=args.customize_suffix,
        reparam_samples=args.reparam_samples,
    )

    # tokenized_prompts as an anchor to locate the eos position
    tokenized_prompts = prompt_learner.tokenized_prompts.to(accelerator.device)

    # Freeze vae and unet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    logger.info("Freeze VAE, custom text encoder, and UNet")

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod ** 0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                for i, model in enumerate(models):
                    if isinstance(model, PromptLearner):
                        torch.save(
                            {"model_state_dict": {"ctx": accelerator.unwrap_model(prompt_learner).state_dict()["ctx"]}},
                            os.path.join(output_dir, "prompt_learner.pt")
                        )
                    # make sure to pop weight so that corresponding model is not saved again
                    weights.pop()

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                if isinstance(model, PromptLearner):
                    state_dict = torch.load(os.path.join(input_dir, "prompt_learner.pt"))["model_state_dict"]
                    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                    assert "ctx" not in missing_keys
                    assert len(unexpected_keys) == 0


        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
                args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    # Only update prompt_learner
    optimizer = optimizer_cls(
        prompt_learner.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    else:
        data_files = {}
        if args.train_data_dir is not None:
            if not args.use_classname:
                data_files["train"] = os.path.join(args.train_data_dir, "**")
            elif args.n_cls == 100 and not os.path.basename(args.train_data_dir).startswith('n'):
                data_files["train"] = [os.path.join(args.train_data_dir, cls, "**") for cls in subset]
            else:
                data_files["train"] = os.path.join(args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    column_names = dataset["train"].column_names

    # 6. Get the column names for input/target.
    dataset_columns = DATASET_NAME_MAPPING.get(args.dataset_name, None)
    if args.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = args.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    if not args.use_classname:
        assert args.caption_column not in column_names, "Customize prompt should be passed as argument instead of metadata.jsonl"
    elif args.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = args.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        if args.caption_column not in examples.keys():
            examples["input_ids"] = torch.tensor(
                [0 for _ in examples["pixel_values"]],
                dtype=torch.long,
                requires_grad=False,
            ).detach()
        else:
            # Postpone tokenization to prompt_learner, output class name idx here
            # examples["input_ids"] = tokenize_captions(examples)
            examples["input_ids"] = torch.tensor(
                [list(classnames.keys()).index(c) for c in examples[args.caption_column]],
                dtype=torch.long,
                requires_grad=False,
            ).detach()
        return examples

    with accelerator.main_process_first():
        if args.max_train_samples is not None:
            dataset["train"] = dataset["train"].shuffle(seed=args.seed).select(range(args.max_train_samples))
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )
    assert len(train_dataloader) > 0, \
        "Empty dataloader, if number of training images is small, set drop_last=False in dataloader creation"


    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
  
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )


    # Move everything to gpu and cast to weight_dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    prompt_learner.to(accelerator.device, dtype=weight_dtype)

    # Prepare everything with our `accelerator`.
    unet, vae = accelerator.prepare(unet, vae)
    prompt_learner, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        prompt_learner, optimizer, train_dataloader, lr_scheduler
    )

    # Compile model
    if args.compile and version.parse(torch.__version__) >= version.parse("2.0"):
        vae = torch.compile(vae)
        unet = torch.compile(unet)
        prompt_learner = torch.compile(prompt_learner)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)


    # Potentially load checkpoint.pt file
    if args.load_checkpoint:
        assert os.path.isfile(args.load_checkpoint) and args.load_checkpoint.endswith(".pt"), \
        f"{args.load_checkoint} does not exist or is not a .pt file"
        ckpt = torch.load(args.load_checkpoint)
        ckpt = ckpt.get("model_state_dict", ckpt)
        for k, v in ckpt.items():
            if k.replace("_orig_mod.", "").replace("module.", "") == "ctx":
                ckpt = OrderedDict({"ctx": v})
        missing_keys, unexpected_keys = prompt_learner.load_state_dict(ckpt, strict=False)
        assert "ctx" not in missing_keys
        assert len(unexpected_keys) == 0
        print(f"Checkpoint loaded from {args.load_checkpoint}")

    # Initial validation
    if accelerator.is_main_process and args.validation_steps > 0:
        log_validation(
            prompt_learner,
            vae,
            unet,
            args,
            accelerator,
            classnames,
            weight_dtype,
            0,
        )
    accelerator.wait_for_everyone()

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    trainable_params = 0
    for name, param in chain(prompt_learner.named_parameters(), vae.named_parameters(), unet.named_parameters()):
        if param.requires_grad:
            trainable_params += param.numel()
            print(name, param.numel())
    logger.info(f"Total learnable parameters = {trainable_params}")


    # Potentially load in the weights and states from a previous save
    resume_step = 0
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(0, args.max_train_steps), disable=not accelerator.is_local_main_process, initial=global_step)
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        step_start = time.time()
        prompt_learner.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            data_time = time.time() - step_start

            # batch["input_ids"] (B,) class name indices
            with accelerator.accumulate(prompt_learner):
                # Convert images to latent space
                image_encoding_start = time.time()
                with accelerator.autocast() and torch.no_grad():
                    latents = vae.encode(batch["pixel_values"].to(weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                image_encoding_time = time.time() - image_encoding_start

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.latent_noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.latent_noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noise_start = time.time()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                noise_time = time.time() - noise_start

                # Get the prompt token embeddings and then text embedding from prompt learner for conditioning
                prompt_start = time.time()
                with accelerator.autocast():
                    prompt_hidden_states, ortho_loss = prompt_learner(batch["input_ids"], batch["pixel_values"])
                prompt_time = time.time() - prompt_start

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                forward_start = time.time()
                with accelerator.autocast():
                    assert prompt_hidden_states.ndim == 4 and prompt_hidden_states.shape[1] == args.reparam_samples, \
                        f"Unknown prompt_hidden_states shape {prompt_hidden_states.shape}"
                    prompt_hidden_states = rearrange(prompt_hidden_states, "b n l d -> (b n) l d")
                    noisy_latents = repeat(noisy_latents, "b c h w -> (b n) c h w", n=args.reparam_samples)
                    timesteps = repeat(timesteps, "t -> (t n)", n=args.reparam_samples)
                    model_pred = unet(noisy_latents, timesteps, prompt_hidden_states).sample
                    model_pred = rearrange(model_pred, "(b n) c h w -> b n c h w", b=args.train_batch_size, n=args.reparam_samples)
                forward_time = time.time() - forward_start

                if args.snr_gamma is None:
                    target = repeat(target, "b c h w -> b n c h w", n=args.reparam_samples)
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    mse_loss_weights = (
                            torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    mse_loss_weights = rearrange(mse_loss_weights, "(b n) -> b n", b=args.train_batch_size, n=args.reparam_samples)
                    target = repeat(target, "b c h w -> b n c h w", n=args.reparam_samples)
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(2, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()
                
                # Add orthogonal loss
                loss += args.ortho_loss_weight * ortho_loss
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                backward_start = time.time()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(prompt_learner.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                backward_time = time.time() - backward_start
                  
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                accelerator.log({"ortho_loss": ortho_loss}, step=global_step)
                accelerator.log({"lr": float(lr_scheduler.get_last_lr()[0])}, step=global_step)
                accelerator.log({"epoch": epoch}, step=global_step)
                train_loss = 0.0

            step_time = time.time() - step_start
            step_start = time.time()
            accelerator.log({
                "data_time": data_time,
                "step_time": step_time,
                "image_encoding_time": image_encoding_time,
                "noise_time": noise_time,
                "prompt_time": prompt_time,
                "forward_time": forward_time,
                "backward_time": backward_time,
            }, step=global_step)

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            
            del batch, latents, noise, bsz, timesteps, noisy_latents, prompt_hidden_states, model_pred, target, loss, avg_loss
            del ortho_loss
            gc.collect()
            if accelerator.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process and args.checkpointing_steps > 0 and global_step % args.checkpointing_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                print(f"Checkpointer saved in {save_path}")

            if accelerator.is_main_process and args.validation_steps > 0 and global_step % args.validation_steps == 0:
                log_validation(
                    prompt_learner,
                    vae,
                    unet,
                    args,
                    accelerator,
                    classnames,
                    weight_dtype,
                    global_step,
                )
            accelerator.wait_for_everyone()

    # Save the trained prompt_learner
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, f"final-{global_step}.pt")
        prompt_learner = accelerator.unwrap_model(prompt_learner)
        torch.save({
            "epoch": args.num_train_epochs,
            "model_state_dict": prompt_learner.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
        }, save_path)
        print(f"Final weights saved in {save_path}")
        prompt_learner.fit()
        save_path = save_path.replace(".pt", ".npz")
        np.savez(save_path, means=prompt_learner.means, stds=prompt_learner.stds, prompt_texts=prompt_learner.prompt_texts)
        print(f"Final fitted prompts saved in {save_path}")

        if args.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    main()
