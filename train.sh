#!/bin/bash

TRAIN_DATA_DIR="sample_images/cathedral/"  # Input image directory
CUSTOMIZE_PREFIX=""  # A prefix string added to the learnable prompts during training, e.g., "a painting of"
CUSTOMIZE_SUFFIX=""  # A suffix string added to the learnable prompts during training
OUTPUT_DIR="output/"  # Directory to save checkpoints and final weights of prompts
PRETRAINED_MODEL_NAME_OR_PATH="stabilityai/stable-diffusion-2-1"  # Huggingface backbone diffusion model
MAX_TRAIN_STEPS=2000  # Total optimization steps
GRADIENT_ACCUMULATION_STEPS=1
CHECKPOINTING_STEPS=200  # Save checkpoint every ${CHECKPOINTING_STEPS} steps
REPARAM_SAMPLES=4  # Number of samples used to approximate expectation of loss function
N_CTX=4  # Length of learnable prompts
N_PROMPTS=32  # Number of prompts to model the prompt distribution
ORTHO_LOSS_WEIGHT=0.001  # Weight of orthogonal loss, 0 to disable

LR_WARMUP_STEPS=0
LR_SCHEDULER="constant"
LEARNING_RATE=0.001
BATCH_SIZE=4
REPORT_TO="wandb"


accelerate launch train.py \
  --train_data_dir=$TRAIN_DATA_DIR \
  --pretrained_model_name_or_path=$PRETRAINED_MODEL_NAME_OR_PATH \
  --output_dir=$OUTPUT_DIR \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --train_batch_size=$BATCH_SIZE \
  --n_ctx=$N_CTX \
  --n_prompts=$N_PROMPTS \
  --checkpointing_steps=$CHECKPOINTING_STEPS \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --lr_scheduler=$LR_SCHEDULER \
  --lr_warmup_steps=$LR_WARMUP_STEPS \
  --learning_rate=$LEARNING_RATE \
  --reparam \
  --reparam_samples=$REPARAM_SAMPLES \
  --ortho_loss_weight=$ORTHO_LOSS_WEIGHT \
  --customize_prefix="$CUSTOMIZE_PREFIX" \
  --customize_suffix="$CUSTOMIZE_SUFFIX" \
  --report_to=$REPORT_TO
