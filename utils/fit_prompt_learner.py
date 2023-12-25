# Fit distribution of learned prompt collections

import torch
import numpy as np
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../SDCoOp"))

from prompt_learner import PromptLearner
from utils.imagenet_classes import wnid2classname_simple as classnames
classids = list(classnames.keys())
classnames = list(classnames.values())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights_path", type=str, help=".pt ckpt path")
    parser.add_argument("--output_path", type=str, help=".npz save path")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--cls_pos",
        type=str,
        default="end",
        choices=["end", "middle", "front"],
        help="Position of class name token in prompt"
    )
    parser.add_argument(
        "--use_classname",
        default=True,
        action="store_true",
        help="Use class names in prompt learner"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    assert os.path.exists(args.weights_path)
    if args.output_path is None:
        args.output_path = args.weights_path.replace(".pt", ".npz")
    state_dict = torch.load(args.weights_path)
    state_dict = state_dict.get("model_state_dict", state_dict)
    ctx_tensor = state_dict["ctx"]
    assert ctx_tensor.ndim == 4
    _, n_prompts, n_ctx, _ = ctx_tensor.shape

    prompt_learner = PromptLearner(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        classnames=classnames,
        n_ctx=n_ctx,
        cls_pos=args.cls_pos,
        n_prompts=n_prompts,
        use_classname=args.use_classname,
    )
    prompt_learner.load_state_dict(state_dict)
    prompt_learner.fit()
    np.savez(args.output_path, means=prompt_learner.means, stds=prompt_learner.stds, prompt_texts=prompt_learner.prompt_texts)
    print(f"saved to {args.output_path}")
