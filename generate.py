import argparse
import os
import version
import numpy as np
from numpy.random import normal, seed
import torch
from collections import OrderedDict
from tqdm import tqdm
from diffusers import StableDiffusionPipeline

from prompt_learner import PromptLearner

torch.backends.cuda.matmul.allow_tf32 = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--weights_path", type=str, default=None, nargs='+')
    parser.add_argument("--output_dir", default="output_images")
    parser.add_argument("--resolution", default=768, type=int, help="saved image resolution, <=768")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="stabilityai/stable-diffusion-2-1",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )

    # Prompt learner specific args
    parser.add_argument(
        "--cls_pos",
        type=str,
        default="end",
        choices=["end", "middle", "front"],
        help="Position of class name token in prompt"
    )
    parser.add_argument(
        "--n_images",
        type=int,
        default=50,
        help="Number of images to generate for each class"
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=None,
        help="Number of prompts used for generate for each class"
    )
    parser.add_argument(
        "--customize_prefix",
        type=str,
        default=None,
        help="Prefix added to learned context in prompt learner"
    )
    parser.add_argument(
        "--customize_suffix",
        type=str,
        default=None,
        help="Suffix added to learned context in prompt learner"
    )
    parser.add_argument(
        "--sample_ctx",
        action="store_true",
        help="Sample from distribution at text embedding space pre text encoder, experimental use only"
    )
    parser.add_argument(
        "--distribution_weight",
        type=float,
        default=[1],
        nargs='+',
        help="Weights used to combine multiple distributions"
    )
    parser.add_argument(
        "--std_scale",
        type=float,
        default=[1],
        nargs='+',
        help="Scaling std of distribution"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile pytorch model"
    )
    parser.add_argument(
        "--reparam_seed",
        type=int,
        default=None,
        help="Seed for same instance reparameterization generation"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    assert len(args.weights_path) == len(args.std_scale) == len(args.distribution_weight)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        safety_checker=None,
        torch_dtype=torch.float16
    )
    pipe.enable_xformers_memory_efficient_attention()
    pipe = pipe.to("cuda")

    # Load prompt learner ctx
    all_means, all_stds = [], []
    for weights_path, distribution_weight, std_scale in zip(args.weights_path, args.distribution_weight, args.std_scale):
        n_prompts, n_ctx = 1, 0
        ckpt, means, stds, prompt_texts = None, None, None, None
        if weights_path.endswith(".pt"):
            ckpt = torch.load(weights_path)
            ckpt = ckpt.get("model_state_dict", ckpt)
            for k, v in ckpt.items():
                if k.replace("_orig_mod.", "").replace("module.", "") == "ctx":
                    ckpt = OrderedDict({"ctx": v})
            ctx = ckpt["ctx"]
            n_ctx = ctx.shape[-2]
            assert ctx.ndim == 4
            n_prompts = ctx.shape[1]
        elif weights_path.endswith(".npz"):
            ckpt = np.load(args.weights_path)
            means = ckpt["means"]
            stds = ckpt["stds"]
            prompt_texts = ckpt["prompt_texts"]

        # Initialize prompt learner
        prompt_learner = PromptLearner(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            classnames=None,
            n_ctx=n_ctx,
            cls_pos=args.cls_pos,
            n_prompts=n_prompts,
            use_classname=False,
            customize_prefix=args.customize_prefix,
            customize_suffix=args.customize_suffix,
        )
        prompt_learner = prompt_learner.to("cuda")

        if weights_path.endswith(".pt"):
            missing_keys, unexpected_keys = prompt_learner.load_state_dict(ckpt, strict=False)
            assert "ctx" not in missing_keys
            assert len(unexpected_keys) == 0
            if args.sample_ctx:
                prompt_learner.fit_ctx(prefix=args.customize_prefix, suffix=args.customize_suffix)
                all_means.append(prompt_learner.ctx_means)
                all_stds.append(prompt_learner.ctx_stds)
            else:
                prompt_learner.fit(prefix=args.customize_prefix, suffix=args.customize_suffix)
                all_means.append(prompt_learner.means)
                all_stds.append(prompt_learner.stds)
        elif weights_path.endswith(".npz"):
            if args.customize_prefix or args.customize_suffix or args.sample_ctx:
                raise ValueError("Must provide ctx pre text encoder (.pt) for concatenation with prefix or suffix")
            all_means.append(means)
            all_stds.append(stds)
        print(f"prompt_lerner loaded weights from {weights_path}")
        prompt_learner.eval()

    # Sanity check
    assert all_stds[0].shape == all_means[0].shape
    assert np.sum(args.distribution_weight) == 1
    assert len(all_means) == len(all_stds) == len(args.distribution_weight) == len(args.std_scale)
    for i in range(len(all_means)):
        assert all_means[i].shape == all_means[0].shape
        assert all_stds[i].shape == all_stds[0].shape

    # Combine distributions
    all_means = [mean * weight for mean, weight in zip(all_means, args.distribution_weight)]
    all_means = np.sum(all_means, axis=0)
    all_vars = [np.square(std * scale * weight) for std, scale, weight in zip(all_stds, args.std_scale, args.distribution_weight)]
    all_stds = np.sqrt(np.sum(all_vars, axis=0))
    prompt_learner.means = all_means
    prompt_learner.stds = all_stds

    if args.compile and version.parse(torch.__version__) >= version.parse("2.0"):
        pipe.unet = torch.compile(pipe.unet)
        pipe.vae = torch.compile(pipe.vae)
        prompt_learner = torch.compile(prompt_learner)

    reparam_epsilon = None
    if args.reparam_seed is not None:
        seed(args.reparam_seed)
        reparam_epsilon = normal(loc=0, scale=1, size=prompt_learner.means[0].shape)

    """
    Total number of images generated = args.n_images
    Number of prompts used = args.n_prompts
    Each prompt used to generate num_images_per_prompt images
    Each forward generates args.bsz images
    Each prompt is used num_forward_per_prompt times
    """
    if args.n_prompts is None:  # Sample for every forward
        args.n_prompts = args.n_images // args.bsz
    num_images_per_prompt = args.n_images // args.n_prompts
    num_forward_per_prompt = max(1, num_images_per_prompt // args.bsz)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Generating {args.n_images} images using {args.n_prompts} prompts in {args.output_dir}")
    cur_i = len(list(os.listdir(args.output_dir))) + 1
    for _ in tqdm(range(args.n_prompts)):
        if args.sample_ctx:
            prompt_embs, caption = prompt_learner.sample_ctx(0, interpret=True)
        else:
            prompt_embs, caption = prompt_learner.sample(0, interpret=True, reparam_epsilon=reparam_epsilon)
        caption = ''.join([word[0] for word in caption])
        caption = caption.replace("</w>", ' ')
        caption = caption.replace('"', "")  # server don't like ", will map to %2522
        caption = caption.replace('/', "")  # otherwise will split directory
        caption = caption.replace('.', "")
        caption = caption[:100].strip()
        for _ in range(num_forward_per_prompt):
            print(f"generating {args.bsz} images of {caption}")
            with torch.no_grad():
                if args.reparam_seed is not None:
                    torch.manual_seed(0)
                x = pipe(prompt_embeds=prompt_embs, num_images_per_prompt=args.bsz)
            images = x.images
            for img in images:
                img.resize((args.resolution, args.resolution)).save(
                    os.path.join(args.output_dir, f"{caption}_{cur_i}.png")
                )
                cur_i += 1