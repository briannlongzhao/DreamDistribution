import torch
import torch.nn as nn
import numpy as np
from numpy.random import normal
from einops import repeat, rearrange
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModel, AutoProcessor
from torch.nn.functional import cosine_similarity, normalize

from utils.interpret_prompt import interpret_ctx


class PromptLearner(nn.Module):
    """
    PromptLearner class implements learnable prompt embeddings
    Input class idx, output learnable prompt embedding of that class

    The learnable context has shape (n_cls, n_prompts, n_ctx, dim)
    """

    def __init__(
        self,
        pretrained_model_name_or_path,
        classnames,
        n_ctx=4,
        n_prompts=32,
        cls_pos="end",
        dtype=torch.float32,
        use_classname=True,
        customize_prefix=None,
        customize_suffix=None,
        reparam_samples=4
    ):
        super().__init__()
        self.dtype = dtype
        self.n_prompts = n_prompts
        self.classnames = classnames
        self.use_classname = use_classname
        self.customize_prefix = customize_prefix
        self.customize_suffix = customize_suffix
        self.reparam_samples = reparam_samples
        if customize_suffix is not None or customize_prefix is not None \
        or self.classnames is None or self.use_classname is False:  # Disable classname for customize generation
            self.use_classname = False
            self.classnames = None
        self.n_cls = len(self.classnames) if self.classnames is not None else 1

        self.tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
        self.vision_encoder = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        self.vision_processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
        text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
        self.text_encoder = CustomTextEncoder(text_encoder.text_model)
        self.vision_encoder.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        ctx_dim = self.text_encoder.final_layer_norm.weight.shape[0]

        # random initialization
        print("Initializing class-specific contexts with prompt distribution learning")
        ctx_vectors = torch.empty(self.n_cls, n_prompts, n_ctx, ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        prompt_placeholder = " ".join(["X"] * n_ctx)

        print(f"Number of context words (tokens): {n_ctx}")
        print(f"Number of prompts per class: {n_prompts}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        if self.use_classname:
            self.classnames = [name.replace("_", " ") for name in self.classnames]
            self.name_lens = [len(self.tokenizer(name).input_ids) - 2 for name in self.classnames]
            prompts = [prompt_placeholder + " " + name + "." for name in self.classnames]
        else:
            self.customize_prefix = "" if self.customize_prefix is None else self.customize_prefix
            self.customize_suffix = "" if self.customize_suffix is None else self.customize_suffix
            prompts = [(self.customize_prefix + " " + prompt_placeholder + " " + self.customize_suffix).strip()]

        self.embedder = self.text_encoder.embeddings

        # tokenized_prompts as an anchor for retrieving position of eos token in each class prompt
        tokenized_prompts = torch.cat([
            self.tokenizer(
                p,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids
            for p in prompts
        ]).to(self.embedder.position_ids.device)
        with torch.no_grad():
            embedding = self.embedder(tokenized_prompts).type(self.dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS

        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.n_tokens = self.tokenizer.model_max_length
        self.class_token_position = cls_pos
        self.means = None
        self.stds = None
        self.prompt_texts = None
        self.prompt_freq = np.ones((self.n_cls, n_prompts))


    """
    Interpret the learned context vector by finding the closest word in embedding space
    If prompt distribution learning, interpret the mean
    """
    def interpret(self, cls_idx=0):
        ctx = self.ctx[cls_idx].mean(dim=0)
        eow = "</w>"
        _, words, _ = interpret_ctx(ctx, tokenizer=self.tokenizer, embedder=self.embedder, topk=1)
        if self.use_classname:
            if self.class_token_position == "end":
                words = words + [[self.classnames[cls_idx]]]
            elif self.class_token_position == "front":
                words = [[self.classnames[cls_idx]]] + words
            elif self.class_token_position == "middle":
                words = words[:len(words) / 2] + [[self.class_token_position[cls_idx]]] + words[len(words) / 2:]
        else:
            if self.customize_prefix:
                words = [[self.customize_prefix+eow]] + words
            if self.customize_suffix:
                words = words + [[eow+self.customize_suffix+eow]]
        words = ''.join([word[0] for word in words]).replace(eow, ' ')
        words = words.strip()
        return words


    """
    Concat the input ctx with the tokens of class names represented by cls_idx
    Input ctx with shape (B,n_ctx,d) or (B,p,n_ctx,d), cls_idx is list of length B 
    """
    def concat(self, ctx, cls_idx):
        prefix = self.token_prefix[cls_idx]
        suffix = self.token_suffix[cls_idx]

        if ctx.ndim == 4:  # (B,p,n_ctx,d)
            p = ctx.shape[1]
            prefix = repeat(prefix, "b l d -> b p l d", p=p)
            suffix = repeat(suffix, "b l d -> b p l d", p=p)

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (*, 1, dim)
                    ctx,  # (*, n_ctx, dim)
                    suffix,  # (*, *, dim)
                ],
                dim=-2,
            )
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(cls_idx.shape[0]):
                name_len = self.name_lens[cls_idx[i]]
                prefix_i = prefix[i:i+1, :, :]  # keep dim
                class_i = suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                ctx_i_half1 = ctx[i:i+1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i:i+1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (*, 1, dim)
                        ctx_i_half1,  # (*, n_ctx//2, dim)
                        class_i,  # (*, name_len, dim)
                        ctx_i_half2,  # (*, n_ctx//2, dim)
                        suffix_i,  # (*, *, dim)
                    ],
                    dim=-2,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(cls_idx.shape[0]):
                name_len = self.name_lens[cls_idx[i]]
                prefix_i = prefix[i:i+1, :, :]
                class_i = suffix[i:i+1, :name_len, :]
                suffix_i = suffix[i:i+1, name_len:, :]
                ctx_i = ctx[i:i+1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (*, 1, dim)
                        class_i,  # (*, name_len, dim)
                        ctx_i,  # (*, n_ctx, dim)
                        suffix_i,  # (*, *, dim)
                    ],
                    dim=-2,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError

        return prompts


    """
    Concat the given context vectors with input prefix and suffix pre text encoder
    Input ctx: (B,n_prompts,n_ctx,d=1024) or (B,n_ctx,d=1024)
    Output: (B,n_prompts,l=77,d=1024) or (B,l=77,d=1024)
    """
    def concat_custom(self, ctx, prefix="", suffix=""):
        if prefix is None or prefix == "":
            prefix_tokens = torch.Tensor(
                [[self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]]
            ).to(self.embedder.position_ids.device).to(torch.int64)
        else:
            prefix_tokens = self.tokenizer(
                prefix,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.embedder.position_ids.device)
        if suffix is None or suffix == "":
            suffix_tokens = torch.Tensor(
                [[self.tokenizer.bos_token_id, self.tokenizer.eos_token_id]+[self.tokenizer.pad_token_id]*(self.n_tokens-2)]
            ).to(self.embedder.position_ids.device).to(torch.int64)
        else:
            suffix_tokens = self.tokenizer(
                suffix,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(self.embedder.position_ids.device)
        prefix_eos_position = (prefix_tokens == self.tokenizer.eos_token_id).nonzero()[-1, -1].item()
        suffix_eos_position = (suffix_tokens == self.tokenizer.eos_token_id).nonzero()[-1, -1].item()
        assert prefix_eos_position + self.n_ctx + suffix_eos_position <= self.n_tokens, "prefix+ctx+suffix too long"
        prefix_embeddings = self.embedder(prefix_tokens).type(self.dtype)
        suffix_embeddings = self.embedder(suffix_tokens).type(self.dtype)
        prefix_embeddings = prefix_embeddings[..., :prefix_eos_position, :]
        suffix_embeddings = suffix_embeddings[..., 1:(self.n_tokens-self.n_ctx-prefix_eos_position+1), :]
        assert ctx.ndim == 4, "ctx should have shape (B,p,l,d)"
        assert ctx.shape[1] == self.n_prompts, f"ctx shape {ctx.shape}[1] should match self.n_prompts={self.n_prompts}"
        prefix_embeddings = repeat(prefix_embeddings, "1 l d -> b p l d", b=ctx.shape[0], p=ctx.shape[1])
        suffix_embeddings = repeat(suffix_embeddings, "1 l d -> b p l d", b=ctx.shape[0], p=ctx.shape[1])
        prompts = torch.cat([prefix_embeddings, ctx, suffix_embeddings], dim=-2)
        return prompts
    
    
    """
    Input pooled_prompt (B,n_prompts,d=1024)
    Output orthogonal loss (scalar)
    """
    def orthogonal_loss(self, pooled_prompts):
        pooled_prompts = normalize(pooled_prompts, dim=-1)
        cos_sim = pooled_prompts @ pooled_prompts.transpose(1, 2)  # (B,n_prompts,n_prompts)
        diag_mask = torch.eye(cos_sim.shape[1], device=cos_sim.device).bool()
        cos_sim[:, diag_mask] = 0
        loss_per_batch = (cos_sim ** 2).sum(dim=(1, 2)) / (cos_sim.shape[1] * (cos_sim.shape[1] - 1))
        loss = loss_per_batch.mean()
        return loss
        
    
    """
    Input class indices (B)
    Output tokenized class prompts (B,l=77,d=1024)
    """
    def forward(self, cls_idx, imgs=None, interpret=False):
        ctx = self.ctx[cls_idx]
        if self.use_classname:
            prompts = self.concat(ctx, cls_idx)
        else:
            prompts = self.concat_custom(ctx, self.customize_prefix, self.customize_suffix)
        prompts_hidden_state, pooled_prompts = self.text_encoder(
            prompts,
            pooled=True,
            tokenized_prompts=self.tokenized_prompts.to(cls_idx.device)[cls_idx]
        )
        assert pooled_prompts.ndim == 3, f"pooled_prompts should have shape (B,n_prompts,d), now is {pooled_prompts.shape}"
        ortho_loss = self.orthogonal_loss(pooled_prompts)
        prompts_hidden_state = self.reparameterize(prompts_hidden_state, n_samples=self.reparam_samples)  # (B,n,l,d)
        if interpret:
            prompts = prompts.mean(dim=1)
            _, caption, _ = interpret_ctx(torch.squeeze(prompts), tokenizer=self.tokenizer, embedder=self.embedder, topk=1)
            return prompts_hidden_state, caption, ortho_loss
        return prompts_hidden_state, ortho_loss

    """
    For each class, fit the collection of prompts as a normal distribution 
    (after concat with classnames/prefix/suffix and post text encoder) 
    Store in self.means and self.stds
    Currently not taking covariance matrix
    """
    def fit(self, prefix=None, suffix=None):
        self.prompt_texts = []
        self.means = np.empty((self.n_cls, self.n_tokens, self.ctx_dim))  # (n_cls*77*1024)
        self.stds = np.empty(self.means.shape)  # (n_cls,77,1024)
        for cls in tqdm(range(self.n_cls), desc="fit prompt learner distribution"):
            cls_ctx = self.ctx[cls].unsqueeze(0)  # (1,p,l,d) or (1,l,d)
            if self.use_classname:
                cls_prompts = self.concat(cls_ctx, [cls])
            else:
                cls_prompts = self.concat_custom(cls_ctx, prefix=prefix, suffix=suffix)
            cls_prompts = self.text_encoder(cls_prompts)[0]
            self.means[cls] = cls_prompts.mean(dim=0).detach().cpu().numpy()
            self.stds[cls] = cls_prompts.std(dim=0).detach().cpu().numpy()
            if self.n_prompts == 1:
                self.stds[cls] = np.zeros(self.stds[cls].shape)
            prompt_text = self.interpret(cls)
            self.prompt_texts.append(prompt_text)


    """
    After prompts are fit, sample from stored means and stds as input to diffusion
    If not prompt distribution learning, return self.forward 
    """
    def sample(self, cls_idx, interpret=False, reparam_epsilon=None):
        assert self.means is not None and self.stds is not None
        if reparam_epsilon is not None:
            prompt = torch.from_numpy(self.means[cls_idx] + reparam_epsilon * self.stds[cls_idx]).unsqueeze(dim=0)
        else:
            prompt = torch.from_numpy(normal(self.means[cls_idx], self.stds[cls_idx])).unsqueeze(dim=0)
        if interpret:
            caption = self.prompt_texts[cls_idx]
            return prompt, caption
        return prompt


    def reparameterize(self, prompts, n_samples=1):
        std = torch.std(prompts, dim=1)  # (B,l,d)
        mu = torch.mean(prompts, dim=1)  # (B,l,d)
        eps = torch.randn_like(repeat(std, "b l d -> b n l d", n=n_samples))  # (B,n,l,d)
        prompt = mu.unsqueeze(1) + eps * std.unsqueeze(1)
        return prompt  # (B,n,l,d)


    """
    For each class, fit the collection of prompts as a normal distribution (pre text encoder) 
    Store in self.ctx_means and self.ctx_stds
    Experimental use only
    """
    def fit_ctx(self, prefix=None, suffix=None):
        self.ctx_means = np.empty((self.n_cls, self.n_tokens, self.ctx_dim))  # (n_cls,l=77,d=1024)
        self.ctx_stds = np.empty(self.ctx_means.shape)  # (n_cls,l=77,d=1024)
        self.prompt_texts = []
        for cls in tqdm(range(self.n_cls), desc="fit prompt learner distribution"):
            cls_ctx = self.ctx[cls].unsqueeze(0)  # (1,p,l,d)
            if self.use_classname:
                cls_prompts = self.concat(cls_ctx, [cls])
            else:
                cls_prompts = self.concat_custom(cls_ctx, prefix=prefix, suffix=suffix)
            cls_prompts = cls_prompts[0]
            self.ctx_means[cls] = cls_prompts.mean(dim=0).detach().cpu().numpy()
            self.ctx_stds[cls] = cls_prompts.std(dim=0).detach().cpu().numpy()
            prompt_text = self.interpret(cls)
            self.prompt_texts.append(prompt_text)

    """
    After prompts are fit, sample from stored ctx_means and ctx_stds and feed to text encoder as input to diffusion
    If not prompt distribution learning, return self.forward
    Experimental use only 
    """
    def sample_ctx(self, cls_idx, interpret=False):
        assert self.ctx_means is not None and self.ctx_stds is not None
        prompt = torch.from_numpy(normal(self.ctx_means[cls_idx], self.ctx_stds[cls_idx])).unsqueeze(dim=0)
        prompt = self.text_encoder(prompt.to(dtype=self.dtype, device=self.embedder.position_ids.device))
        if interpret:
            caption = self.prompt_texts[cls_idx]
            return prompt, caption
        return prompt




"""
Original text_encoder: CLIPTextModel,
    input (B,77) prompt token indices,
    convert token into token embeddings
    output (B,77,1024) prompt embedding
CustomTextEncoder: Remove token embedding process in CLIPTextModel, token embeddings output from PromptLearner
    input (B,77,1024) prompt token embeddings
    output (B,77,1024) prompt embedding
"""

class CustomTextEncoder(nn.Module):
    def __init__(self, text_transformer, dtype=torch.float32):
        super().__init__()
        self.encoder = text_transformer.encoder
        self.positional_embedding = text_transformer.embeddings.position_embedding
        self.final_layer_norm = text_transformer.final_layer_norm
        self.dtype = dtype
        self.embeddings = text_transformer.embeddings

    def forward(self, prompts, pooled=False, tokenized_prompts=None):
        prompts = prompts + self.positional_embedding.weight.type(self.dtype)
        n_prompts = 1
        prompts_dim = prompts.ndim
        if prompts.ndim == 4:
            n_prompts = prompts.shape[1]
            prompts = rearrange(prompts, "b p l d -> (b p) l d")
            if tokenized_prompts is not None:
                tokenized_prompts = repeat(tokenized_prompts, "b l -> (b p) l", p=n_prompts)
        # Prepare attention mask
        attention_mask = torch.empty(prompts.shape[0], prompts.shape[1], prompts.shape[1], dtype=prompts.dtype)
        attention_mask.fill_(torch.tensor(torch.finfo(prompts.dtype).min))
        attention_mask.triu_(1)  # zero out the lower diagonal
        attention_mask = attention_mask.unsqueeze(1).to(prompts.device)  # expand mask
        prompts = self.encoder(
            inputs_embeds=prompts,  # (B,77,1024) or ((B,P),77,1024) float16
            causal_attention_mask=attention_mask,  # (B,1,77,77) or ((B,P),1,77,1024) float16
        )  # (B,77,1024)
        prompts = self.final_layer_norm(prompts[0]).type(self.dtype)

        # prompts.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eos embedding (eos_token is the highest number in each sequence)
        pooled_prompts = None
        if pooled:
            assert tokenized_prompts is not None
            pooled_prompts = prompts[torch.arange(prompts.shape[0]), tokenized_prompts.argmax(dim=-1)]  # (B,1024)
            # pooled_prompts = pooled_prompts @ self.text_projection  # (B,1024)
        if prompts_dim == 4:
            prompts = rearrange(prompts, "(b p) l d -> b p l d", p=n_prompts)
            if pooled_prompts is not None:
                pooled_prompts = rearrange(pooled_prompts, "(b p) d -> b p d", p=n_prompts)

        # return prompts if not pooled else pooled_prompts
        if pooled:
            return prompts, pooled_prompts
        return prompts
