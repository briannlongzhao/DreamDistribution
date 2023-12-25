import os
import argparse
import torch


# Input single ctx vector
def interpret_ctx(ctx, tokenizer, embedder, topk=1, print_info=False):
    ranks, ctx_words, dists = [], [], []
    token_embedding = embedder.token_embedding.weight

    # Single context
    distance = torch.cdist(ctx, token_embedding)
    sorted_idxs = torch.argsort(distance, dim=1)
    sorted_idxs = sorted_idxs[:, :topk]

    if print_info:
        print(f"Size of token embedding: {token_embedding.shape}")
        print(f"Size of context: {ctx.shape}")
        print(f"Return the top-{topk} matched words")
        print(f"Size of distance matrix: {distance.shape}")

    for m, idxs in enumerate(sorted_idxs):
        words = [tokenizer.decoder[idx.item()] for idx in idxs]
        dist = [f"{distance[m, idx].item():.4f}" for idx in idxs]
        ranks.append(m+1)
        ctx_words.append(words)
        dists.append(dist)
    return ranks, ctx_words, dists



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fpath", type=str, help="Path to the learned prompt")
    parser.add_argument("topk", type=int, help="Select top-k similar words")
    args = parser.parse_args()

    fpath = args.fpath
    topk = args.topk

    assert os.path.exists(fpath)


