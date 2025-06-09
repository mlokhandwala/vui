import math
import time
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor


def load_what_you_can(checkpoint: dict, model: torch.nn.Module):
    """
    This method takes a checkpoint and loads as many weights from it as possible:

    If they are the same shape, there's nothing to do

    Will load the smallest shape otherwise.
    """
    import torch

    model_state_dict = model.state_dict()
    checkpoint_state_dict = checkpoint

    for name, param in checkpoint_state_dict.items():
        if name not in model_state_dict:
            print(f"Ignoring parameter '{name}' because it is not found in the model")
            continue

        model_state = model_state_dict[name]
        mshape = model_state.shape
        pshape = param.shape

        if pshape == mshape:
            model_state.copy_(param)
            continue

        if len(pshape) != len(mshape):
            # Completely different shapes so probably unwise to merge
            continue

        min_shape = [
            min(param.shape[i], model_state.shape[i]) for i in range(len(param.shape))
        ]
        print(name, "model:", mshape, "chkpt:", pshape, "loading:", min_shape)
        idxs = torch.meshgrid(*[torch.arange(s) for s in min_shape])
        model_state[tuple(idxs)].copy_(param[tuple(idxs)])

    return model.load_state_dict(model_state_dict)


def multimap(
    items: list, func: callable, workers=4, desc=None, thread=False, chunk_size=128
) -> list:
    """
    Quick and dirty multiprocessing that will return the result of func if it returns None
    """
    from tqdm.contrib.concurrent import process_map, thread_map

    m = thread_map if thread else process_map
    length = None
    try:
        length = len(items)
    except Exception as e:
        print(e, "getting length")

    results = m(
        func,
        items,
        leave=False,
        desc=desc,
        max_workers=workers,
        total=length,
        chunksize=chunk_size,
    )
    return list(filter(lambda x: x is not None, results))


def round_up(num: float, factor: int):
    return factor * math.ceil(num / factor)


def left_padding_mask(lengths, max_len, device=None, dtype=None):
    masks = []
    if not max_len:
        max_len = max(lengths)
    for l in lengths:
        mask = torch.empty(l, l, device=device, dtype=dtype).fill_(-torch.inf).triu_(1)
        diff = max_len - l
        mask = F.pad(mask, (diff, 0, diff, 0), value=-torch.inf)
        masks.append(mask)

    masks = torch.stack(masks)
    return masks[:, None]


def seed_all(seed: int):
    import random

    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def split_bucket_path(url: str) -> tuple[str, str]:
    url = url.replace("s3://", "")
    url = url.replace("sj://", "")
    url = url.replace("r2://", "")
    bucket = url.split("/")[0]
    path = "/".join(url.split("/")[1:])
    return bucket, path


def prob_mask_like(shape, prob: float, device):
    import torch

    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def round_up_to_multiple(n: int, multiple: int) -> int:
    if n % multiple != 0:
        n += multiple - (n % multiple)

    return n


def warmup_then_cosine_decay(
    step: int, *, warmup_steps: int, steps: int, min_lr: float, max_lr: float
):
    eps = 1e-9
    cooldown_steps = warmup_steps
    if step < warmup_steps:
        return min_lr + step * (max_lr - min_lr) / (warmup_steps)
    elif step > steps:
        return min_lr
    elif step < steps - cooldown_steps:
        decay_ratio = (step - warmup_steps) / (steps - warmup_steps - cooldown_steps)
        # assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)
    else:
        # decay from min_lr to 0
        return min_lr * (steps - step) / cooldown_steps + eps


def decay_to_zero(step: int, *, decay_steps: int, steps: int, max_lr: float):
    if step > steps:
        return 0.0
    else:
        gradient = -max_lr / decay_steps

        return max_lr + gradient * step


def cross_entropy_loss(logits, mask, targets):
    import torch
    import torch.nn.functional as F

    B, Q, T, _ = logits.size()
    assert logits.shape[:-1] == targets.shape
    assert mask.shape == targets.shape
    loss = torch.zeros([], device=targets.device)
    codebook_losses = []
    for q in range(Q):
        logits_q = (
            logits[:, q, ...].contiguous().view(-1, logits.size(-1))
        )  # [B x T, card]
        targets_q = targets[:, q, ...].contiguous().view(-1)  # [B x T]
        mask_q = mask[:, q, ...].contiguous().view(-1)  # [B x T]
        ce_targets = targets_q[mask_q]
        ce_logits = logits_q[mask_q]
        q_ce = F.cross_entropy(ce_logits, ce_targets)
        loss += q_ce
        codebook_losses.append(q_ce.detach())
    # average cross entropy across codebooks
    loss = loss / Q
    return loss, codebook_losses


def build_optimizer(
    module, *, weight_decay: float, lr: float, betas: tuple[float, float]
):
    import torch

    param_dict = {pn: p for pn, p in module.named_parameters() if p.requires_grad}

    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    # num_decay_params = sum(p.numel() for p in decay_params)
    # num_nodecay_params = sum(p.numel() for p in nodecay_params)
    # print(
    #     f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    # )
    # print(
    #     f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    # )
    optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=betas, fused=True)

    return optimizer


def pad_or_cut_right(t: Tensor, padlen: int, value=0) -> Tensor:
    current_len = t.shape[-1]

    if current_len == padlen:
        return t

    if current_len < padlen:
        # Need to pad
        pad_size = (0, padlen - current_len)
        return F.pad(t, pad_size, value=value)
    # Need to cut
    return t[:padlen]


def pad_or_cut_left(t: Tensor, value: int) -> Tensor:
    dims = t.ndim
    current_len = t.shape[0]

    if current_len == value:
        return t

    if current_len < value:
        # Need to pad
        pad_size = (0,) * (2 * (dims - 1)) + (value - current_len, 0)
        return F.pad(t, pad_size)
    # Need to cut
    return t[-value:]


class timer:
    def __init__(self, name=""):
        self.name = name

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.t
        print(f"{self.name} {elapsed:.4f}")


def get_basename_without_extension(file_path):
    from pathlib import Path

    p = Path(file_path)
    return p.stem



def decompile_state_dict(state_dict):
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    # state_dict = convert_old_weight_norm_to_new(state_dict)
    return {k.replace("module.", ""): v for k, v in state_dict.items()}
