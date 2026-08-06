"""HAP: Head-adaptive Attention Pruning.

Each attention head attends only within its own scope, taken from the offline
scope plan, and the attention outside that scope is pruned. The mask is built
with PyTorch FlexAttention and cached per sequence length and layer.
"""
import time

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

BLOCK = 64
TEXT_LEN = 512
ANCHOR_OFF = 1 << 30


class HapRuntime:
    """Holds the compiled FlexAttention kernel and the per-layer block masks."""

    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, num_heads=24, device="cuda"):
        self.device = device
        self.num_heads = num_heads
        self.half_buf = torch.zeros(num_heads, device=device, dtype=torch.int32)
        self.anchor_buf = torch.tensor(ANCHOR_OFF, device=device, dtype=torch.int32)

        def mask_mod(b, h, q, k):
            text_q = q < TEXT_LEN                       # text queries attend everything
            text_k = k < TEXT_LEN                       # text keys are attended by everyone
            qb = (q - TEXT_LEN) // BLOCK
            kb = (k - TEXT_LEN) // BLOCK
            band = (qb - kb).abs() <= self.half_buf[h]  # this head's scope
            anchor = (kb % self.anchor_buf) == 0        # globally visible blocks
            return text_q | text_k | band | anchor

        self.mask_mod = mask_mod
        self.flex = torch.compile(flex_attention, dynamic=False)
        self.mask_cache = {}
        self.active = {}

    @staticmethod
    def band_blocks(alphas, betas, seq_len):
        """Scope (alpha, beta) -> width of the band in 64-token blocks."""
        nbx = seq_len // BLOCK
        return [max(2 * int(a // BLOCK + b * nbx) - 1, 1)
                for a, b in zip(alphas, betas)]

    def prepare_mask(self, seq_len, layer, band, anchor_stride=0):
        stride = anchor_stride if anchor_stride and anchor_stride > 0 else ANCHOR_OFF
        key = (seq_len, layer, stride)
        if key not in self.mask_cache:
            half = torch.tensor([(int(x) - 1) // 2 for x in band],
                                device=self.device, dtype=torch.int32)
            self.half_buf.copy_(half)
            self.anchor_buf.fill_(stride)
            block_mask = create_block_mask(
                self.mask_mod, B=None, H=self.num_heads,
                Q_LEN=seq_len, KV_LEN=seq_len, device=self.device, _compile=True)
            self.mask_cache[key] = (block_mask, half, stride)
        self.active[(seq_len, layer)] = key
        return self.mask_cache[key]

    def attn(self, query, key, value, layer, scale):
        block_mask, half, stride = self.mask_cache[self.active[(query.shape[2], layer)]]
        self.half_buf.copy_(half)
        self.anchor_buf.fill_(stride)
        return self.flex(query, key, value, block_mask=block_mask, scale=scale)


def iter_blocks(transformer):
    return list(transformer.transformer_blocks) + list(transformer.single_transformer_blocks)


def prepare_hap_masks(scope_plan, transformer, seq_len, anchor_stride=0, verbose=True):
    """Build the block mask of every layer for this sequence length."""
    runtime = HapRuntime.get()
    t0 = time.time()
    built = 0
    for layer, block in enumerate(iter_blocks(transformer)):
        band = runtime.band_blocks(scope_plan["alphas"][layer],
                                   scope_plan["betas"][layer], seq_len)
        before = len(runtime.mask_cache)
        runtime.prepare_mask(seq_len, layer, band, anchor_stride=anchor_stride)
        built += len(runtime.mask_cache) - before
        block.attn.processor.hap_layer_key = layer
    if verbose and built:
        print(f"[HAP] built {built} block masks for seq_len={seq_len} "
              f"in {time.time() - t0:.1f}s")


def set_hap(transformer, on):
    """Enable or disable HAP on every attention block."""
    for block in iter_blocks(transformer):
        block.attn.processor.hap_on = bool(on)
