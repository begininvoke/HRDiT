"""SPA: Spatial Position Alignment.

Token indices are mapped to bundle indices before they enter the positional
encoding, and the attention contribution is averaged over a set of bundle
mappings whose boundaries slide.
"""
import math

import torch


def _phi(x, n1, size):
    """Bundle mapping: 0 for x < n1, else ceil((x + 1 - n1) / size)."""
    return torch.where(x < n1, torch.zeros_like(x), (x + 1 - n1 + size - 1) // size)


def build_bundle_id_variants(img_ids, group_num):
    """Build the bundle-index variants of ``img_ids`` used by SPA.

    ``img_ids`` is [T_img, 3] with column 1 the row index and column 2 the
    column index. ``group_num`` sets how many bundles each axis is divided into.
    Returns one variant per sliding position of the bundle boundaries.
    """
    rows = img_ids[:, 1].long()
    cols = img_ids[:, 2].long()
    s_row = max(1, math.ceil(rows.max().item() / (group_num - 1)))
    s_col = max(1, math.ceil(cols.max().item() / (group_num - 1)))

    def variant(n1_row, n1_col):
        ids = img_ids.clone()
        ids[:, 1] = _phi(rows, n1_row, s_row).to(img_ids.dtype)
        ids[:, 2] = _phi(cols, n1_col, s_col).to(img_ids.dtype)
        return ids

    variants = [variant(s_row, s_col)]
    variants += [variant(n, s_col) for n in range(1, s_row)]
    variants += [variant(s_row, m) for m in range(1, s_col)]
    return variants


def _iter_blocks(transformer):
    return list(transformer.transformer_blocks) + list(transformer.single_transformer_blocks)


def set_spa(transformer, on):
    """Enable or disable SPA on every attention block."""
    for block in _iter_blocks(transformer):
        processor = block.attn.processor
        processor.spa_on = bool(on) and getattr(processor, "spa_allowed", True)


def set_spa_filter(transformer, double_ids=None, single_ids=None):
    """Restrict SPA to a subset of blocks; ``None`` allows every block."""
    double = list(transformer.transformer_blocks)
    single = list(transformer.single_transformer_blocks)
    for idx, block in enumerate(double):
        block.attn.processor.spa_allowed = double_ids is None or idx in double_ids
    for idx, block in enumerate(single):
        block.attn.processor.spa_allowed = single_ids is None or idx in single_ids
