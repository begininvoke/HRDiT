import inspect
import math
from typing import Callable, List, Optional, Tuple, Union
from diffusers.models.attention_processor import Attention

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import os
from torchvision.transforms import GaussianBlur

import pdb
import pywt
import torch.fft as fft
from torch import Tensor
import time
from .hap import HapRuntime

class Custom_FluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        self.block_size = 64
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        proportional_attention = True,
        image_rotary_emb_nor_list = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        spa_active = getattr(self, "spa_on", False) and image_rotary_emb_nor_list is not None
        if image_rotary_emb is not None and not spa_active:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        train_seq_len = 64 ** 2 + 512
        if proportional_attention:
            attention_scale = math.sqrt(math.log(key.size(2), train_seq_len) / head_dim)
        else:
            attention_scale = math.sqrt(1 / head_dim)

        if spa_active:
            hidden_states = self._spa_attention(
                query, key, value, image_rotary_emb_nor_list, attention_scale)
        elif getattr(self, "hap_on", False) and hasattr(self, "hap_layer_key"):
            hidden_states = HapRuntime.get().attn(
                query, key, value, self.hap_layer_key, attention_scale)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=False,
                scale=attention_scale)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

    def _spa_attention(self, query, key, value, rope_variants, attention_scale):
        """SPA (paper Eq.6) via per-variant flash attention.

        mean_n(softmax(A_n)) @ V == mean_n(softmax(A_n) @ V), so averaging the
        per-variant attention OUTPUTS is element-wise identical to averaging the
        attention maps, at O(V*T*D) memory instead of O(V*T^2). When hap_on is
        set the per-variant attention additionally uses the flex-HAP block mask.
        """
        use_hap = getattr(self, "hap_on", False) and hasattr(self, "hap_layer_key")
        acc = None
        for rope_v in rope_variants:
            q_v = apply_rotary_emb(query, rope_v)
            k_v = apply_rotary_emb(key, rope_v)
            if use_hap:
                out = HapRuntime.get().attn(q_v, k_v, value, self.hap_layer_key, attention_scale)
            else:
                out = F.scaled_dot_product_attention(
                    q_v, k_v, value, dropout_p=0.0, is_causal=False, scale=attention_scale)
            acc = out if acc is None else acc + out
        return acc / len(rope_variants)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None] # equal to .unsqueeze(0).unsqueeze(0)
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)

def prep_attn_processor(transformer):
    for name, module in transformer.named_modules():
        module_name = module.__class__.__name__  
        if name.split('.')[-1] == 'attn':
            attn_number = name.split('.')[-2]
            module.processor = Custom_FluxAttnProcessor2_0()
            module.processor.attn_number = attn_number
    return


def add_scope_plan(scope_plan, transformer):
    """Attach the per-head attention scopes to every attention processor."""
    for index_block, block in enumerate(transformer.transformer_blocks):
        alphas = scope_plan["alphas"][index_block]
        betas = scope_plan["betas"][index_block]
        block.attn.processor.alphas = alphas
        block.attn.processor.betas = betas

    for index_block, block in enumerate(transformer.single_transformer_blocks):
        alphas = scope_plan["alphas"][index_block + len(transformer.transformer_blocks)]
        betas = scope_plan["betas"][index_block + len(transformer.transformer_blocks)]
        block.attn.processor.alphas = alphas
        block.attn.processor.betas = betas


def remove_scope_plan(transformer):
    """Detach the attention scopes, restoring dense attention."""
    for index_block, block in enumerate(transformer.transformer_blocks):
        if hasattr(block.attn.processor, "alphas"):
            del block.attn.processor.alphas
        if hasattr(block.attn.processor, "betas"):
            del block.attn.processor.betas

    for index_block, block in enumerate(transformer.single_transformer_blocks):
        if hasattr(block.attn.processor, "alphas"):
            del block.attn.processor.alphas
        if hasattr(block.attn.processor, "betas"):
            del block.attn.processor.betas


def butterworth_low_pass_filter_2d(shape, device, n=4, ratio=0.25):
    """
    Compute the Butterworth low-pass filter mask for a 2D image.

    Args:
        shape: (H, W) shape of the filter
        n: order of the filter, larger n ~ ideal, smaller n ~ gaussian
        d_s: normalized stop frequency for spatial dimensions (0.0-1.0)
    """
    H, W = shape[-2], shape[-1]
    mask = torch.zeros(shape).to(device)
    d_s = ratio
    
    if d_s == 0:
        return mask
    
    for h in range(H):
        for w in range(W):
            d_square = ((2 * h / H - 1) ** 2 + (2 * w / W - 1) ** 2)
            mask[..., h, w] = 1 / (1 + (d_square / d_s ** 2) ** n)
    
    return mask

def set_seeds(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def gaussian_blur_image_sharpening(image, kernel_size=3, sigma=(0.1, 2.0), alpha=1):
    gaussian_blur = GaussianBlur(kernel_size=kernel_size, sigma=sigma)
    image_blurred = gaussian_blur(image)
    image_sharpened = (alpha + 1) * image - alpha * image_blurred

    return image_sharpened

def split_frequency_components_dwt(x, wavelet='haar', level=1):
    device = x.device
    dtype = x.dtype
    x = x.type(torch.float32)
    x = x.cpu().numpy()

    B, C, H, W = x.shape
    low_freq_components = []

    # Using list comprehension to improve performance
    for b in range(B):
        for c in range(C):
            coeffs = pywt.wavedec2(x[b, c], wavelet=wavelet, level=level)
            low_freq, *high_freq = coeffs
            low_freq_components.append([low_freq] + [(np.zeros_like(detail[0]), np.zeros_like(detail[1]), np.zeros_like(detail[2])) for detail in high_freq])

    # Convert list of numpy arrays to a single numpy array for better performance
    x_low_freq = np.stack([pywt.waverec2(low_freq_components[i], wavelet=wavelet) for i in range(B * C)])

    # Convert the numpy array to a tensor
    x_low_freq = torch.from_numpy(x_low_freq).view(B, C, H, W).type(dtype).to(device)

    return x_low_freq

def split_frequency_components_fft(x, freq_filter, is_low = True):
    x_freq = fft.fftshift(fft.fft2(x.to(freq_filter.dtype)))
    x_split_freq = x_freq * freq_filter if is_low else x_freq * (1 - freq_filter)
    x_split = fft.ifft2(fft.ifftshift(x_split_freq)).real
    return x_split


