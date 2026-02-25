import math
from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from .augmentations import GaussianSmoothing


class PatchEmbedding(nn.Module):
    """
    Segments neural sequences into non-overlapping patches of length Tin and
    projects each patch into the model dimension.
    """

    def __init__(self, Tin: int, feat_dim: int, hdim: int, eps: float = 1e-6):
        super().__init__()
        self.Tin = Tin
        self.feat_dim = feat_dim
        self.in_dim = Tin * feat_dim
        self.ln1 = nn.LayerNorm(self.in_dim, eps=eps)
        self.proj = nn.Linear(self.in_dim, hdim)
        self.ln2 = nn.LayerNorm(hdim, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        bsz, T, F = x.shape
        if F != self.feat_dim:
            raise ValueError(f"expected feature dim {self.feat_dim}, got {F}")

        n_patch = T // self.Tin
        if n_patch == 0:
            raise ValueError("input sequence shorter than Tin")

        x = x[:, : n_patch * self.Tin, :]
        x = x.view(bsz, n_patch, self.Tin * F)
        x = self.ln1(x)
        x = self.proj(x)
        x = self.ln2(x)
        return x


class RelativePositionBias(nn.Module):
    """Relative attention bias shared with the teammate's implementation."""

    def __init__(self, n_heads: int, max_dist: int, by_head: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.max_dist = max_dist
        self.L = 2 * max_dist + 1
        self.by_head = by_head
        if by_head:
            self.relative_bias = nn.Parameter(torch.zeros(n_heads, self.L))
        else:
            self.relative_bias = nn.Parameter(torch.zeros(1, self.L))

    def forward(self, seq_len: int, device=None) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        rel = positions[None, :] - positions[:, None]
        clipped = rel.clamp(-self.max_dist, self.max_dist) + self.max_dist
        table = self.relative_bias
        bias = table[..., clipped]
        if not self.by_head:
            bias = bias.expand(self.n_heads, -1, -1)
        return bias


class Attention(nn.Module):
    def __init__(self, hdim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if hdim % n_heads != 0:
            raise ValueError("hdim must be divisible by number of heads")
        self.hdim = hdim
        self.n_heads = n_heads
        self.head_dim = hdim // n_heads

        self.qkv = nn.Linear(hdim, 3 * hdim)
        self.out = nn.Linear(hdim, hdim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None, rel_bias=None, key_padding_mask=None):
        bsz, seq_len, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(dim, dim=-1)

        def reshape_heads(tensor):
            return tensor.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if rel_bias is not None:
            scores = scores + rel_bias.unsqueeze(0)

        if causal_mask is not None:
            if causal_mask.dim() == 2:
                scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)
            elif causal_mask.dim() == 4:
                scores = scores + causal_mask
            else:
                raise ValueError("Unsupported causal_mask rank")

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        out = self.out(out)
        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return out


class FeedForward(nn.Module):
    def __init__(self, hdim: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln = nn.LayerNorm(hdim)
        self.fc1 = nn.Linear(hdim, ff_dim)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ff_dim, hdim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(
        self,
        hdim: int,
        n_heads: int,
        ff_dim: int,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.ln_attn = nn.LayerNorm(hdim)
        self.attn = Attention(hdim, n_heads, attn_dropout)
        self.ff = FeedForward(hdim, ff_dim, dropout)

    def forward(self, x, causal_mask=None, rel_bias=None, key_padding_mask=None):
        attn_in = self.ln_attn(x)
        attn_out = self.attn(
            attn_in, causal_mask=causal_mask, rel_bias=rel_bias, key_padding_mask=key_padding_mask
        )
        x = x + attn_out
        ff_out = self.ff(x)
        x = x + ff_out
        return x

# Will probably need to redefine this final architecture using the above blocks
# since it was specialized for another task previously
class StreamingTransformerDecoder(nn.Module):
    def __init__(
        self,
        neural_dim: int,
        n_phonemes: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        stride_len: int,
        kernel_len: int,
        gaussian_smooth_width: float,
        intermediate_layer: int,
        day_count: int,
        diphone_context: Optional[int] = None,
        time_mask_prob: float = 0.0,
        rel_pos_max_dist: Optional[int] = None,
        rel_bias_by_head: bool = True,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.neural_dim = neural_dim
        self.n_phonemes = n_phonemes
        self.blank_count = n_phonemes + 1
        self.kernel_len = kernel_len
        self.stride_len = stride_len
        self.gaussian = GaussianSmoothing(neural_dim, 20, gaussian_smooth_width, dim=1)
        self.time_mask_prob = time_mask_prob

        self.day_weights = nn.Parameter(torch.randn(day_count, neural_dim, neural_dim))
        self.day_bias = nn.Parameter(torch.zeros(day_count, 1, neural_dim))
        for d in range(day_count):
            with torch.no_grad():
                self.day_weights[d] = torch.eye(neural_dim)

        self.input_nonlin = nn.Softsign()
        self.patch_embed = PatchEmbedding(kernel_len, neural_dim, d_model)
        self.input_dropout = nn.Dropout(dropout)

        if intermediate_layer >= num_layers:
            raise ValueError("intermediate_layer must be < num_layers")
        self.intermediate_layer = intermediate_layer

        ff_dim = ff_mult * d_model
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    hdim=d_model,
                    n_heads=nhead,
                    ff_dim=ff_dim,
                    attn_dropout=dropout,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.rel_pos_bias = (
            RelativePositionBias(nhead, rel_pos_max_dist, rel_bias_by_head)
            if rel_pos_max_dist is not None
            else None
        )
        self.final_ln = nn.LayerNorm(d_model)

        self.intermediate_head = nn.Linear(d_model, self.blank_count)
        self.feedback_proj = nn.Linear(self.blank_count, d_model)
        self.final_head = nn.Linear(d_model, self.blank_count)

        diphone_dim = diphone_context or n_phonemes
        self.diphone_dim = diphone_dim
        self.diphone_head = nn.Linear(d_model, diphone_dim * diphone_dim)

    def _apply_preproc(
        self, neural: torch.Tensor, lengths: torch.Tensor, day_idx: torch.Tensor
    ) -> torch.Tensor:
        neural = torch.permute(neural, (0, 2, 1))
        neural = self.gaussian(neural)
        neural = torch.permute(neural, (0, 2, 1))

        weights = torch.index_select(self.day_weights, 0, day_idx)
        transformed = torch.einsum("btd,bdk->btk", neural, weights) + torch.index_select(
            self.day_bias, 0, day_idx
        )
        transformed = self.input_nonlin(transformed)
        max_len = transformed.shape[1]
        mask = torch.arange(max_len, device=transformed.device).unsqueeze(0) >= lengths.unsqueeze(1)
        transformed = transformed.masked_fill(mask.unsqueeze(-1), 0.0)
        return transformed

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.float32)
        mask = torch.triu(mask, diagonal=1)
        mask = mask * -1e9
        return mask

    @staticmethod
    def _padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        positions = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return positions >= lengths.unsqueeze(1)

    def _apply_time_mask(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.time_mask_prob <= 0.0:
            return x
        mask = torch.rand(x.shape[0], x.shape[1], device=x.device) < self.time_mask_prob
        return x.masked_fill(mask.unsqueeze(-1), 0.0)

    def marginalize_diphone(self, diphone_logits: torch.Tensor) -> torch.Tensor:
        bsz, steps, _ = diphone_logits.shape
        logits = diphone_logits.view(bsz, steps, self.diphone_dim, self.diphone_dim)
        log_probs = F.log_softmax(logits.view(bsz, steps, -1), dim=-1).view(
            bsz, steps, self.diphone_dim, self.diphone_dim
        )
        left = torch.logsumexp(log_probs, dim=3)
        right = torch.logsumexp(log_probs, dim=2)
        combined = torch.logaddexp(left, right) - math.log(2.0)
        combined = combined - torch.logsumexp(combined, dim=-1, keepdim=True)
        return combined

    def forward(
        self,
        neural: torch.Tensor,
        lengths: torch.Tensor,
        day_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        transformed = self._apply_preproc(neural, lengths, day_idx)
        patches = self.patch_embed(transformed)
        patches = self.input_dropout(patches)
        patches = self._apply_time_mask(patches)

        seq_len = patches.shape[1]
        device = patches.device
        causal_mask = self._causal_mask(seq_len, device)
        rel_bias = self.rel_pos_bias(seq_len, device) if self.rel_pos_bias else None

        eff_lengths = torch.div(lengths, self.kernel_len, rounding_mode="floor").clamp(min=1)
        pad_mask = self._padding_mask(eff_lengths, seq_len)

        hidden = patches
        intermediate_logits = None
        intermediate_log_probs = None
        for idx, layer in enumerate(self.layers):
            hidden = layer(
                hidden, causal_mask=causal_mask, rel_bias=rel_bias, key_padding_mask=pad_mask
            )
            if idx == self.intermediate_layer:
                intermediate_logits = self.intermediate_head(hidden)
                intermediate_log_probs = F.log_softmax(intermediate_logits, dim=-1)
                feedback = self.feedback_proj(intermediate_logits)
                hidden = hidden + feedback

        hidden = self.final_ln(hidden)
        final_logits = self.final_head(hidden)
        final_log_probs = F.log_softmax(final_logits, dim=-1)

        diphone_logits = self.diphone_head(hidden)
        diphone_log_probs = self.marginalize_diphone(diphone_logits)

        blank = final_log_probs[..., :1]
        phone_log_probs = final_log_probs[..., 1:]
        valid_dim = min(self.diphone_dim, self.n_phonemes)
        diphone_slice = diphone_log_probs[..., :valid_dim]
        phone_slice = phone_log_probs[..., :valid_dim]
        if valid_dim < self.n_phonemes:
            pad_width = self.n_phonemes - valid_dim
            diphone_slice = F.pad(diphone_slice, (0, pad_width), value=-100)
            phone_slice = F.pad(phone_slice, (0, pad_width), value=-100)

        fused_phone = torch.logsumexp(
            torch.stack([phone_slice, diphone_slice], dim=0),
            dim=0,
        ) - math.log(2.0)
        fused = torch.cat([blank, fused_phone], dim=-1)
        fused = fused - torch.logsumexp(fused, dim=-1, keepdim=True)

        return {
            "log_probs": fused,
            "intermediate_log_probs": intermediate_log_probs,
            "intermediate_logits": intermediate_logits,
            "raw_logits": final_logits,
            "diphone_logits": diphone_logits,
            "diphone_log_probs": diphone_log_probs,
            "eff_lengths": eff_lengths.to(torch.int32),
        }


