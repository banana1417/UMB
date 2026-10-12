import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel, ViTModel

from config import UMBConfig


class CrossModalInteraction(nn.Module):
    """One directional CLS-to-sequence cross-modal interaction."""

    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.gate = nn.Linear(2 * hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, cls_query, sequence, key_padding_mask=None):
        attended, _ = self.attention(
            query=cls_query.unsqueeze(1),
            key=sequence,
            value=sequence,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        evidence = attended.squeeze(1)
        gate = torch.sigmoid(self.gate(torch.cat([cls_query, evidence], dim=-1)))
        return self.norm(cls_query + self.dropout(gate * evidence))


class BidirectionalCMI(nn.Module):
    """Apply CMI in both directions for a modality pair."""

    def __init__(self, hidden_size, num_heads, dropout):
        super().__init__()
        self.a_to_b = CrossModalInteraction(hidden_size, num_heads, dropout)
        self.b_to_a = CrossModalInteraction(hidden_size, num_heads, dropout)

    def forward(self, cls_a, seq_a, mask_a, cls_b, seq_b, mask_b):
        refined_a = self.a_to_b(cls_a, seq_b, mask_b)
        refined_b = self.b_to_a(cls_b, seq_a, mask_a)
        return refined_a, refined_b


class MultiHeadMLP(nn.Module):
    """Independent two-layer MLP applied to each of H feature chunks."""

    def __init__(self, hidden_size, heads, dropout):
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError("hidden_size must be divisible by umb_heads")
        head_dim = hidden_size // heads
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(head_dim, head_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(head_dim, head_dim),
                )
                for _ in range(heads)
            ]
        )
        self.heads = heads

    def forward(self, x):
        chunks = x.chunk(self.heads, dim=-1)
        return torch.cat([block(chunk) for block, chunk in zip(self.blocks, chunks)], dim=-1)


class UMBBranch(nn.Module):
    def __init__(self, hidden_size, heads, dropout, add_input_residual):
        super().__init__()
        self.multi_head = MultiHeadMLP(hidden_size, heads, dropout)
        self.post_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.post_norm = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.add_input_residual = add_input_residual

    def forward(self, h, x):
        z = self.multi_head(h)
        z = self.post_norm(h + self.post_mlp(z))
        z = self.ffn_norm(z + self.ffn(z) + (x if self.add_input_residual else 0.0))
        return z


class UncertaintyModelingBlock(nn.Module):
    """Map a refined feature to a diagonal Gaussian N(mu, sigma^2)."""

    def __init__(self, hidden_size, heads, dropout, eps):
        super().__init__()
        self.eps = eps
        self.shared = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.mean_branch = UMBBranch(hidden_size, heads, dropout, True)
        self.var_branch = UMBBranch(hidden_size, heads, dropout, False)
        self.mean_head = nn.Linear(hidden_size, hidden_size)
        self.var_head = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        h = self.shared(x)
        mean_feature = self.mean_branch(h, x)
        var_feature = self.var_branch(h, x)
        mu = self.mean_head(mean_feature)
        var = F.softplus(self.var_head(var_feature)) + self.eps
        return mu, var


class FakeNewsUMB(nn.Module):
    """FakeNewsUMB: title + image + true rationale + false rationale."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or UMBConfig()
        d = self.config.hidden_size

        self.title_encoder = RobertaModel.from_pretrained(self.config.text_model_name)
        self.true_reason_encoder = RobertaModel.from_pretrained(self.config.text_model_name)
        self.false_reason_encoder = RobertaModel.from_pretrained(self.config.text_model_name)
        self.image_encoder = ViTModel.from_pretrained(self.config.vision_model_name)

        self.title_image_cmi = BidirectionalCMI(
            d, self.config.cross_modal_heads, self.config.dropout
        )
        self.reason_cmi = BidirectionalCMI(
            d, self.config.cross_modal_heads, self.config.dropout
        )

        self.umb_title = UncertaintyModelingBlock(d, self.config.umb_heads, self.config.dropout, self.config.eps)
        self.umb_image = UncertaintyModelingBlock(d, self.config.umb_heads, self.config.dropout, self.config.eps)
        self.umb_true_reason = UncertaintyModelingBlock(d, self.config.umb_heads, self.config.dropout, self.config.eps)
        self.umb_false_reason = UncertaintyModelingBlock(d, self.config.umb_heads, self.config.dropout, self.config.eps)

        self.prediction_projection = nn.Linear(4 * d, d)
        self.prediction_norm = nn.LayerNorm(d)
        self.logit_mean_head = nn.Linear(d, 1)
        self.logit_log_variance_head = nn.Linear(d, 1)

        if self.config.freeze_backbones:
            self.freeze_backbones()

    def freeze_backbones(self):
        for encoder in (
            self.title_encoder,
            self.true_reason_encoder,
            self.false_reason_encoder,
            self.image_encoder,
        ):
            for parameter in encoder.parameters():
                parameter.requires_grad = False

    @staticmethod
    def _text(encoder, input_ids, attention_mask):
        output = encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence = output.last_hidden_state
        cls = sequence[:, 0]
        padding_mask = attention_mask.eq(0)
        return cls, sequence, padding_mask

    @staticmethod
    def _image(encoder, pixel_values):
        sequence = encoder(pixel_values=pixel_values).last_hidden_state
        return sequence[:, 0], sequence, None

    def _fuse(self, mus, variances):
        mean_variances = torch.stack([var.mean(dim=-1) for var in variances], dim=-1)
        inverse_variances = 1.0 / (mean_variances + self.config.eps)
        weights = F.softmax(inverse_variances, dim=-1)
        fused = torch.cat(
            [weights[:, i:i + 1] * mus[i] for i in range(len(mus))], dim=-1
        )
        return fused, weights

    def forward(
        self,
        image,
        title_input_ids,
        title_attention_mask,
        true_reason_input_ids,
        true_reason_attention_mask,
        false_reason_input_ids,
        false_reason_attention_mask,
    ):
        title_cls, title_seq, title_mask = self._text(
            self.title_encoder, title_input_ids, title_attention_mask
        )
        true_cls, true_seq, true_mask = self._text(
            self.true_reason_encoder, true_reason_input_ids, true_reason_attention_mask
        )
        false_cls, false_seq, false_mask = self._text(
            self.false_reason_encoder, false_reason_input_ids, false_reason_attention_mask
        )
        image_cls, image_seq, image_mask = self._image(self.image_encoder, image)

        title_refined, image_refined = self.title_image_cmi(
            title_cls, title_seq, title_mask,
            image_cls, image_seq, image_mask,
        )
        true_refined, false_refined = self.reason_cmi(
            true_cls, true_seq, true_mask,
            false_cls, false_seq, false_mask,
        )

        mu_title, var_title = self.umb_title(title_refined)
        mu_image, var_image = self.umb_image(image_refined)
        mu_true, var_true = self.umb_true_reason(true_refined)
        mu_false, var_false = self.umb_false_reason(false_refined)

        fused, modality_weights = self._fuse(
            [mu_title, mu_image, mu_true, mu_false],
            [var_title, var_image, var_true, var_false],
        )
        hidden = self.prediction_norm(F.relu(self.prediction_projection(fused)))
        logit_mu = self.logit_mean_head(hidden).squeeze(-1)
        pred_var = F.softplus(self.logit_log_variance_head(hidden).squeeze(-1))
        pred_var = pred_var + self.config.pred_variance_floor

        return {
            "logit_mu": logit_mu,
            "pred_var": pred_var,
            "prob": torch.sigmoid(logit_mu),
            "modality_weights": modality_weights,
            "mu_title": mu_title,
            "var_title": var_title,
            "mu_image": mu_image,
            "var_image": var_image,
            "mu_true_reason": mu_true,
            "var_true_reason": var_true,
            "mu_false_reason": mu_false,
            "var_false_reason": var_false,
        }

    def sample_predictions(self, outputs, num_samples=None):
        num_samples = num_samples or self.config.reparameterization_samples
        std = torch.sqrt(outputs["pred_var"].clamp_min(self.config.eps))
        noise = torch.randn(
            num_samples, *outputs["logit_mu"].shape,
            device=std.device,
            dtype=std.dtype,
        )
        logits = outputs["logit_mu"].unsqueeze(0) + noise * std.unsqueeze(0)
        return torch.sigmoid(logits).mean(dim=0)
