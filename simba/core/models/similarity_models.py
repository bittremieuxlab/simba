import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from simba.core.models.spectrum_encoder import (
    SpectrumTransformerEncoderCustom,
)
from simba.utils.logger_setup import logger


def _filip_aggregate_last_dim(sim, valid_mask, aggregation, temperature=None):
    """Reduce `sim`'s last axis according to `aggregation` -- shared by the
    paired (_filip_aggregate_over_j) and batched (_filip_similarity_matrix)
    FILIP aggregation paths, so the token-level contrastive loss can mirror
    whichever aggregation the primary head uses. `valid_mask` must be a bool
    mask broadcastable against `sim`, True for real (non-padding) entries
    along that same last axis.
    - "hard_max": literal max over the valid axis -- the original FILIP
      paper's mechanism, no learned parameter involved in this step.
    - "mean": literal unweighted average over the valid axis -- no max, no
      attention, just the plain average of the raw cosine similarities.
    - "soft_max": learned-temperature log-mean-exp (`temperature` required,
      already clamped by the caller) -- interpolates between the two above
      as training adjusts the temperature. The "- log(n_valid)" correction
      is essential here (unlike hard_max/mean, which have no such bias):
      without it, raw log-sum-exp carries a systematic bias proportional to
      how many valid entries exist, unrelated to actual similarity."""
    neg_fill = -1e9
    n_valid = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1).float()

    if aggregation == "hard_max":
        return sim.masked_fill(~valid_mask, neg_fill).max(dim=-1).values

    if aggregation == "mean":
        return sim.masked_fill(~valid_mask, 0.0).sum(dim=-1) / n_valid.squeeze(-1)

    # "soft_max"
    sim_masked = sim.masked_fill(~valid_mask, neg_fill)
    return temperature * (
        torch.logsumexp(sim_masked / temperature, dim=-1) - n_valid.squeeze(-1).log()
    )


def _filip_paired_weighted_mean(
    per_token_scores, tokens, valid, use_importance_weighting, importance_mlp
):
    """Pool per-token scores over the valid-token axis. Plain masked mean by
    default; when `use_importance_weighting` is set, weights each token by
    a learned, independently-scored (sigmoid, not softmax -- no forced
    competition between peaks, so several peaks can all matter at once)
    importance gate instead of treating every peak equally. Shared by any
    head using the paired FILIP mechanism (see _filip_paired_score), each
    with its own dedicated `importance_mlp`."""
    if use_importance_weighting:
        importance = torch.sigmoid(importance_mlp(tokens).squeeze(-1))
        importance = torch.where(valid, importance, torch.zeros_like(importance))
        return (importance * per_token_scores).sum(dim=1) / importance.sum(dim=1).clamp(
            min=1e-6
        )
    return per_token_scores.sum(dim=1) / valid.sum(dim=1).clamp(min=1)


def _filip_paired_cross_attention_score(
    tokens0,
    valid0,
    tokens1,
    valid1,
    attn_q,
    attn_k,
    weight_similarities,
    use_importance_weighting,
    importance_mlp,
):
    """Cross-attention alternative to the max/log-mean-exp aggregation:
    each spectrum's tokens attend over the OTHER spectrum's valid tokens
    via standard scaled-dot-product attention (masked at padding
    positions), producing learned attention weights instead of a fixed
    temperature/max-based combination rule. `attn_q`/`attn_k` are learned
    projections used only to derive those weights.

    Two ways to combine the weights with the actual token comparison
    (`weight_similarities` selects which):
    - False: blend the RAW token vectors by the attention weights first,
      then take one cosine similarity against the raw query token.
      Cheaper, but cosine is not linear, so blending vectors that point in
      different directions before comparing can partially cancel out real
      per-token signal.
    - True: compute the full raw-token cosine-similarity matrix first, then
      take the attention-weighted AVERAGE of those scalar similarities.
      Avoids the vector-cancellation issue since scalars can't
      destructively interfere, and keeps the comparison itself unchanged
      from the base cosine kernel -- only the combination rule is learned.

    Either way: averaged over valid tokens (see _filip_paired_weighted_mean),
    then symmetrically averaged over both directions. Returns (B,)."""
    d_k = tokens0.shape[-1]
    scale = d_k**0.5
    neg_fill = -1e9

    def _direction(q_tokens, q_valid, kv_tokens, kv_valid):
        q = attn_q(q_tokens)  # (B, N, d)
        k = attn_k(kv_tokens)  # (B, M, d)
        logits = torch.bmm(q, k.transpose(1, 2)) / scale  # (B, N, M)
        logits = logits.masked_fill(~kv_valid.unsqueeze(1), neg_fill)
        weights = torch.softmax(logits, dim=2)
        if weight_similarities:
            q_norm = F.normalize(q_tokens, p=2, dim=-1)
            kv_norm = F.normalize(kv_tokens, p=2, dim=-1)
            cos_sim = torch.bmm(q_norm, kv_norm.transpose(1, 2))  # (B, N, M)
            per_token = (weights * cos_sim).sum(dim=2)  # (B, N)
        else:
            attended = torch.bmm(weights, kv_tokens)  # (B, N, d) -- raw values
            per_token = F.cosine_similarity(
                attended, q_tokens, dim=-1
            )  # (B, N) -- raw query
        per_token = torch.where(q_valid, per_token, torch.zeros_like(per_token))
        return _filip_paired_weighted_mean(
            per_token, q_tokens, q_valid, use_importance_weighting, importance_mlp
        )

    score_0to1 = _direction(tokens0, valid0, tokens1, valid1)
    score_1to0 = _direction(tokens1, valid1, tokens0, valid0)
    return 0.5 * (score_0to1 + score_1to0)


def _filip_paired_aggregation_score(
    tokens0,
    valid0,
    tokens1,
    valid1,
    aggregation,
    log_temperature,
    use_importance_weighting,
    importance_mlp,
):
    """FILIP-style fine-grained token similarity (Yao et al., 2021), no
    cross-attention: L2-normalize every peak token, build the full pairwise
    cosine-similarity matrix between the two spectra's token sequences,
    mask out padding ("decoy") tokens on both sides, then for each
    direction combine the other side's valid tokens via `aggregation`
    ("soft_max" learned log-mean-exp -- needs `log_temperature`,
    "hard_max", or "mean" -- see _filip_aggregate_last_dim) and pool over
    this side's valid tokens (see _filip_paired_weighted_mean); the final
    score is the average of the two directional scores. Returns (B,)."""
    t0 = F.normalize(tokens0, p=2, dim=-1)  # (B, N, D)
    t1 = F.normalize(tokens1, p=2, dim=-1)  # (B, M, D)
    raw_sim = torch.bmm(t0, t1.transpose(1, 2))  # (B, N, M)

    def _aggregate_over_j(sim, valid_j):
        mask = valid_j.unsqueeze(1)  # (B, 1, M), broadcasts over N
        if aggregation == "soft_max":
            temperature = log_temperature.exp().clamp(min=1e-3, max=10.0)
            return _filip_aggregate_last_dim(sim, mask, "soft_max", temperature)
        return _filip_aggregate_last_dim(sim, mask, aggregation)

    # direction 0 -> 1: combine spectrum 1's valid tokens (last axis).
    per_peak_0to1 = _aggregate_over_j(raw_sim, valid1)  # (B, N)
    per_peak_0to1 = torch.where(valid0, per_peak_0to1, torch.zeros_like(per_peak_0to1))
    score_0to1 = _filip_paired_weighted_mean(
        per_peak_0to1, tokens0, valid0, use_importance_weighting, importance_mlp
    )

    # direction 1 -> 0: transpose so spectrum 0's tokens are again the
    # last axis, reusing the exact same aggregation logic.
    per_peak_1to0 = _aggregate_over_j(raw_sim.transpose(1, 2), valid0)  # (B, M)
    per_peak_1to0 = torch.where(valid1, per_peak_1to0, torch.zeros_like(per_peak_1to0))
    score_1to0 = _filip_paired_weighted_mean(
        per_peak_1to0, tokens1, valid1, use_importance_weighting, importance_mlp
    )

    return 0.5 * (score_0to1 + score_1to0)


def _filip_paired_score(
    tokens0,
    valid0,
    tokens1,
    valid1,
    use_cross_attention,
    weight_similarities,
    aggregation,
    use_importance_weighting,
    attn_q=None,
    attn_k=None,
    importance_mlp=None,
    log_temperature=None,
):
    """The full paired (B,) FILIP token-interaction score -- shared by any
    head using this architecture with its own dedicated weights: the
    primary head (SimilarityModelMultitask._filip_similarity) and the
    FILIP-token spectral-cosine head
    (SimilarityModelMultitask._filip_spectral_cosine_predict)."""
    if use_cross_attention:
        return _filip_paired_cross_attention_score(
            tokens0,
            valid0,
            tokens1,
            valid1,
            attn_q,
            attn_k,
            weight_similarities,
            use_importance_weighting,
            importance_mlp,
        )
    return _filip_paired_aggregation_score(
        tokens0,
        valid0,
        tokens1,
        valid1,
        aggregation,
        log_temperature,
        use_importance_weighting,
        importance_mlp,
    )


def _filip_masked_mean_pool(tokens, valid):
    """Plain masked mean over the valid-token axis, pooling per-token
    VECTORS into one representation per spectrum (unlike
    _filip_paired_weighted_mean, which pools per-token scalar scores).
    Parameter-free -- used by the FILIP-token mces_bucket head, which
    (unlike the primary/contrastive/spectral-cosine heads) never compares
    the two spectra's tokens against each other: it only needs one
    independent representation per spectrum, combined afterwards via
    abs-difference exactly as the original CLS-embedding version did.
    tokens: (B, N, D); valid: (B, N). Returns (B, D)."""
    valid_f = valid.unsqueeze(-1).to(tokens.dtype)
    return (tokens * valid_f).sum(dim=1) / valid_f.sum(dim=1).clamp(min=1)


class FixedLinearRegression(nn.Module):
    """
    linear layer for computing sum of dot product
    """

    def __init__(self, d_model):
        super().__init__()
        self.weight = nn.Parameter(
            torch.ones(1, d_model)
        )  # Fixed weight initialized to 1
        self.bias = nn.Parameter(torch.zeros(1))  # Bias initialized to 0

        # Freeze the parameters
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias


class SimilarityModel(pl.LightningModule):
    """It receives a set of pairs of molecules and it must train the similarity model based on it. Embed spectra."""

    def __init__(
        self,
        d_model,
        n_layers,
        dropout=0.1,
        weights=None,
        lr=None,
        use_element_wise=True,
        use_cosine_distance=True,  # element wise instead of concat for mixing info between embeddings
        use_adduct=False,
        use_ce=False,
        use_ion_activation=False,
        use_ion_method=False,
        use_ion_mode=False,
    ):
        """Initialize the CCSPredictor"""
        super().__init__()
        self.weights = weights

        # Add a linear layer for projection
        self.use_element_wise = use_element_wise
        self.linear = nn.Linear(d_model, d_model)
        self.linear_regression = nn.Linear(d_model, 1)
        self.fixed_linear_regression = FixedLinearRegression(d_model)

        self.relu = nn.ReLU()
        self.use_adduct = use_adduct
        self.use_ce = use_ce
        self.use_ion_activation = use_ion_activation
        self.use_ion_method = use_ion_method
        self.use_ion_mode = use_ion_mode

        self.spectrum_encoder = SpectrumTransformerEncoderCustom(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            use_adduct=use_adduct,
            use_ce=use_ce,
            use_ion_activation=use_ion_activation,
            use_ion_method=use_ion_method,
            use_ion_mode=use_ion_mode,
        )

        self.regression_loss = nn.MSELoss()
        self.dropout = nn.Dropout(p=dropout)

        self.train_loss_list = []
        self.val_loss_list = []
        self.lr = lr
        self.use_cosine_distance = use_cosine_distance
        if self.use_cosine_distance:
            self.linear_cosine = nn.Linear(d_model, d_model)

        self.cosine_similarity = nn.CosineSimilarity(dim=1)

        self.use_cosine_library = True

        # print(f"Using cosine library from Pytorch?: {self.use_cosine_library}")

    def normalized_dot_product(self, a, b):
        # Normalize inputs
        a_norm = torch.nn.functional.normalize(a, p=2, dim=-1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=-1)

        # Compute dot product
        dot_product = torch.sum(a_norm * b_norm, dim=-1)
        return dot_product

    def forward(self, batch):
        """The inference pass"""

        kwargs_0 = {
            "precursor_mass": batch["precursor_mass_0"].float(),
        }
        kwargs_1 = {
            "precursor_mass": batch["precursor_mass_1"].float(),
        }
        # extra data
        if self.use_ion_mode:
            kwargs_0["ionmode"] = batch["ionmode_0"].float()
            kwargs_1["ionmode"] = batch["ionmode_1"].float()
            kwargs_0["precursor_charge"] = batch["precursor_charge_0"].float()
            kwargs_1["precursor_charge"] = batch["precursor_charge_1"].float()
        if self.use_adduct:
            kwargs_0["ionmode"] = batch["ionmode_0"].float()
            kwargs_1["ionmode"] = batch["ionmode_1"].float()
            kwargs_0["adduct"] = batch["adduct_0"].float()
            kwargs_1["adduct"] = batch["adduct_1"].float()

        if self.use_ce:
            logger.info("Using CE in the model")
            kwargs_0["ce"] = batch["ce_0"].float()
            kwargs_1["ce"] = batch["ce_1"].float()

        if self.use_ion_activation:
            kwargs_0["ion_activation"] = batch["ion_activation_0"].float()
            kwargs_1["ion_activation"] = batch["ion_activation_1"].float()

        if self.use_ion_method:
            kwargs_0["ion_method"] = batch["ion_method_0"].float()
            kwargs_1["ion_method"] = batch["ion_method_1"].float()

        # ensure there are no nans
        batch["mz_0"] = torch.nan_to_num(batch["mz_0"], nan=0.0, posinf=0.0, neginf=0.0)
        batch["mz_1"] = torch.nan_to_num(batch["mz_1"], nan=0.0, posinf=0.0, neginf=0.0)
        batch["intensity_0"] = torch.nan_to_num(
            batch["intensity_0"], nan=0.0, posinf=0.0, neginf=0.0
        )
        batch["intensity_1"] = torch.nan_to_num(
            batch["intensity_1"], nan=0.0, posinf=0.0, neginf=0.0
        )

        emb0, _ = self.spectrum_encoder(
            mz_array=batch["mz_0"].float(),
            intensity_array=batch["intensity_0"].float(),
            **kwargs_0,
        )
        emb1, _ = self.spectrum_encoder(
            mz_array=batch["mz_1"].float(),
            intensity_array=batch["intensity_1"].float(),
            **kwargs_1,
        )

        emb0 = emb0[:, 0, :]
        emb1 = emb1[:, 0, :]

        emb0 = self.relu(emb0)
        emb1 = self.relu(emb1)

        if self.use_cosine_distance:
            if self.use_cosine_library:
                emb = self.cosine_similarity(emb0, emb1)

                # Reshape the tensor
                emb = emb.reshape(-1, 1)

            else:
                # ensure the embeddings are positive
                emb0_l2 = torch.norm(emb0, p=2, dim=-1, keepdim=True)
                emb1_l2 = torch.norm(emb1, p=2, dim=-1, keepdim=True)
                emb = (emb0 * emb1) / (emb0_l2 * emb1_l2)
                emb = self.fixed_linear_regression(emb)
                # emb = (emb+1)/2

        else:
            emb = emb0 + emb1
            emb = self.linear(emb)
            emb = self.dropout(emb)
            emb = self.relu(emb)
            emb = self.linear_regression(emb)

        return emb

    def step(self, batch, batch_idx, threshold=0.5):
        """A training/validation/inference step."""
        spec = self(batch)

        target = torch.tensor(batch["similarity"]).to(self.device)
        target = target.view(-1)

        # adjust scale
        # target = 2*(target-0.5)
        loss = self.regression_loss(spec.float(), target.view(-1, 1).float()).float()

        return loss.float()

    def training_step(self, batch, batch_idx):
        """A training step"""
        loss = self.step(batch, batch_idx)
        # self.train_loss_list.append(loss.item())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """A validation step"""
        loss = self.step(batch, batch_idx)
        # self.val_loss_list.append(loss.item())
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        """A predict step"""
        spec = self(batch)
        # if self.use_cosine_library:
        # spec= (spec+1)/2
        return spec

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        # optimizer = DAdaptAdam(self.parameters(), lr=1)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.RAdam(self.parameters(), lr=1e-3)
        return optimizer

    def load_weights(self):
        weights = {}
        for name, param in self.named_parameters():
            weights[name] = np.array(param.data)
        return weights

    def load_pretrained_maldi_embedder(self, model_path):
        # original weights
        original_weights = self.load_weights()

        # Load weights from the checkpoint
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
        )

        # Load weights into model B from the checkpoint
        checkpoint_keys = checkpoint["state_dict"].keys()
        original_embedder_keys = (
            self.state_dict().keys()
        )  # Assuming `model` is your target model

        # Load weights for shared layers
        for key in checkpoint_keys:
            if key in original_embedder_keys:
                self.state_dict()[key].copy_(checkpoint["state_dict"][key])

        # new weights
        new_weights = self.load_weights()

        ## sanity check (the weights of the model changed?):
        if not (self.are_weights_changed(original_weights, new_weights)):
            print("INFO: Correctly loaded pretrained Maldi Model")
        else:
            raise ValueError("ERROR!!!: Error loading Maldi model")

    def are_weights_changed(
        self,
        original_weights,
        new_weights,
        layer_test="spectrum_encoder.transformer_encoder.layers.0.norm2.bias",
    ):
        return np.array_equal(original_weights[layer_test], new_weights[layer_test])

    def set_freeze_layers(self, layer_names_to_freeze, freeze):
        # Freeze specified layers
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names_to_freeze):
                param.requires_grad = not (freeze)
            else:
                param.requires_grad = True

    def get_maldi_embedder_keys(self, model_path):
        # Load weights from the checkpoint
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
        )

        # Load weights into model B from the checkpoint
        return checkpoint["state_dict"].keys()

    def get_all_keys(self):
        return self.state_dict().keys()


class SimilarityModelMultitask(SimilarityModel):
    """It receives a set of pairs of molecules and it must train the similarity model based on it. Embeds spectra."""

    def __init__(
        self,
        d_model,
        n_layers,
        dropout=0.1,
        weights=None,
        lr=None,
        use_element_wise=True,
        use_cosine_distance=True,  # element wise instead of concat for mixing info between embeddings
        mces_max_value=40.0,  # must match model.tasks.mces.max_value; used by the mces_bucket head
        use_mces_bucket_head=False,  # optional second target: CORN-style ordinal classification on MCES buckets
        mces_bucket_bin_edges=None,  # required (from config) when use_mces_bucket_head=True
        mces_bucket_use_mlp=False,
        mces_bucket_loss_weight=1.0,
        mces_bucket_use_filip_tokens=False,  # build bucket_repr from a plain masked mean of the valid FILIP tokens instead of the CLS embeddings
        use_contrastive_loss=False,
        contrastive_temperature=0.07,
        contrastive_loss_weight=1.0,
        contrastive_use_projection_head=False,
        contrastive_use_filip_tokens=False,  # score the in-batch contrastive loss with FILIP token interaction instead of CLS cosine
        use_spectral_cosine_head=False,
        spectral_cosine_loss_weight=1.0,
        spectral_cosine_use_filip_tokens=False,  # score the spectral-cosine head with FILIP token interaction instead of CLS cosine
        use_filip_head=False,  # FILIP-style fine-grained token similarity, replaces the CLS-cosine primary score
        filip_use_cross_attention=False,  # replace the max/log-mean-exp aggregation with learned cross-attention
        filip_aggregation="soft_max",  # only used when filip_use_cross_attention=False: "soft_max" (learned-temperature log-mean-exp, default), "hard_max" (literal max, the original FILIP paper), or "mean" (literal unweighted average, no max/attention at all)
        filip_use_importance_weighting=False,  # weight the per-token pooling by a learned, independent (sigmoid) importance gate instead of a plain mean
        filip_cross_attention_weight_similarities=False,  # combine the raw cosine-similarity matrix with learned attention weights, instead of blending raw tokens then comparing
        use_precursor_mz_for_model=True,
        use_adduct=False,
        use_ce=False,
        use_ion_activation=False,
        use_ion_method=False,
        use_ion_mode=False,
    ):
        """Initialize the CCSPredictor"""
        super().__init__(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            weights=weights,
            lr=lr,
            use_element_wise=use_element_wise,
            use_cosine_distance=use_cosine_distance,
            use_adduct=use_adduct,
            use_ce=use_ce,
            use_ion_activation=use_ion_activation,
            use_ion_method=use_ion_method,
            use_ion_mode=use_ion_mode,
        )
        self.weights = weights
        self.mces_max_value = mces_max_value

        self.dropout = nn.Dropout(p=dropout)

        self.use_mces_bucket_head = use_mces_bucket_head
        if self.use_mces_bucket_head:
            bucket_edges_t = torch.tensor(
                list(mces_bucket_bin_edges), dtype=torch.float32
            )
            self.register_buffer("mces_bucket_bin_edges", bucket_edges_t)
            # +1 for the open-ended top bin, +1 for the singleton "exactly 0" bin
            self.mces_bucket_n_classes = len(mces_bucket_bin_edges) + 2
            self.mces_bucket_use_mlp = mces_bucket_use_mlp
            self.mces_bucket_loss_weight = mces_bucket_loss_weight
            bucket_input_dim = d_model
            if mces_bucket_use_mlp:
                self.mces_bucket_mlp = nn.Sequential(
                    nn.Linear(bucket_input_dim, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, d_model),
                )
                bucket_head_input_dim = d_model
            else:
                bucket_head_input_dim = bucket_input_dim
            self.mces_bucket_head = nn.Linear(
                bucket_head_input_dim, self.mces_bucket_n_classes - 1
            )

        self.use_contrastive_loss = use_contrastive_loss
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_loss_weight = contrastive_loss_weight
        self.contrastive_use_projection_head = contrastive_use_projection_head
        if self.use_contrastive_loss and self.contrastive_use_projection_head:
            self.contrastive_projection = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
        self.use_spectral_cosine_head = use_spectral_cosine_head
        self.spectral_cosine_loss_weight = spectral_cosine_loss_weight
        if self.use_spectral_cosine_head:
            self.spectral_cosine_projection = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )

        self.use_filip_head = use_filip_head
        self.filip_use_cross_attention = filip_use_cross_attention and use_filip_head
        self.filip_aggregation = filip_aggregation
        if self.use_filip_head and self.filip_aggregation == "soft_max":
            # Learnable temperature for the token-similarity soft-max
            # (log-sum-exp) aggregation, parameterized in log-space so it
            # stays positive; starts at temperature=exp(0)=1.0 (a mild
            # softening of the hard max). Only created for "soft_max";
            # "hard_max"/"mean" need no learned parameter for this step.
            self.filip_log_temperature = nn.Parameter(torch.zeros(1))
        if self.filip_use_cross_attention:
            # Learned cross-attention alternative to the fixed cosine +
            # max/log-mean-exp kernel: each spectrum's tokens attend over
            # the other spectrum's valid tokens with standard scaled-dot-
            # product attention, giving a learned, context-aware blend
            # instead of "pick your single best-matching peak." Only Q/K
            # are learned projections (used purely to derive attention
            # weights) -- the blended value and the final comparison both
            # stay in the raw token space, the same shared embedding space
            # cosine similarity is already used in everywhere else in this
            # model. (No dedicated V projection: blending a separately-
            # learned V and then comparing it to Q would require training
            # to also align two otherwise-unrelated projection spaces for
            # the cosine comparison to mean anything -- the path of least
            # resistance there is collapsing W_Q ~= W_V, which would waste
            # the point of a separate V.)
            self.filip_attn_q = nn.Linear(d_model, d_model)
            self.filip_attn_k = nn.Linear(d_model, d_model)
        self.filip_cross_attention_weight_similarities = (
            filip_cross_attention_weight_similarities and self.filip_use_cross_attention
        )
        self.filip_use_importance_weighting = (
            filip_use_importance_weighting and use_filip_head
        )
        if self.filip_use_importance_weighting:
            # Per-peak importance gate for pooling: sigmoid (independent
            # per token, no forced competition between peaks -- unlike
            # softmax, several peaks can all score highly important at
            # once), applied to the raw token so a distinctive/diagnostic
            # peak can matter more than a generic one, instead of every
            # peak contributing equally to the final averaged score.
            self.filip_importance_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
            )

        self.contrastive_use_filip_tokens = (
            self.use_contrastive_loss
            and contrastive_use_filip_tokens
            and use_filip_head
        )
        if self.contrastive_use_filip_tokens:
            # Dedicated per-token projection (applied independently to each
            # of the ~100 peak tokens) -- keeps the raw FILIP tokens that
            # drive the primary MCES regression free from the contrastive
            # loss's own discrimination pressure, same reasoning as
            # contrastive_projection/spectral_cosine_projection above.
            self.filip_contrastive_projection = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            # Mirror whichever token-interaction mechanism the primary FILIP
            # head uses (soft-max/hard_max/mean aggregation, cross-attention,
            # weight_similarities, importance weighting), but with fully
            # separate (dedicated) weights below -- so the contrastive
            # loss's own discrimination pressure can never warp the primary
            # head's own projections/gate.
            self.filip_contrastive_use_cross_attention = self.filip_use_cross_attention
            self.filip_contrastive_weight_similarities = (
                self.filip_cross_attention_weight_similarities
            )
            self.filip_contrastive_use_importance_weighting = (
                self.filip_use_importance_weighting
            )
            self.filip_contrastive_aggregation = self.filip_aggregation
            if self.filip_contrastive_use_cross_attention:
                self.filip_contrastive_attn_q = nn.Linear(d_model, d_model)
                self.filip_contrastive_attn_k = nn.Linear(d_model, d_model)
            elif self.filip_contrastive_aggregation == "soft_max":
                # Separate learned temperature for this projected token
                # space's own soft-max aggregation (independent of
                # filip_log_temperature, which governs the primary,
                # unprojected FILIP score). Only created for "soft_max";
                # "hard_max"/"mean" need no learned parameter for this
                # step, same as the primary head.
                self.filip_contrastive_log_temperature = nn.Parameter(torch.zeros(1))
            if self.filip_contrastive_use_importance_weighting:
                self.filip_contrastive_importance_mlp = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, 1),
                )

        self.spectral_cosine_use_filip_tokens = (
            self.use_spectral_cosine_head
            and spectral_cosine_use_filip_tokens
            and use_filip_head
        )
        if self.spectral_cosine_use_filip_tokens:
            # Dedicated per-token projection (applied independently to each
            # of the ~100 peak tokens) -- keeps the raw FILIP tokens that
            # drive the primary MCES regression free from this head's own
            # discrimination pressure, same reasoning as
            # filip_contrastive_projection above.
            self.spectral_cosine_filip_projection = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            # Mirror whichever token-interaction mechanism the primary FILIP
            # head uses, but with fully separate (dedicated) weights below --
            # same reasoning as filip_contrastive_use_cross_attention etc.
            self.spectral_cosine_filip_use_cross_attention = (
                self.filip_use_cross_attention
            )
            self.spectral_cosine_filip_weight_similarities = (
                self.filip_cross_attention_weight_similarities
            )
            self.spectral_cosine_filip_use_importance_weighting = (
                self.filip_use_importance_weighting
            )
            self.spectral_cosine_filip_aggregation = self.filip_aggregation
            if self.spectral_cosine_filip_use_cross_attention:
                self.spectral_cosine_filip_attn_q = nn.Linear(d_model, d_model)
                self.spectral_cosine_filip_attn_k = nn.Linear(d_model, d_model)
            elif self.spectral_cosine_filip_aggregation == "soft_max":
                self.spectral_cosine_filip_log_temperature = nn.Parameter(
                    torch.zeros(1)
                )
            if self.spectral_cosine_filip_use_importance_weighting:
                self.spectral_cosine_filip_importance_mlp = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, 1),
                )

        # No dedicated weights needed here (unlike the two heads above): the
        # bucket head never compares the two spectra's tokens against each
        # other, it only needs one plain, parameter-free pooled
        # representation per spectrum (see _filip_masked_mean_pool) fed into
        # the exact same abs-difference + optional MLP + head pipeline the
        # CLS-embedding version already used.
        self.mces_bucket_use_filip_tokens = (
            self.use_mces_bucket_head
            and mces_bucket_use_filip_tokens
            and use_filip_head
        )

        self.use_precursor_mz_for_model = use_precursor_mz_for_model

    def forward(self, batch, return_spectrum_output=False, return_tokens=False):
        # … compute raw emb0, emb1, apply relu, etc. …

        # nans to zeros
        batch["precursor_mass_0"] = torch.nan_to_num(
            batch["precursor_mass_0"], nan=0.0, posinf=0.0, neginf=0.0
        )
        batch["precursor_mass_1"] = torch.nan_to_num(
            batch["precursor_mass_1"], nan=0.0, posinf=0.0, neginf=0.0
        )
        batch["precursor_charge_0"] = torch.nan_to_num(
            batch["precursor_charge_0"], nan=0.0, posinf=0.0, neginf=0.0
        )
        batch["precursor_charge_1"] = torch.nan_to_num(
            batch["precursor_charge_1"], nan=0.0, posinf=0.0, neginf=0.0
        )

        """The inference pass"""
        if self.use_precursor_mz_for_model:
            mz_0 = batch["precursor_mass_0"].float()
            mz_1 = batch["precursor_mass_1"].float()
        else:
            mz_0 = torch.zeros_like(batch["precursor_mass_0"].float())
            mz_1 = torch.zeros_like(batch["precursor_mass_1"].float())
        kwargs_0 = {
            "precursor_mass": mz_0,
            "precursor_charge": batch["precursor_charge_0"].float(),
        }
        kwargs_1 = {
            "precursor_mass": mz_1,
            "precursor_charge": batch["precursor_charge_1"].float(),
        }

        if self.use_ion_mode:
            batch["ionmode_0"] = torch.nan_to_num(
                batch["ionmode_0"], nan=0.0, posinf=0.0, neginf=0.0
            )
            batch["ionmode_1"] = torch.nan_to_num(
                batch["ionmode_1"], nan=0.0, posinf=0.0, neginf=0.0
            )
            kwargs_0["ionmode"] = batch["ionmode_0"].float()
            kwargs_1["ionmode"] = batch["ionmode_1"].float()

        if self.use_adduct:
            batch["adduct_0"] = torch.nan_to_num(
                batch["adduct_0"], nan=0.0, posinf=0.0, neginf=0.0
            )
            batch["adduct_1"] = torch.nan_to_num(
                batch["adduct_1"], nan=0.0, posinf=0.0, neginf=0.0
            )
            kwargs_0["adduct"] = batch["adduct_0"].float()
            kwargs_1["adduct"] = batch["adduct_1"].float()

        if self.use_ce:
            batch["ce_0"] = torch.nan_to_num(
                batch["ce_0"], nan=0.0, posinf=0.0, neginf=0.0
            )
            batch["ce_1"] = torch.nan_to_num(
                batch["ce_1"], nan=0.0, posinf=0.0, neginf=0.0
            )
            kwargs_0["ce"] = batch["ce_0"].float()
            kwargs_1["ce"] = batch["ce_1"].float()

        if self.use_ion_activation:
            batch["ion_activation_0"] = torch.nan_to_num(
                batch["ion_activation_0"], nan=0.0, posinf=0.0, neginf=0.0
            )
            batch["ion_activation_1"] = torch.nan_to_num(
                batch["ion_activation_1"], nan=0.0, posinf=0.0, neginf=0.0
            )
            kwargs_0["ion_activation"] = batch["ion_activation_0"].float()
            kwargs_1["ion_activation"] = batch["ion_activation_1"].float()

        if self.use_ion_method:
            batch["ion_method_0"] = torch.nan_to_num(
                batch["ion_method_0"], nan=0.0, posinf=0.0, neginf=0.0
            )
            batch["ion_method_1"] = torch.nan_to_num(
                batch["ion_method_1"], nan=0.0, posinf=0.0, neginf=0.0
            )
            kwargs_0["ion_method"] = batch["ion_method_0"].float()
            kwargs_1["ion_method"] = batch["ion_method_1"].float()
        # intensity and mz
        batch["intensity_0"] = torch.nan_to_num(
            batch["intensity_0"], nan=0.0, posinf=0.0, neginf=0.0
        )
        batch["intensity_1"] = torch.nan_to_num(
            batch["intensity_1"], nan=0.0, posinf=0.0, neginf=0.0
        )
        batch["mz_0"] = torch.nan_to_num(batch["mz_0"], nan=0.0, posinf=0.0, neginf=0.0)
        batch["mz_1"] = torch.nan_to_num(batch["mz_1"], nan=0.0, posinf=0.0, neginf=0.0)

        latent0, pad_mask0 = self.spectrum_encoder(
            mz_array=batch["mz_0"].float(),
            intensity_array=batch["intensity_0"].float(),
            **kwargs_0,
        )
        latent1, pad_mask1 = self.spectrum_encoder(
            mz_array=batch["mz_1"].float(),
            intensity_array=batch["intensity_1"].float(),
            **kwargs_1,
        )

        emb0 = self.relu(latent0[:, 0, :])
        emb1 = self.relu(latent1[:, 0, :])

        filip_score = None
        tokens0 = tokens1 = valid0 = valid1 = None
        if self.use_filip_head:
            tokens0 = self.relu(latent0[:, 1:, :])
            tokens1 = self.relu(latent1[:, 1:, :])
            valid0 = ~pad_mask0[:, 1:]
            valid1 = ~pad_mask1[:, 1:]
            filip_score = self._filip_similarity(tokens0, valid0, tokens1, valid1)

        if return_spectrum_output:
            result = (
                *self.compute_from_embeddings(
                    emb0,
                    emb1,
                    filip_score=filip_score,
                    tokens0=tokens0,
                    valid0=valid0,
                    tokens1=tokens1,
                    valid1=valid1,
                ),
                emb0,
                emb1,
            )
        else:
            result = self.compute_from_embeddings(
                emb0,
                emb1,
                filip_score=filip_score,
                tokens0=tokens0,
                valid0=valid0,
                tokens1=tokens1,
                valid1=valid1,
            )

        if return_tokens:
            return (*result, tokens0, valid0, tokens1, valid1)
        return result

    def _filip_similarity(self, tokens0, valid0, tokens1, valid1):
        """Primary head's paired FILIP score -- see _filip_paired_score."""
        return _filip_paired_score(
            tokens0,
            valid0,
            tokens1,
            valid1,
            use_cross_attention=self.filip_use_cross_attention,
            weight_similarities=self.filip_cross_attention_weight_similarities,
            aggregation=self.filip_aggregation,
            use_importance_weighting=self.filip_use_importance_weighting,
            attn_q=getattr(self, "filip_attn_q", None),
            attn_k=getattr(self, "filip_attn_k", None),
            importance_mlp=getattr(self, "filip_importance_mlp", None),
            log_temperature=getattr(self, "filip_log_temperature", None),
        )

    def _filip_spectral_cosine_predict(self, tokens0, valid0, tokens1, valid1):
        """FILIP-token spectral-cosine head: tokens are first passed
        through the dedicated spectral_cosine_filip_projection (own weights,
        keeps this head's own discrimination pressure off the primary
        head's token space), then scored with the exact same paired FILIP
        mechanism as the primary head (mirroring its cross-attention/
        aggregation/importance-weighting configuration -- see
        spectral_cosine_use_filip_tokens), but via fully separate dedicated
        spectral_cosine_filip_* weights. Returns (B,)."""
        proj0 = self.spectral_cosine_filip_projection(tokens0)
        proj1 = self.spectral_cosine_filip_projection(tokens1)
        return _filip_paired_score(
            proj0,
            valid0,
            proj1,
            valid1,
            use_cross_attention=self.spectral_cosine_filip_use_cross_attention,
            weight_similarities=self.spectral_cosine_filip_weight_similarities,
            aggregation=self.spectral_cosine_filip_aggregation,
            use_importance_weighting=self.spectral_cosine_filip_use_importance_weighting,
            attn_q=getattr(self, "spectral_cosine_filip_attn_q", None),
            attn_k=getattr(self, "spectral_cosine_filip_attn_k", None),
            importance_mlp=getattr(self, "spectral_cosine_filip_importance_mlp", None),
            log_temperature=getattr(
                self, "spectral_cosine_filip_log_temperature", None
            ),
        )

    @staticmethod
    def _filip_similarity_matrix(
        tokens_a, valid_a, tokens_b, valid_b, aggregation, temperature=None
    ):
        """Same FILIP aggregation as _filip_soft_max_similarity /
        _filip_aggregate_over_j (aggregation: "soft_max" requires
        `temperature`, already clamped by the caller; "hard_max"/"mean"
        don't use it), but computes the full (P, Q) cross matrix between
        every sequence in `tokens_a` and every sequence in `tokens_b`,
        instead of a paired (B,) score -- used for the token-level in-batch
        contrastive loss, mirroring whichever aggregation the primary head
        uses. tokens_a: (P, N, D), valid_a: (P, N); tokens_b: (Q, M, D),
        valid_b: (Q, M). Returns (P, Q)."""
        ta = F.normalize(tokens_a, p=2, dim=-1)  # (P, N, D)
        tb = F.normalize(tokens_b, p=2, dim=-1)  # (Q, M, D)
        raw_sim = torch.einsum("pnd,qmd->pqnm", ta, tb)  # (P, Q, N, M)

        # direction a -> b: aggregate over b's valid tokens (already the
        # last axis), then average over a's valid tokens.
        mask_b = valid_b[None, :, None, :]  # (1, Q, 1, M)
        per_token_ab = _filip_aggregate_last_dim(
            raw_sim, mask_b, aggregation, temperature
        )  # (P, Q, N)
        mask_a_n = valid_a[:, None, :]  # (P, 1, N)
        per_token_ab = torch.where(
            mask_a_n, per_token_ab, torch.zeros_like(per_token_ab)
        )
        score_ab = per_token_ab.sum(dim=2) / valid_a.sum(dim=1).clamp(min=1)[:, None]

        # direction b -> a: transpose so a's tokens are again the last axis,
        # reusing the exact same aggregation logic.
        mask_a = valid_a[:, None, None, :]  # (P, 1, 1, N)
        per_token_ba = _filip_aggregate_last_dim(
            raw_sim.transpose(2, 3), mask_a, aggregation, temperature
        )  # (P, Q, M)
        mask_b_m = valid_b[None, :, :]  # (1, Q, M)
        per_token_ba = torch.where(
            mask_b_m, per_token_ba, torch.zeros_like(per_token_ba)
        )
        score_ba = per_token_ba.sum(dim=2) / valid_b.sum(dim=1).clamp(min=1)[None, :]

        return 0.5 * (score_ab + score_ba)

    def _filip_contrastive_cross_attention_matrix(
        self, tokens_a, valid_a, tokens_b, valid_b
    ):
        """Batched cross-batch generalization of
        _filip_cross_attention_similarity, used for the FILIP-scored in-batch
        contrastive loss when the primary head also uses cross-attention.
        Mirrors filip_cross_attention_weight_similarities /
        filip_use_importance_weighting via the dedicated
        filip_contrastive_weight_similarities /
        filip_contrastive_use_importance_weighting flags, using fully
        separate filip_contrastive_attn_q/k (and, if applicable,
        filip_contrastive_importance_mlp) weights -- same mechanism as the
        primary head, but this loss can never warp the primary head's own
        projections/gate. tokens_a: (P, N, D), valid_a: (P, N); tokens_b:
        (Q, M, D), valid_b: (Q, M). Returns (P, Q)."""
        d_k = tokens_a.shape[-1]
        scale = d_k**0.5
        neg_fill = -1e9

        def _direction(q_tokens, q_valid, kv_tokens, kv_valid):
            # q_tokens: (P, N, D), kv_tokens: (Q, M, D)
            q = self.filip_contrastive_attn_q(q_tokens)  # (P, N, D)
            k = self.filip_contrastive_attn_k(kv_tokens)  # (Q, M, D)
            logits = torch.einsum("pnd,qmd->pqnm", q, k) / scale  # (P, Q, N, M)
            logits = logits.masked_fill(~kv_valid[None, :, None, :], neg_fill)
            weights = torch.softmax(logits, dim=3)  # (P, Q, N, M)

            if self.filip_contrastive_weight_similarities:
                q_norm = F.normalize(q_tokens, p=2, dim=-1)
                kv_norm = F.normalize(kv_tokens, p=2, dim=-1)
                cos_sim = torch.einsum("pnd,qmd->pqnm", q_norm, kv_norm)  # (P, Q, N, M)
                per_token = (weights * cos_sim).sum(dim=3)  # (P, Q, N)
            else:
                attended = torch.einsum(
                    "pqnm,qmd->pqnd", weights, kv_tokens
                )  # (P, Q, N, D) -- raw values
                per_token = F.cosine_similarity(
                    attended, q_tokens[:, None, :, :], dim=-1
                )  # (P, Q, N) -- raw query

            per_token = torch.where(
                q_valid[:, None, :], per_token, torch.zeros_like(per_token)
            )

            if self.filip_contrastive_use_importance_weighting:
                importance = torch.sigmoid(
                    self.filip_contrastive_importance_mlp(q_tokens).squeeze(-1)
                )  # (P, N)
                importance = torch.where(
                    q_valid, importance, torch.zeros_like(importance)
                )
                numer = (importance[:, None, :] * per_token).sum(dim=2)  # (P, Q)
                denom = importance.sum(dim=1).clamp(min=1e-6)[:, None]  # (P, 1)
                return numer / denom
            n_valid = q_valid.sum(dim=1).clamp(min=1).float()[:, None]  # (P, 1)
            return per_token.sum(dim=2) / n_valid

        score_ab = _direction(tokens_a, valid_a, tokens_b, valid_b)  # (P, Q)
        score_ba = _direction(tokens_b, valid_b, tokens_a, valid_a).transpose(
            0, 1
        )  # (P, Q)
        return 0.5 * (score_ab + score_ba)

    def _filip_contrastive_loss_info_nce(
        self, tokens0, valid0, tokens1, valid1, mol_idx_0, mol_idx_1
    ):
        """In-batch InfoNCE, same recipe as _contrastive_loss_info_nce, but
        scored with the FILIP token-interaction matrix (via a dedicated
        per-token projection, and either the dedicated soft-max temperature
        or the dedicated cross-attention weights -- see
        filip_contrastive_use_cross_attention) instead of pooled-embedding
        cosine similarity. Returns (loss_or_None, n_pairs)."""
        is_self = mol_idx_0.view(-1) == mol_idx_1.view(-1)
        n_pairs = int(is_self.sum().item())
        if n_pairs < 2:
            return None, n_pairs

        proj0 = self.filip_contrastive_projection(tokens0[is_self])
        proj1 = self.filip_contrastive_projection(tokens1[is_self])
        valid0_self = valid0[is_self]
        valid1_self = valid1[is_self]

        if self.filip_contrastive_use_cross_attention:
            sim_matrix = self._filip_contrastive_cross_attention_matrix(
                proj0, valid0_self, proj1, valid1_self
            )  # (n_pairs, n_pairs)
        else:
            temperature = None
            if self.filip_contrastive_aggregation == "soft_max":
                temperature = self.filip_contrastive_log_temperature.exp().clamp(
                    min=1e-3, max=10.0
                )
            sim_matrix = self._filip_similarity_matrix(
                proj0,
                valid0_self,
                proj1,
                valid1_self,
                self.filip_contrastive_aggregation,
                temperature,
            )  # (n_pairs, n_pairs)

        logits = sim_matrix / self.contrastive_temperature
        labels = torch.arange(n_pairs, device=logits.device)
        loss = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )
        return loss, n_pairs

    def _forward_with_embeddings(self, batch):
        """(logits_list, emb0, emb1, tokens0, valid0, tokens1, valid1) --
        emb0/emb1 populated whenever an embedding-level auxiliary loss
        (contrastive, spectral-cosine) is enabled; tokens0/valid0/tokens1/
        valid1 populated when either contrastive_use_filip_tokens or
        spectral_cosine_use_filip_tokens is active. Unused entries are
        None."""
        need_cls = self.use_contrastive_loss or self.use_spectral_cosine_head
        need_tokens = (
            self.contrastive_use_filip_tokens or self.spectral_cosine_use_filip_tokens
        )
        if need_tokens:
            *logits_list, emb0, emb1, tokens0, valid0, tokens1, valid1 = self(
                batch, return_spectrum_output=True, return_tokens=True
            )
            return logits_list, emb0, emb1, tokens0, valid0, tokens1, valid1
        if need_cls:
            *logits_list, emb0, emb1 = self(batch, return_spectrum_output=True)
            return logits_list, emb0, emb1, None, None, None, None
        return self(batch), None, None, None, None, None, None

    def training_step(self, batch, batch_idx):
        logits_list, emb0, emb1, tokens0, valid0, tokens1, valid1 = (
            self._forward_with_embeddings(batch)
        )
        logits2 = logits_list[0]  # [B] similarity
        target2 = batch["mces"].to(dtype=torch.float32, device=self.device).view(-1)

        loss = self.step(
            batch,
            batch_idx,
            logits_list=logits_list,
            emb0=emb0,
            emb1=emb1,
            tokens0=tokens0,
            valid0=valid0,
            tokens1=tokens1,
            valid1=valid1,
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return {
            "loss": loss,
            "mces_pred": logits2.detach().view(-1).cpu(),
            "mces_target": target2.cpu(),
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """Validation step — returns loss + predictions for the MCES scatter plot."""
        logits_list, emb0, emb1, tokens0, valid0, tokens1, valid1 = (
            self._forward_with_embeddings(batch)
        )
        logits2 = logits_list[0]  # [B] similarity
        target2 = batch["mces"].to(dtype=torch.float32, device=self.device).view(-1)

        loss = self.step(
            batch,
            batch_idx,
            logits_list=logits_list,
            emb0=emb0,
            emb1=emb1,
            tokens0=tokens0,
            valid0=valid0,
            tokens1=tokens1,
            valid1=valid1,
        )
        self.log("validation_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # MCES MAE in raw MCES units: target2 = 1 - MCES/40, logits2 = predicted similarity
        mces_mae = (self.mces_max_value * (logits2.view(-1) - target2).abs()).mean()
        self.log("val_mces_mae", mces_mae, on_step=False, on_epoch=True, prog_bar=False)

        result = {
            "loss": loss,
            "mces_pred": logits2.view(-1).cpu(),
            "mces_target": target2.cpu(),
        }
        if self.use_mces_bucket_head:
            logits3 = logits_list[1]
            raw_mces_target = (1.0 - target2) * self.mces_max_value
            result["mces_bucket_pred"] = self._corn_decode_bin_generic(logits3).cpu()
            result["mces_bucket_target"] = self._mces_bucket_target_bins(
                raw_mces_target
            ).cpu()
        if self.use_spectral_cosine_head:
            pred_spectral_cosine = self._spectral_cosine_score(
                emb0, emb1, tokens0, valid0, tokens1, valid1
            )
            target_spectral_cosine = (
                batch["spectral_cosine"]
                .to(dtype=torch.float32, device=self.device)
                .view(-1)
            )
            result["spectral_cosine_pred"] = pred_spectral_cosine.view(-1).cpu()
            result["spectral_cosine_target"] = target_spectral_cosine.cpu()
        return result

    def step(
        self,
        batch,
        batch_idx,
        threshold=0.5,
        weight_loss2=None,
        logits_list=None,
        emb0=None,
        emb1=None,
        tokens0=None,
        valid0=None,
        tokens1=None,
        valid1=None,
    ):
        if logits_list is None:
            logits_list = self(batch)
        logits2 = logits_list[0]
        logits3 = logits_list[1] if self.use_mces_bucket_head else None
        target2 = batch["mces"].to(dtype=torch.float32, device=self.device)
        target2 = target2.view(-1)

        squared_diff = (logits2.view(-1, 1).float() - target2.view(-1, 1).float()) ** 2
        loss2 = squared_diff.view(-1, 1).mean()

        if self.use_mces_bucket_head:
            raw_mces_target = (1.0 - target2) * self.mces_max_value
            bucket_target_bins = self._mces_bucket_target_bins(raw_mces_target)
            loss3 = self._corn_loss_generic(
                logits3, bucket_target_bins, self.mces_bucket_n_classes
            )

        use_mces_bucket = self.use_mces_bucket_head
        self.log("loss_mces", loss2, on_step=True, on_epoch=True, prog_bar=False)
        if use_mces_bucket:
            self.log(
                "loss_mces_bucket", loss3, on_step=True, on_epoch=True, prog_bar=False
            )

        loss = loss2
        if use_mces_bucket:
            loss = loss + (self.mces_bucket_loss_weight * loss3)

        if self.use_contrastive_loss:
            if self.contrastive_use_filip_tokens:
                loss_contrastive, n_pairs = self._filip_contrastive_loss_info_nce(
                    tokens0,
                    valid0,
                    tokens1,
                    valid1,
                    batch["mol_idx_0"],
                    batch["mol_idx_1"],
                )
            else:
                loss_contrastive, n_pairs = self._contrastive_loss_info_nce(
                    emb0, emb1, batch["mol_idx_0"], batch["mol_idx_1"]
                )
            self.log(
                "contrastive_n_pairs",
                float(n_pairs),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )
            if loss_contrastive is not None:
                self.log(
                    "loss_contrastive",
                    loss_contrastive,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=False,
                )
                loss = loss + (self.contrastive_loss_weight * loss_contrastive)

        if self.use_spectral_cosine_head:
            pred_spectral_cosine = self._spectral_cosine_score(
                emb0, emb1, tokens0, valid0, tokens1, valid1
            )
            target_spectral_cosine = (
                batch["spectral_cosine"].to(dtype=torch.float32, device=self.device)
            ).view(-1)
            loss_spectral_cosine = F.mse_loss(
                pred_spectral_cosine.view(-1), target_spectral_cosine
            )
            self.log(
                "loss_spectral_cosine",
                loss_spectral_cosine,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
            )
            loss = loss + (self.spectral_cosine_loss_weight * loss_spectral_cosine)
        return loss

    def _spectral_cosine_score(self, emb0, emb1, tokens0, valid0, tokens1, valid1):
        """Dispatches the spectral-cosine head's prediction to either the
        FILIP-token mechanism (spectral_cosine_use_filip_tokens) or the
        original pooled-CLS-embedding cosine (_spectral_cosine_predict)."""
        if self.spectral_cosine_use_filip_tokens:
            return self._filip_spectral_cosine_predict(tokens0, valid0, tokens1, valid1)
        return self._spectral_cosine_predict(emb0, emb1)

    def _spectral_cosine_predict(self, emb0, emb1):
        """Predicted spectral-cosine score: cosine similarity of emb0/emb1
        after the dedicated spectral_cosine_projection, trained via MSE
        against the true binned spectral cosine (see
        simba/core/data/spectral_cosine.py)."""
        proj0 = self.spectral_cosine_projection(emb0)
        proj1 = self.spectral_cosine_projection(emb1)
        return self.cosine_similarity(proj0, proj1)

    def _contrastive_loss_info_nce(self, emb0, emb1, mol_idx_0, mol_idx_1):
        """In-batch InfoNCE over the batch's self-pairs (same molecule, two
        spectra): symmetric cross-entropy over the N x N cosine-similarity
        matrix of their embeddings, true match on the diagonal. Returns
        (loss_or_None, n_pairs) - loss is None when fewer than 2 self-pairs
        landed in this batch (can't form negatives)."""
        is_self = mol_idx_0.view(-1) == mol_idx_1.view(-1)
        n_pairs = int(is_self.sum().item())
        if n_pairs < 2:
            return None, n_pairs

        emb0_self = emb0[is_self]
        emb1_self = emb1[is_self]
        if self.contrastive_use_projection_head:
            emb0_self = self.contrastive_projection(emb0_self)
            emb1_self = self.contrastive_projection(emb1_self)
        anchor = F.normalize(emb0_self, p=2, dim=-1)
        positive = F.normalize(emb1_self, p=2, dim=-1)
        logits = (anchor @ positive.T) / self.contrastive_temperature
        labels = torch.arange(n_pairs, device=logits.device)
        loss = 0.5 * (
            F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
        )
        return loss, n_pairs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def compute_from_embeddings(
        self,
        emb0: torch.Tensor,
        emb1: torch.Tensor,
        filip_score: torch.Tensor = None,
        tokens0: torch.Tensor = None,
        valid0: torch.Tensor = None,
        tokens1: torch.Tensor = None,
        valid1: torch.Tensor = None,
    ):
        """
        Take two activated embeddings (after ReLU, fingerprint fusion, etc.)
        and run all the FC layers + similarity heads to produce the
        emb_sim_2 similarity score (and optional emb_sim_3 bucket logits).
        When use_filip_head is set, emb_sim_2 is the precomputed FILIP
        fine-grained token score (see _filip_similarity/forward) instead of
        the CLS-token cosine similarity. tokens0/valid0/tokens1/valid1 are
        only needed (and only used) when mces_bucket_use_filip_tokens is set.
        """
        if self.use_filip_head:
            emb_sim_2 = filip_score
        else:
            emb_sim_2 = self.cosine_similarity(emb0, emb1)

        if self.use_mces_bucket_head:
            if self.mces_bucket_use_filip_tokens:
                pooled0 = _filip_masked_mean_pool(tokens0, valid0)
                pooled1 = _filip_masked_mean_pool(tokens1, valid1)
                bucket_repr = torch.abs(pooled0 - pooled1)
            else:
                bucket_repr = torch.abs(emb0 - emb1)
            if self.mces_bucket_use_mlp:
                bucket_repr = self.mces_bucket_mlp(bucket_repr)
            emb_sim_3 = self.mces_bucket_head(
                bucket_repr
            )  # (B, mces_bucket_n_classes - 1) raw logits
            return (emb_sim_2, emb_sim_3)
        return (emb_sim_2,)

    @staticmethod
    def _corn_loss_generic(
        logits: torch.Tensor, target_bins: torch.Tensor, n_classes: int
    ) -> torch.Tensor:
        """CORN ordinal loss (Shi, Cao & Raschka, 2021/2023): for threshold j,
        only pairs whose true bin already exceeds j-1 contribute, with binary
        target 1{target_bin > j}."""
        total_loss = logits.new_tensor(0.0)
        total_count = logits.new_tensor(0.0)
        for j in range(n_classes - 1):
            mask = target_bins >= j
            if not mask.any():
                continue
            target_j = (target_bins[mask] > j).float()
            total_loss = total_loss + F.binary_cross_entropy_with_logits(
                logits[mask, j], target_j, reduction="sum"
            )
            total_count = total_count + mask.sum()
        return total_loss / total_count.clamp(min=1)

    @staticmethod
    def _corn_decode_bin_generic(logits: torch.Tensor) -> torch.Tensor:
        """Chain-rule decode (cumulative product of conditional probabilities)
        to a predicted ordinal bin index."""
        probas = torch.sigmoid(logits)
        cumprod = torch.cumprod(probas, dim=1)
        return (cumprod > 0.5).sum(dim=1)

    def _mces_bucket_target_bins(self, raw_mces: torch.Tensor) -> torch.Tensor:
        """Discretize raw MCES into bucket indices: 0 is its own class
        (self-pairs), then left-open/right-closed bins up to a final
        catch-all bin past the last edge."""
        raw = raw_mces.clamp(min=0)
        is_zero = raw == 0
        non_zero_bin = torch.bucketize(raw, self.mces_bucket_bin_edges, right=False)
        return torch.where(is_zero, torch.zeros_like(non_zero_bin), non_zero_bin + 1)


class EmbeddingExtractor(pl.LightningModule):
    def __init__(self, model_path, D_MODEL, N_LAYERS, multitasking=False, config=None):
        super().__init__()
        self.multitasking = multitasking
        self.config = config
        self.model = self.load_twin_network(
            model_path, D_MODEL, N_LAYERS
        ).spectrum_encoder
        self.relu = nn.ReLU()

    def load_twin_network(self, model_path, D_MODEL, N_LAYERS, strict=False):
        lr = self.config.optimizer.lr
        use_cosine_distance = (
            self.config.model.tasks.cosine_similarity.use_cosine_distance
        )

        if self.multitasking:
            return SimilarityModelMultitask.load_from_checkpoint(
                model_path,
                d_model=int(D_MODEL),
                n_layers=int(N_LAYERS),
                weights=None,
                lr=lr,
                use_cosine_distance=use_cosine_distance,
                strict=strict,
                use_adduct=self.config.model.features.use_adduct,
                use_ce=self.config.model.features.use_ce,
                use_ion_activation=self.config.model.features.use_ion_activation,
                use_ion_method=self.config.model.features.use_ion_method,
                use_ion_mode=self.config.model.features.use_ion_mode,
            )

        else:
            return SimilarityModel.load_from_checkpoint(
                model_path,
                d_model=int(D_MODEL),
                n_layers=int(N_LAYERS),
                weights=None,
                lr=lr,
                use_cosine_distance=use_cosine_distance,
                strict=strict,
                use_adduct=self.config.model.features.use_adduct,
                use_ce=self.config.model.features.use_ce,
                use_ion_activation=self.config.model.features.use_ion_activation,
                use_ion_method=self.config.model.features.use_ion_method,
                use_ion_mode=self.config.model.features.use_ion_mode,
            )

    def forward(self, batch):
        """The inference pass"""

        # extra data
        kwargs = {
            "precursor_mass": batch["precursor_mass"].float(),
            "precursor_charge": batch["precursor_charge"].float(),
        }

        # Add metadata fields if present in batch
        if "ionmode" in batch:
            kwargs["ionmode"] = batch["ionmode"].float()
        if "adduct" in batch:
            kwargs["adduct"] = batch["adduct"].float()
        if "ce" in batch:
            kwargs["ce"] = batch["ce"].float()
        if "ion_activation" in batch:
            kwargs["ion_activation"] = batch["ion_activation"].float()
        if "ion_method" in batch:
            kwargs["ion_method"] = batch["ion_method"].float()

        emb, _ = self.model(
            mz_array=batch["mz"].float(),
            intensity_array=batch["intensity"].float(),
            **kwargs,
        )

        emb = emb[:, 0, :]
        emb = self.relu(emb)

        return emb

    def get_embeddings(self, dataloader_spectrums, device="gpu"):
        predictor = pl.Trainer(
            max_epochs=0, enable_progress_bar=True, accelerator=device
        )
        embeddings = predictor.predict(
            self,
            dataloader_spectrums,
        )
        return self.flat_predictions(embeddings)

    def flat_predictions(self, preds):
        # flat the results
        concatenated_tensor = torch.cat(preds, dim=0)
        return concatenated_tensor.detach().numpy()
