# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Autoformer_EncDec import series_decomp


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x


class AttentionPatchAggregation(nn.Module):
    def __init__(self, N, P, S, te_dim, hid_dim, history, dropout_rate=0.1):
        super().__init__()
        self.N = N
        self.P = P
        self.S = max(history / P, 1e-6) if S is None else S
        self.history = history
        self.hid_dim = hid_dim
        self.te_dim = te_dim
        self.feature_dim = 1 + te_dim
        self.delta_left_params = nn.Parameter(torch.zeros(N, P))
        self.raw_log_width_params = nn.Parameter(torch.full((N, P), math.log(self.S)))
        self.tau_params = nn.Parameter(torch.zeros(N))
        self.projection_layer = nn.Linear(self.feature_dim, self.hid_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim * 2, hid_dim)
        )
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, t_stacked, x_with_te, mask_stacked):
        current_device = t_stacked.device
        B_N, L_obs_pad, _ = t_stacked.shape
        B = B_N // self.N
        patch_centers = torch.linspace(self.S / 2, self.history - self.S / 2, self.P, device=current_device)
        base_left_boundaries = (patch_centers - self.S / 2).unsqueeze(0)
        t_left_n_p = base_left_boundaries + self.delta_left_params
        width_learned_n_p = torch.exp(self.raw_log_width_params) + 1e-6
        t_right_n_p = t_left_n_p + width_learned_n_p
        current_variable_taus = F.softplus(self.tau_params).unsqueeze(-1) + 1e-6
        t_left_b_n = t_left_n_p.unsqueeze(0).expand(B, -1, -1).reshape(B_N, self.P).unsqueeze(-1)
        t_right_b_n = t_right_n_p.unsqueeze(0).expand(B, -1, -1).reshape(B_N, self.P).unsqueeze(-1)
        taus_b_n = current_variable_taus.unsqueeze(0).expand(B, -1, -1).reshape(B_N, 1).unsqueeze(-1)
        t_raw_b_n = t_stacked.transpose(-1, -2)
        weights_raw = torch.sigmoid((t_right_b_n - t_raw_b_n) / taus_b_n) * \
                      torch.sigmoid((t_raw_b_n - t_left_b_n) / taus_b_n)
        mask_b_n = mask_stacked.transpose(-1, -2)
        temporal_weights = weights_raw * mask_b_n
        sum_weights = temporal_weights.sum(dim=-1, keepdim=True) + 1e-9
        weighted_features_sum = torch.bmm(temporal_weights, x_with_te)
        h_patches_avg = weighted_features_sum / sum_weights
        h_patches_proj = self.projection_layer(h_patches_avg)
        h_patches = self.norm(h_patches_proj + self.ffn(h_patches_proj))
        return h_patches


class APNSeasonal(nn.Module):
    """APN for processing seasonal component"""
    def __init__(self, configs):
        super(APNSeasonal, self).__init__()
        self.configs = configs
        self.hid_dim = configs.d_model

        self.te_dim = getattr(configs, 'apn_te_dim', 32)
        self.N = configs.enc_in
        self.P = getattr(configs, 'apn_npatch', 16)

        self.dropout_rate = configs.dropout
        self.batch_size = None

        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, self.te_dim - 1)

        self.patching = AttentionPatchAggregation(
            N=self.N,
            P=self.P,
            S=None,
            te_dim=self.te_dim,
            hid_dim=self.hid_dim,
            history=1.0,
            dropout_rate=self.dropout_rate
        )

        self.patch_pos_enc = PositionalEncoding(self.hid_dim, max_len=self.P)
        self.var_queries = nn.Parameter(torch.randn(1, self.N, 1, self.hid_dim))
        self.aggregation_norm = nn.LayerNorm(self.hid_dim)

        self.decoder = nn.Sequential(
            nn.Linear(self.hid_dim + self.te_dim, self.hid_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hid_dim * 2, 1)
        )

    def LearnableTE(self, tt):
        out1 = self.te_scale(tt)
        out2 = torch.sin(self.te_periodic(tt))
        return torch.cat([out1, out2], -1)

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor, y_mark: torch.Tensor) -> torch.Tensor:
        B, L_obs, N_vars_from_X = x.shape
        self.batch_size = B

        time_features = x_mark[:, :, [0]]

        X_stacked = x.permute(0, 2, 1).reshape(B * N_vars_from_X, L_obs, 1)
        mask_stacked = torch.ones_like(X_stacked)

        time_features_stacked = time_features.repeat(1, 1, N_vars_from_X).permute(0, 2, 1).reshape(B * N_vars_from_X,
                                                                                                   L_obs, 1)

        te_his = self.LearnableTE(time_features_stacked)
        X_with_te = torch.cat([X_stacked, te_his], dim=-1)

        # APN processing
        h_patches_stacked = self.patching(time_features_stacked, X_with_te, mask_stacked)
        h_patches_stacked_pe = self.patch_pos_enc(h_patches_stacked)
        h_patches_updated = h_patches_stacked_pe.view(B, N_vars_from_X, self.P, self.hid_dim)
        attn_scores = torch.matmul(self.var_queries, h_patches_updated.transpose(-1, -2)) * (self.hid_dim ** -0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        h_final = torch.matmul(attn_weights, h_patches_updated)
        h_final = h_final.squeeze(-2)
        h_final = self.aggregation_norm(h_final)

        # Decode
        time_steps_to_predict = y_mark[:, :, [0]]
        L_pred = time_steps_to_predict.shape[1]
        h_expanded = h_final.unsqueeze(dim=-2).repeat(1, 1, L_pred, 1)
        time_steps_to_predict_exp = time_steps_to_predict.view(B, 1, L_pred, 1).repeat(1, N_vars_from_X, 1, 1)
        te_pred = self.LearnableTE(time_steps_to_predict_exp)
        decoder_input = torch.cat([h_expanded, te_pred], dim=-1)
        outputs_raw = self.decoder(decoder_input)
        outputs = outputs_raw.squeeze(-1).permute(0, 2, 1)
        return outputs


class TrendPredictor(nn.Module):
    """Simple trend predictor using linear projection"""
    def __init__(self, enc_in, pred_len):
        super(TrendPredictor, self).__init__()
        self.pred_len = pred_len
        self.enc_in = enc_in
        # Project trend from input length to output length
        self.trend_proj = nn.Linear(enc_in, pred_len)

    def forward(self, trend_enc, trend_dec=None):
        """
        trend_enc: [B, L_enc, N_vars] - trend component from decomposition
        trend_dec: not used, kept for compatibility
        """
        B, L_enc, N_vars = trend_enc.shape
        # Global average pooling for trend: [B, N_vars]
        trend_global = trend_enc.mean(dim=1)
        # Project to prediction length: [B, pred_len]
        trend_pred = self.trend_proj(trend_global)
        # Expand to [B, pred_len, N_vars]
        trend_pred = trend_pred.unsqueeze(-1).expand(-1, -1, N_vars)
        return trend_pred


class Model(nn.Module):
    """
    APN with Series Decomposition
    Decomposes time series into seasonal and trend components, processes each with specialized modules.

    Paper: Rethinking Irregular Multivariate Time Series Forecasting (AAAI 2026)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len

        # Decomposition module
        self.decomposition = series_decomp(kernel_size=configs.moving_avg)

        # Seasonal component processor
        self.seasonal_model = APNSeasonal(configs)

        # Trend component processor
        self.trend_model = TrendPredictor(configs.enc_in, configs.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None):
        """
        Parameters:
        -----------
        x_enc: [B, L_enc, N_vars] - encoder input (historical observations)
        x_mark_enc: [B, L_enc, N_time_features] - encoder time features
        x_dec: not used (kept for compatibility)
        x_mark_dec: [B, L_label+pred, N_time_features] - may contain label_len + pred_len time steps

        Returns:
        --------
        outputs: [B, L_pred, N_vars] - predictions
        """
        B, L_enc, _ = x_enc.shape

        # Handle x_mark_dec which may contain label_len + pred_len time steps
        if x_mark_dec is not None:
            # Use only the last pred_len time steps from x_mark_dec
            x_mark_dec = x_mark_dec[:, -self.pred_len:, :]

        # Create prediction timestamps if not provided
        if x_mark_dec is None or x_mark_dec.size(1) != self.pred_len:
            L_pred = self.pred_len
            # Normalized timestamps continuing after input
            start_time = torch.linspace(L_enc / (L_enc + L_pred), 1 - 1 / (L_enc + L_pred),
                                        L_enc, device=x_enc.device)
            end_time = torch.linspace(1 - 1 / (L_enc + L_pred) + 1 / (L_enc + L_pred),
                                      1 - 1 / (L_enc + L_pred) / L_pred,
                                      L_pred, device=x_enc.device)
            x_mark_dec = torch.cat([
                start_time.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1),
                end_time.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)
            ], dim=1)

        # Decompose input into seasonal and trend
        seasonal_enc, trend_enc = self.decomposition(x_enc)

        # Process seasonal component with APN
        seasonal_pred = self.seasonal_model(seasonal_enc, x_mark_enc, x_mark_dec)

        # Process trend component with simple projection
        trend_pred = self.trend_model(trend_enc, x_mark_dec)

        # Combine seasonal and trend predictions
        outputs = seasonal_pred + trend_pred

        return outputs
