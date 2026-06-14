import torch
import torch.nn as nn


class CDLinearHead(nn.Module):
    """Channel-dependent linear map: all input channels x time -> all output channels x horizon."""

    def __init__(self, in_channels: int, out_channels: int, seq_len: int, pred_len: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.cross_linear = nn.Linear(
            in_channels * seq_len,
            out_channels * pred_len,
        )

    def forward(self, x_bcl: torch.Tensor) -> torch.Tensor:
        # x_bcl: [B, C_in, L]
        batch_size = x_bcl.size(0)
        out_flat = self.cross_linear(x_bcl.reshape(batch_size, -1))
        return out_flat.reshape(batch_size, self.out_channels, self.pred_len)


class Model(nn.Module):
    """
    Channel-Dependent Linear (CD-Linear).
    Flattens [C_in, seq_len] and maps to [C_out, pred_len] with one fully-connected layer.
    enc_in and c_out may differ (e.g. extra feature columns in, original targets out).
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name in {'classification', 'anomaly_detection', 'imputation'}:
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.in_channels = configs.enc_in
        self.out_channels = configs.c_out
        self.head = CDLinearHead(
            self.in_channels,
            self.out_channels,
            self.seq_len,
            self.pred_len,
        )

        if self.task_name == 'classification':
            self.projection = nn.Linear(
                configs.c_out * configs.seq_len, configs.num_class)

    def encoder(self, x):
        # x: [B, L, C_in]
        x_bcl = x.permute(0, 2, 1)
        y_bch = self.head(x_bcl)
        return y_bch.permute(0, 2, 1)

    def forecast(self, x_enc):
        return self.encoder(x_enc)

    def imputation(self, x_enc):
        return self.encoder(x_enc)

    def anomaly_detection(self, x_enc):
        return self.encoder(x_enc)

    def classification(self, x_enc):
        enc_out = self.encoder(x_enc)
        output = enc_out.reshape(enc_out.shape[0], -1)
        return self.projection(output)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in {'long_term_forecast', 'short_term_forecast'}:
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            return self.imputation(x_enc)
        if self.task_name == 'anomaly_detection':
            return self.anomaly_detection(x_enc)
        if self.task_name == 'classification':
            return self.classification(x_enc)
        return None
