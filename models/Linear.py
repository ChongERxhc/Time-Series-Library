import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    A single linear layer mapping seq_len -> pred_len per channel (LTSF-Linear).
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        if self.task_name in {'classification', 'anomaly_detection', 'imputation'}:
            self.pred_len = configs.seq_len
        else:
            self.pred_len = configs.pred_len

        self.channels = configs.enc_in
        self.individual = getattr(configs, 'individual', False)

        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
            self.Linear.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        if self.task_name == 'classification':
            self.projection = nn.Linear(
                configs.enc_in * configs.seq_len, configs.num_class)

    def encoder(self, x):
        # x: [B, L, C]
        if self.individual:
            output = torch.zeros(
                [x.size(0), self.pred_len, x.size(2)],
                dtype=x.dtype,
                device=x.device,
            )
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            return output
        return self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)

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
