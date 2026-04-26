import torch
import torch.nn as nn


class ResidualConv1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, dropout=0.15):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.drop(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.act(out)
        return out


class AdditiveAttention(nn.Module):
    def __init__(self, in_dim, attn_dim=128):
        super().__init__()
        self.proj = nn.Linear(in_dim, attn_dim)
        self.score = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, x):
        e = torch.tanh(self.proj(x))
        a = torch.softmax(self.score(e), dim=1)
        context = torch.sum(a * x, dim=1)
        return context, a.squeeze(-1)


class CNNBiGRUAttentionNet(nn.Module):
    def __init__(self, in_channels=27, conv_channels=64, gru_hidden=64, gru_layers=2, dropout=0.25):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.conv_block1 = ResidualConv1D(64, 64, kernel_size=3, dropout=0.10)
        self.conv_block2 = ResidualConv1D(64, conv_channels, kernel_size=5, dropout=0.15)
        self.conv_drop = nn.Dropout(0.20)
        self.bigru = nn.GRU(
            input_size=conv_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.attn = AdditiveAttention(in_dim=gru_hidden * 2, attn_dim=256)
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv_stem(x)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_drop(x)
        x = x.transpose(1, 2)
        gru_out, _ = self.bigru(x)
        attn_vec, attn_weights = self.attn(gru_out)
        max_vec, _ = torch.max(gru_out, dim=1)
        feat = torch.cat([attn_vec, max_vec], dim=1)
        logits = self.fc(feat).squeeze(-1)
        return logits, attn_weights
