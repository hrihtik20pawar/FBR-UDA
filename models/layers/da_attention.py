import torch
import torch.nn as nn


class _PositionAttentionModule(nn.Module):
    """ Position attention module """
    def __init__(self, in_channels):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, h, w = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, h * w).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, h * w)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, h * w)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, h, w)
        out = self.alpha * feat_e + x
        return out


class _ChannelAttentionModule(nn.Module):
    """ Channel attention module """
    def __init__(self, in_channels):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, h, w = x.size()
        feat_a = x.view(batch_size, -1, h * w)
        feat_a_transpose = feat_a.permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention)
        attention = self.softmax(attention_new)
        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, h, w)
        out = self.beta * feat_e + x
        return out
