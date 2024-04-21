import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.utils.checkpoint
from Layers import *
from einops.layers.torch import Rearrange
from einops import rearrange
from torchsummary import summary


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position, train_shape):
        super(PositionalEncoding, self).__init__()
        self.n_pos_sqrt = int(np.sqrt(n_position))
        self.train_shape = train_shape
        self.register_buffer('hashIndex', self._get_hash_table(n_position))
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))
        self.register_buffer('pos_table_train', self._get_sinusoid_encoding_table_train(n_position, train_shape))

    def _get_hash_table(self, n_position):
        return rearrange(torch.arange(n_position), '(h w) -> h w', h=int(np.sqrt(n_position)), w=int(np.sqrt(n_position)))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table[None, :])

    def _get_sinusoid_encoding_table_train(self, train_shape):
        selectIndex = rearrange(self.hashIndex[:train_shape[0], :train_shape[1]], 'h w -> (h w)')
        return torch.index_select(self.pos_table, dim=1, index=selectIndex)

    def forward(self, x):
        selectIndex = rearrange(self.hashIndex[:self.train_shape[0], :self.train_shape[1]], 'h w -> (h w)')
        return x + torch.index_select(self.pos_table, dim=1, index=selectIndex)


class Encoder(nn.Module):
    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, dropout, n_position, train_shape):
        super().__init__()
        self.lp = nn.Conv2d(3, 1, 1)
        self.new_reorder = Rearrange('b c h w -> b h (c w)')
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        )
        self.stage2 = self._make_layer(3, 16, 24, stride=2)
        self.stage3 = self._make_layer(4, 24, 40, stride=2)
        self.stage4 = self._make_layer(6, 40, 80, stride=2)
        self.stage5 = self._make_layer(3, 80, d_model, stride=1)
        self.reorder = Rearrange('b c h w -> b (h w) c')
        self.pos_enc = PositionalEncoding(d_model, n_position=n_position, train_shape=train_shape)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_heads, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.shape16 = Rearrange('b (h w) d -> b d h w', h=train_shape[0])
        self.inception = InceptionConv(d_model, d_model)
        self.invert_res = self._make_layer(1, d_model, d_model, stride=1)
        self.d1 = Deconv(256, 80, 1)
        self.d2 = Deconv(80, 40, 2)
        self.d3 = Deconv(40, 24, 2)

    def forward(self, input_map):
        stage1_o = self.stage1(input_map)
        stage2_o = self.stage2(stage1_o)
        stage3_o = self.stage3(stage2_o)
        stage4_o = self.stage4(stage3_o)
        stage5_o = self.stage5(stage4_o)
        incep = self.inception(stage5_o)

        ntr = self.layer_norm(self.dropout(self.pos_enc(self.new_reorder(self.lp(input_map)))))
        for enc_layer in self.layer_stack:
            ntr = enc_layer(ntr, slf_attn_mask=None)
        ntr = self.invert_res(self.shape16(ntr))

        stage2_o = stage2_o + self.d3(self.d2(self.d1(ntr)))
        stage3_o = stage3_o + self.d2(self.d1(ntr))
        stage4_o = stage4_o + self.d1(ntr)
        stage5_o = incep + ntr
        return stage1_o, stage2_o, stage3_o, stage4_o, stage5_o

    def _make_layer(self, n, inp, oup, stride):
        layers = []
        layers.append(block(inp, oup, stride))
        for i in range(n - 1):
            layers.append(block(oup, oup, stride=1))
        return nn.Sequential(*layers)


class Decoder_prom(nn.Module):
    def __init__(self, d_model):
        super(Decoder_prom, self).__init__()
        self.dec1 = Deconv(d_model + 80, 128, 2)
        self.dec2 = Deconv(128 + 40, 64, 2)
        self.dec3 = Deconv(64 + 24, 64, 2)
        self.dec4 = Deconv(64 + 16, 64, 2)
        self.final = nn.Sequential(
            Conv_Block(64, 2),
            nn.Conv2d(2, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, stage1_o, stage2_o, stage3_o, stage4_o, stage5_o):
        x0 = self.dec1(torch.concat([stage4_o, stage5_o], dim=1))
        x0 = self.dec2(torch.concat([stage3_o, x0], dim=1))
        x0 = self.dec3(torch.concat([stage2_o, x0], dim=1))
        x0 = self.dec4(torch.concat([stage1_o, x0], dim=1))
        oup = self.final(x0)
        return oup


class MNetDecoder_line(nn.Module):
    def __init__(self, d_model):
        super(MNetDecoder_line, self).__init__()
        final_c = (d_model + 321)
        self.dec1 = nn.Sequential(
            Deconv(d_model + 80, 128, 2),
            Conv_Block(128, 128),
        )
        self.dec2 = nn.Sequential(
            Deconv(128 + 40, 64, 2),
            Conv_Block(64, 64),
        )
        self.dec3 = nn.Sequential(
            Deconv(64 + 24, 64, 2),
            Conv_Block(64, 64),
        )
        self.dec4 = nn.Sequential(
            Deconv(64 + 16, 64, 2),
            Conv_Block(64, 64),
        )
        self.final = nn.Sequential(
            Conv_Block(final_c, 64),
            Conv_Block(64, 2),
            nn.Conv2d(2, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, stage1_o, stage2_o, stage3_o, stage4_o, stage5_o, prom):
        x0 = self.dec1(torch.concat([stage4_o, stage5_o], dim=1))
        x1 = self.dec2(torch.concat([stage3_o, x0], dim=1))
        x2 = self.dec3(torch.concat([stage2_o, x1], dim=1))
        x3 = self.dec4(torch.concat([stage1_o, x2], dim=1))
        r5 = self.up(self.up(self.up(self.up(stage5_o))))
        r4 = self.up(self.up(self.up(x0)))
        r3 = self.up(self.up(x1))
        r2 = self.up(x2)
        oup = self.final(torch.cat((r5, r4, r3, r2, x3, prom), dim=1))

        return oup


class Regress_cost_new(nn.Module):
    def __init__(self, inp):
        super(Regress_cost_new, self).__init__()
        self.down = nn.Sequential(
            nn.BatchNorm2d(inp),
            nn.Conv2d(inp, 512, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(512, 768, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(768, 1024, 3, 1, 1, padding_mode='zeros', bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.MaxPool2d(2, 2, 0),
        )
        self.fc1 = nn.Linear(2 * 2 * 1024, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1)
        self.fcl = nn.Sequential(
            nn.ReLU(),
            nn.LayerNorm(1024, eps=1e-6),
        )

    def forward(self, x):
        x = self.down(x)
        x = self.fc3(self.fcl(self.fc1(torch.flatten(x, 1, 3))))
        return x


class Transformer(nn.Module):
    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, pad_idx, dropout, n_position, train_shape):
        super().__init__()
        self.encoder = Encoder(n_layers=n_layers, n_heads=n_heads, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout, n_position=n_position, train_shape=train_shape)
        self.decoder_prom = Decoder_prom(d_model)
        self.decoder_line = MNetDecoder_line(d_model)
        self.decoder_cost = Regress_cost_new(d_model)

    def forward(self, input_map):
        stage1_o, stage2_o, stage3_o, stage4_o, stage5_o = self.encoder(input_map)
        prom = self.decoder_prom(stage1_o, stage2_o, stage3_o, stage4_o, stage5_o)
        line = self.decoder_line(stage1_o, stage2_o, stage3_o, stage4_o, stage5_o, prom)
        cost = self.decoder_cost(stage5_o)
        return prom, line, cost


class Transformer1(nn.Module):
    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, pad_idx, dropout, n_position, train_shape):
        super().__init__()

        self.encoder = Encoder(n_layers=n_layers, n_heads=n_heads, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout, n_position=n_position, train_shape=train_shape)
        self.decoder_prom = Decoder_prom(d_model)
        self.decoder_line = MNetDecoder_line(d_model)
        self.decoder_cost = Regress_cost_new(d_model)
        for p in self.decoder_line.parameters():
            p.requires_grad = False
        for p in self.decoder_cost.parameters():
            p.requires_grad = False

    def forward(self, input_map):
        stage1_o, stage2_o, stage3_o, stage4_o, stage5_o = self.encoder(input_map)
        prom = self.decoder_prom(stage1_o, stage2_o, stage3_o, stage4_o, stage5_o)
        line = self.decoder_line(stage1_o, stage2_o, stage3_o, stage4_o, stage5_o, prom)
        cost = self.decoder_cost(stage5_o)
        return prom, line, cost


class Transformer2(nn.Module):
    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, pad_idx, dropout, n_position, train_shape):
        super().__init__()

        self.encoder = Encoder(n_layers=n_layers, n_heads=n_heads, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout, n_position=n_position, train_shape=train_shape)
        self.decoder_prom = Decoder_prom(d_model)
        self.decoder_line = MNetDecoder_line(d_model)
        self.decoder_cost = Regress_cost_new(d_model)
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.decoder_prom.parameters():
            p.requires_grad = False
        for p in self.decoder_cost.parameters():
            p.requires_grad = False

    def forward(self, input_map):
        stage1_o, stage2_o, stage3_o, stage4_o, stage5_o = self.encoder(input_map)
        prom = self.decoder_prom(stage1_o, stage2_o, stage3_o, stage4_o, stage5_o)
        line = self.decoder_line(stage1_o, stage2_o, stage3_o, stage4_o, stage5_o, prom)
        cost = self.decoder_cost(stage5_o)
        return prom, line, cost


class Transformer3(nn.Module):
    def __init__(self, n_layers, n_heads, d_k, d_v, d_model, d_inner, pad_idx, dropout, n_position, train_shape):
        super().__init__()

        self.encoder = Encoder(n_layers=n_layers, n_heads=n_heads, d_k=d_k, d_v=d_v, d_model=d_model, d_inner=d_inner, dropout=dropout, n_position=n_position, train_shape=train_shape)
        self.decoder_prom = Decoder_prom(d_model)
        self.decoder_line = MNetDecoder_line(d_model)
        for p in self.parameters():
            p.requires_grad = False
        self.decoder_cost = Regress_cost_new(d_model)

    def forward(self, input_map):
        stage1_o, stage2_o, stage3_o, stage4_o, stage5_o = self.encoder(input_map)
        prom = self.decoder_prom(stage1_o, stage2_o, stage3_o, stage4_o, stage5_o)
        line = self.decoder_line(stage1_o, stage2_o, stage3_o, stage4_o, stage5_o, prom)
        cost = self.decoder_cost(stage5_o)
        return prom, line, cost


if __name__ == '__main__':
    model_args = dict(
        n_layers=10,
        n_heads=8,
        d_k=256,
        d_v=128,
        d_model=256,
        d_inner=512,
        pad_idx=None,
        n_position=16 * 16,
        dropout=0.2,
        train_shape=[16, 16],
    )
    model = Transformer1(**model_args).to('cuda')
    x = torch.rand([2, 3, 256, 256]).to('cuda')
    prom, line, cost = model(x)
    print(f'prom : {prom.shape}')
    print(f'line : {line.shape}')
    print(f'cost : {cost.shape}')

    summary(model, (3, 256, 256))
