# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import sys
import numpy as np

from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from .detr.util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding

from .ir_CSN_50 import build_CSN
from .ir_CSN_152 import build_CSN as build_CSN_152
from .detr.transformer_layers import LSTRTransformerDecoder, LSTRTransformerDecoderLayer, layer_norm


class Backbone(nn.Module):

    def __init__(self, train_backbone: bool, num_channels: int, position_embedding, return_interm_layers, args):
        super().__init__()

        if args.backbone == 'CSN-152':
            print("CSN-152 backbone")
            self.body = build_CSN_152(args)
        else:
            print("CSN-50 backbone")
            self.body = build_CSN(args)
        self.position_embedding = position_embedding
        for name, parameter in self.body.named_parameters():
            if not train_backbone:
                parameter.requires_grad_(False)
        self.ds = args.ava_single_frame
        if args.ava_single_frame:
            if args.ava_temporal_ds_strategy == 'avg':
                self.pool = nn.AvgPool3d((args.ava_temp_len // args.ava_ds_rate, 1, 1))
                # print("avg pool: {}".format(args.ava_temp_len // args.ava_ds_rate))
            elif args.ava_temporal_ds_strategy == 'max':
                self.pool = nn.MaxPool3d((args.ava_temp_len // args.ava_ds_rate, 1, 1))
                print("max pool: {}".format(args.ava_temp_len // args.ava_ds_rate))
            elif args.ava_temporal_ds_strategy == 'decode':
                self.query_pool = nn.Embedding(1, 2048)
                self.pool_decoder = LSTRTransformerDecoder(
                    LSTRTransformerDecoderLayer(d_model=2048, nhead=8, dim_feedforward=2048, dropout=0.1), 1,
                    norm=layer_norm(d_model=2048, condition=True))

        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            # return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(self.body, return_layers=return_layers)
        self.backbone_name = args.backbone
        self.temporal_ds_strategy = args.ava_temporal_ds_strategy

    def forward(self, tensor_list: NestedTensor):
        if "SlowFast" in self.backbone_name:
            xs, xt = self.body([tensor_list.tensors[:, :, ::4, ...], tensor_list.tensors])
            xs_orig = xt
        elif "TPN" in self.backbone_name:
            xs, xt = self.body(tensor_list.tensors)
            xs_orig = xt
        else:
            xs = self.body(tensor_list.tensors) #interm layer features
            # xs_orig = xs
        # if self.ds: xs = self.avg_pool(xs)
        # print(xs['0'].shape)
        # print(xs['1'].shape)
        # print(xs['2'].shape)
        # print(xs['3'].shape)
        # bs, ch, t, w, h = xs.shape
        # if self.ds:
        #     if self.temporal_ds_strategy == 'avg' or self.temporal_ds_strategy == 'max':
        #         xs = self.pool(xs)
        #     elif self.temporal_ds_strategy == 'decode':
        #         xs = xs.view(bs, ch, t, w * h).permute(2, 0, 3, 1).contiguous().view(t, bs * w * h, ch)
        #         query_embed = self.query_pool.weight.unsqueeze(1).repeat(1, bs * w * h, 1)
        #         xs = self.pool_decoder(query_embed, xs)
        #         xs = xs.view(1, bs, w * h, ch).permute(1, 3, 0, 2).contiguous().view(bs, ch, 1, w, h)
        #     else:
        #         xs = xs[:, :, t // 2: t // 2 + 1, ...]
        out: Dict[str, NestedTensor] = {}
        # m = tensor_list.mask
        # assert m is not None

        # mask = F.interpolate(m[None].float(), size=xs.shape[-2:]).to(torch.bool)[0]
        # mask = mask.unsqueeze(1).repeat(1,xs.shape[2],1,1)

        # out = [NestedTensor(xs, mask)]
        # pos = [self.position_embedding(NestedTensor(xs, mask))]

        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            mask = mask.unsqueeze(1).repeat(1,x.shape[2],1,1)
            # print("mask shape: ", mask.shape)
            bs, c, t, h, w = x.shape
            x = x.permute(2,0,1,3,4).reshape(t*bs, c, h, w)
            mask = mask.permute(1,0,2,3).reshape(t*bs, h, w)
            out[name] = NestedTensor(x, mask)
        return out #, pos #, xs_orig


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos #, xl


def build_3d_backbone(args):
    position_embedding = build_position_encoding(args)
    backbone = Backbone(train_backbone=args.ava_train_lr_backbone > 0, 
                     num_channels=args.ava_dim_feedforward, 
                     position_embedding=position_embedding, 
                     return_interm_layers=True,
                     args=args)
    model = Joiner(backbone, position_embedding)
    return model
