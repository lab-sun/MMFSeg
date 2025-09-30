import numpy as np
import torch.nn as nn
import torch

from torch.nn.modules import module
import torch.nn.functional as F


class MLP(nn.Module):
    """
    Linear Embedding: 
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class DecoderHead(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):

        super(DecoderHead, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners

        self.in_channels = in_channels

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.local1 = nn.Conv2d(in_channels=embedding_dim,out_channels=2,kernel_size=1,stride=1)
        self.local2 = nn.Conv2d(in_channels=embedding_dim,out_channels=2,kernel_size=1,stride=1)
        self.local3 = nn.Conv2d(in_channels=embedding_dim,out_channels=2,kernel_size=1,stride=1)
        self.local4 = nn.Conv2d(in_channels=embedding_dim,out_channels=2,kernel_size=1,stride=1)

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim * 4+4, out_channels=embedding_dim, kernel_size=1),
            norm_layer(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)



    def forward(self, inputs):
        # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        c_locals = []

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)
        c4_local = self.local4(_c4)
        c_locals.append(c4_local)
        c4_local_softmax = F.softmax(c4_local,dim=1)[:,1:,:,:]
        _c4 = torch.cat((_c4,c4_local_softmax),dim=1)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)
        c3_local = self.local3(_c3)
        c_locals.append(c3_local)
        c3_local_softmax = F.softmax(c3_local,dim=1)[:,1:,:,:]
        _c3 = torch.cat((_c3,c3_local_softmax),dim=1)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)
        c2_local = self.local2(_c2)
        c_locals.append(c2_local)
        c2_local_softmax = F.softmax(c2_local,dim=1)[:,1:,:,:]
        _c2 = torch.cat((_c2,c2_local_softmax),dim=1)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        c1_local = self.local1(_c1)
        c_locals.append(c1_local)
        c1_local_softmax = F.softmax(c1_local,dim=1)[:,1:,:,:]
        _c1 = torch.cat((_c1,c1_local_softmax),dim=1)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x,c_locals

