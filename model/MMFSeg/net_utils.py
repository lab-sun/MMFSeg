import torch
import torch.nn as nn

from timm.models.layers import trunc_normal_
import math

class photometric_encoder(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(photometric_encoder, self).__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.residual = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True),
                        nn.Conv2d(out_channels//reduction, out_channels//reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels//reduction),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels//reduction, out_channels, kernel_size=1, bias=True),
                        norm_layer(out_channels) 
                        )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True),
                        nn.Conv2d(out_channels//reduction, out_channels//reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels//reduction),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels//reduction, out_channels, kernel_size=1, bias=True),
                        norm_layer(out_channels) 
                        )
        self.norm = norm_layer(out_channels)
    def forward(self, y, x1, x2):
        residualx1 = self.residual(x1)
        x11 = torch.cat((y, x1), dim=1)
        x11 = self.conv1(x11)
        out1 = self.norm(residualx1 + x11)
        residualx2 = self.residual(x2)
        yy = 1 - y
        x22 = torch.cat((yy, x2), dim=1)
        x22 = self.conv2(x22)
        out2 = self.norm(residualx2 + x22)

        return out1, out2


class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim * 4, self.dim * 4 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim * 4 // reduction, self.dim * 2), 
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        avg = self.avg_pool(x).view(B, self.dim * 2)
        max = self.max_pool(x).view(B, self.dim * 2)
        y = torch.cat((avg, max), dim=1) # B 4C
        y = self.mlp(y).view(B, self.dim * 2, 1)
        channel_weights = y.reshape(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4) # 2 B C 1 1
        return channel_weights


class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
                    nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dim // reduction, 2, kernel_size=1), 
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1) # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4) # 2 B 1 H W
        return spatial_weights


class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FeatureRectifyModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x1, x2):
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2 


class IlluminationGuidedFeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(IlluminationGuidedFeatureRectifyModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
        self.photometric_encoder = photometric_encoder(in_channels=dim * 2, out_channels=dim, reduction=reduction)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, y, x1, x2):
        x11, x22 = self.photometric_encoder(y, x1, x2)
        channel_weights = self.channel_weights(x11, x22)
        spatial_weights = self.spatial_weights(x11, x22)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        return out_x1, out_x2 


# Stage 1
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        print("q1 shape: ", q1.shape)
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        print("k1 shape: ", k1.shape)
        print("v1 shape: ", v1.shape)
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 

        return x1, x2


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        print("u1.shape: ",u1.shape)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2


# Stage 2
class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True),
                        nn.Conv2d(out_channels//reduction, out_channels//reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels//reduction),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels//reduction, out_channels, kernel_size=1, bias=True),
                        norm_layer(out_channels) 
                        )
        self.norm = norm_layer(out_channels)
        
    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim*2, out_channels=dim, reduction=reduction, norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        print("x1.shape: ",x1.shape)
        x2 = x2.flatten(2).transpose(1, 2)
        print("x2.shape: ",x2.shape)
        x1, x2 = self.cross(x1, x2) 
        print("x1.shape: ",x1.shape)
        print("x2.shape: ",x2.shape)
        merge = torch.cat((x1, x2), dim=-1)
        print("merge.shape: ",merge.shape)
        merge = self.channel_emb(merge, H, W)
        
        return merge
    

class FeatureAlignmentModule(nn.Module):
    def __init__(self, f1_channel = 96,f2_channel = 32, f1_2_f2 = True):
        super().__init__()
        self.f1_2_f2 = f1_2_f2
        self.f1_channel = f1_channel
        self.f2_channel = f2_channel
        self.conv1 = nn.Conv2d(in_channels=f1_channel+f2_channel+4, out_channels=(f1_channel+f2_channel+4)//4,kernel_size=3,stride=1,padding=1)
        self.conv1_relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=(f1_channel+f2_channel+4)//4,out_channels=2,kernel_size=3,stride=1,padding=1)
        self.conv2_sigmoid = nn.Sigmoid()

        self.transfer_f1_2_f2 = nn.Conv2d(in_channels=f1_channel,out_channels=f2_channel,kernel_size=1)
        self.transfer_f2_2_f1 = nn.Conv2d(in_channels=f2_channel,out_channels=f1_channel,kernel_size=1)

        self.f1_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.f1_max_pool = nn.AdaptiveAvgPool2d(1)
        self.f2_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.f2_max_pool = nn.AdaptiveAvgPool2d(1)

        self.transfer_f1_2_f2_fc1 = nn.Linear(in_features=f2_channel*4,out_features=f2_channel)
        self.transfer_f1_2_f2_bn1 = nn.BatchNorm1d(f2_channel)
        self.transfer_f1_2_f2_relu1 = nn.ReLU()
        self.transfer_f1_2_f2_fc2 = nn.Linear(in_features=f2_channel,out_features=f2_channel*2)
        self.transfer_f1_2_f2_bn2 = nn.BatchNorm1d(f2_channel*2)
        self.transfer_f1_2_f2_relu2 = nn.Sigmoid()

        self.transfer_f2_2_f1_fc1 = nn.Linear(in_features=f1_channel*4,out_features=f1_channel)
        self.transfer_f2_2_f1_bn1 = nn.BatchNorm1d(f1_channel)
        self.transfer_f2_2_f1_relu1 = nn.ReLU()
        self.transfer_f2_2_f1_fc2 = nn.Linear(in_features=f1_channel,out_features=f1_channel*2)
        self.transfer_f2_2_f1_bn2 = nn.BatchNorm1d(f1_channel*2)
        self.transfer_f2_2_f1_relu2 = nn.Sigmoid()
        if self.f1_2_f2:
            dim = f2_channel
        else:
            dim = f1_channel

    def forward(self, f1, f2):

        f1_channel_max = torch.max(f1,dim=1)[0]
        f1_channel_max = f1_channel_max.unsqueeze(1)
        f1_channel_mean = torch.mean(f1,dim=1)
        f1_channel_mean = f1_channel_mean.unsqueeze(1)
        f1_new = torch.cat((f1,f1_channel_max,f1_channel_mean),dim=1)

        f2_channel_max = torch.max(f2,dim=1)[0]
        f2_channel_max = f2_channel_max.unsqueeze(1)
        f2_channel_mean = torch.mean(f2,dim=1)
        f2_channel_mean = f2_channel_mean.unsqueeze(1)
        f2_new = torch.cat((f2,f2_channel_max,f2_channel_mean),dim=1)
        
        f1_f2_cat = torch.cat((f1_new,f2_new),dim=1)

        f1f2 = self.conv1(f1_f2_cat)
        f1f2 = self.conv1_relu(f1f2)
        f1f2_spatial = self.conv2(f1f2)
        f1f2_spatial = self.conv2_sigmoid(f1f2_spatial)

        f1_spatial_weighted = f1 * f1f2_spatial[:,:1,:,:]
        f2_spatial_weighted = f2 * f1f2_spatial[:,1:,:,:]

        if self.f1_2_f2:
            f1_transferred = self.transfer_f1_2_f2(f1_spatial_weighted)
            f2_transferred = f2_spatial_weighted
        else:
            f1_transferred = f1_spatial_weighted
            f2_transferred = self.transfer_f2_2_f1(f2_spatial_weighted)


        f1_avg = self.f1_avg_pool(f1_transferred)
        f1_max = self.f1_max_pool(f1_transferred)

        f2_avg = self.f2_avg_pool(f2_transferred)
        f2_max = self.f2_max_pool(f2_transferred)

        channel_weight = torch.cat((f1_avg,f1_max,f2_avg,f2_max),dim=1)
        B,C,W,H = channel_weight.size()
        channel_weight = channel_weight.view(B,-1)
        if self.f1_2_f2:
            channel_weight = self.transfer_f1_2_f2_fc1(channel_weight)
            channel_weight = self.transfer_f1_2_f2_bn1(channel_weight)
            channel_weight = self.transfer_f1_2_f2_relu1(channel_weight)
            channel_weight = self.transfer_f1_2_f2_fc2(channel_weight)
            channel_weight = self.transfer_f1_2_f2_bn2(channel_weight)
            channel_weight = self.transfer_f1_2_f2_relu2(channel_weight)
            channel_weight_f1 = channel_weight[:,:self.f2_channel].view(B,-1,1,1)
            channel_weight_f2 = channel_weight[:,self.f2_channel:].view(B,-1,1,1)
        else:
            channel_weight = self.transfer_f2_2_f1_fc1(channel_weight)
            channel_weight = self.transfer_f2_2_f1_bn1(channel_weight)
            channel_weight = self.transfer_f2_2_f1_relu1(channel_weight)
            channel_weight = self.transfer_f2_2_f1_fc2(channel_weight)
            channel_weight = self.transfer_f2_2_f1_bn2(channel_weight)
            channel_weight = self.transfer_f2_2_f1_relu2(channel_weight)
            channel_weight_f1 = channel_weight[:,:self.f1_channel].view(B,-1,1,1)
            channel_weight_f2 = channel_weight[:,self.f1_channel:].view(B,-1,1,1)

        f1_transferred = f1_transferred * channel_weight_f1
        f2_transferred = f2_transferred * channel_weight_f2


        f_fusion = f1_transferred+f2_transferred


        return f_fusion
    



# class FeatureAlignmentModule_without_channel(nn.Module):
#     def __init__(self, f1_channel = 96,f2_channel = 32, f1_2_f2 = True, attention = True):
#         super().__init__()
#         self.f1_2_f2 = f1_2_f2
#         self.f1_channel = f1_channel
#         self.f2_channel = f2_channel
#         self.attention = attention

#         self.transfer_f1_2_f2 = nn.Conv2d(in_channels=f1_channel,out_channels=f2_channel,kernel_size=1)
#         self.transfer_f2_2_f1 = nn.Conv2d(in_channels=f2_channel,out_channels=f1_channel,kernel_size=1)

#         self.f1_avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.f1_max_pool = nn.AdaptiveAvgPool2d(1)
#         self.f2_avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.f2_max_pool = nn.AdaptiveAvgPool2d(1)

#         self.transfer_f1_2_f2_fc1 = nn.Linear(in_features=f2_channel*4,out_features=f2_channel)
#         self.transfer_f1_2_f2_bn1 = nn.BatchNorm1d(f2_channel)
#         self.transfer_f1_2_f2_relu1 = nn.ReLU()
#         self.transfer_f1_2_f2_fc2 = nn.Linear(in_features=f2_channel,out_features=f2_channel*2)
#         self.transfer_f1_2_f2_bn2 = nn.BatchNorm1d(f2_channel*2)
#         self.transfer_f1_2_f2_relu2 = nn.Sigmoid()

#         self.transfer_f2_2_f1_fc1 = nn.Linear(in_features=f1_channel*4,out_features=f1_channel)
#         self.transfer_f2_2_f1_bn1 = nn.BatchNorm1d(f1_channel)
#         self.transfer_f2_2_f1_relu1 = nn.ReLU()
#         self.transfer_f2_2_f1_fc2 = nn.Linear(in_features=f1_channel,out_features=f1_channel*2)
#         self.transfer_f2_2_f1_bn2 = nn.BatchNorm1d(f1_channel*2)
#         self.transfer_f2_2_f1_relu2 = nn.Sigmoid()
#         if self.f1_2_f2:
#             dim = f2_channel
#         else:
#             dim = f1_channel
#         if self.attention:
#             self.attention  = Attention(dim=dim)

#     def forward(self, f1, f2):

#         if self.f1_2_f2:
#             f1_transferred = self.transfer_f1_2_f2(f1)
#             f2_transferred = f2
#         else:
#             f1_transferred = f1
#             f2_transferred = self.transfer_f2_2_f1(f2)


#         f1_avg = self.f1_avg_pool(f1_transferred)
#         f1_max = self.f1_max_pool(f1_transferred)

#         f2_avg = self.f2_avg_pool(f2_transferred)
#         f2_max = self.f2_max_pool(f2_transferred)

#         channel_weight = torch.cat((f1_avg,f1_max,f2_avg,f2_max),dim=1)
#         B,C,W,H = channel_weight.size()
#         channel_weight = channel_weight.view(B,-1)
#         if self.f1_2_f2:
#             channel_weight = self.transfer_f1_2_f2_fc1(channel_weight)
#             channel_weight = self.transfer_f1_2_f2_bn1(channel_weight)
#             channel_weight = self.transfer_f1_2_f2_relu1(channel_weight)
#             channel_weight = self.transfer_f1_2_f2_fc2(channel_weight)
#             channel_weight = self.transfer_f1_2_f2_bn2(channel_weight)
#             channel_weight = self.transfer_f1_2_f2_relu2(channel_weight)
#             channel_weight_f1 = channel_weight[:,:self.f2_channel].view(B,-1,1,1)
#             channel_weight_f2 = channel_weight[:,self.f2_channel:].view(B,-1,1,1)
#         else:
#             channel_weight = self.transfer_f2_2_f1_fc1(channel_weight)
#             channel_weight = self.transfer_f2_2_f1_bn1(channel_weight)
#             channel_weight = self.transfer_f2_2_f1_relu1(channel_weight)
#             channel_weight = self.transfer_f2_2_f1_fc2(channel_weight)
#             channel_weight = self.transfer_f2_2_f1_bn2(channel_weight)
#             channel_weight = self.transfer_f2_2_f1_relu2(channel_weight)
#             channel_weight_f1 = channel_weight[:,:self.f1_channel].view(B,-1,1,1)
#             channel_weight_f2 = channel_weight[:,self.f1_channel:].view(B,-1,1,1)

#         f1_transferred = f1_transferred * channel_weight_f1
#         f2_transferred = f2_transferred * channel_weight_f2


#         f_fusion = f1_transferred+f2_transferred

#         # if self.attention:
#         #     f_fusion = self.attention(f_fusion)

#         return f_fusion


# class FeatureAlignmentModule_without_spatial(nn.Module):
#     def __init__(self, f1_channel = 96,f2_channel = 32, f1_2_f2 = True, attention = True):
#         super().__init__()
#         self.f1_2_f2 = f1_2_f2
#         self.f1_channel = f1_channel
#         self.f2_channel = f2_channel
#         self.attention = attention
#         self.conv1 = nn.Conv2d(in_channels=f1_channel+f2_channel+4, out_channels=(f1_channel+f2_channel+4)//4,kernel_size=3,stride=1,padding=1)
#         self.conv1_relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(in_channels=(f1_channel+f2_channel+4)//4,out_channels=2,kernel_size=3,stride=1,padding=1)
#         self.conv2_sigmoid = nn.Sigmoid()

#         self.transfer_f1_2_f2 = nn.Conv2d(in_channels=f1_channel,out_channels=f2_channel,kernel_size=1)
#         self.transfer_f2_2_f1 = nn.Conv2d(in_channels=f2_channel,out_channels=f1_channel,kernel_size=1)

#         if self.f1_2_f2:
#             dim = f2_channel
#         else:
#             dim = f1_channel
#         if self.attention:
#             self.attention  = Attention(dim=dim)

#     def forward(self, f1, f2):

#         f1_channel_max = torch.max(f1,dim=1)[0]
#         f1_channel_max = f1_channel_max.unsqueeze(1)
#         f1_channel_mean = torch.mean(f1,dim=1)
#         f1_channel_mean = f1_channel_mean.unsqueeze(1)
#         f1_new = torch.cat((f1,f1_channel_max,f1_channel_mean),dim=1)

#         f2_channel_max = torch.max(f2,dim=1)[0]
#         f2_channel_max = f2_channel_max.unsqueeze(1)
#         f2_channel_mean = torch.mean(f2,dim=1)
#         f2_channel_mean = f2_channel_mean.unsqueeze(1)
#         f2_new = torch.cat((f2,f2_channel_max,f2_channel_mean),dim=1)
        
#         f1_f2_cat = torch.cat((f1_new,f2_new),dim=1)

#         f1f2 = self.conv1(f1_f2_cat)
#         f1f2 = self.conv1_relu(f1f2)
#         f1f2_spatial = self.conv2(f1f2)
#         f1f2_spatial = self.conv2_sigmoid(f1f2_spatial)

#         f1_spatial_weighted = f1 * f1f2_spatial[:,:1,:,:]
#         f2_spatial_weighted = f2 * f1f2_spatial[:,1:,:,:]

#         if self.f1_2_f2:
#             f1_transferred = self.transfer_f1_2_f2(f1_spatial_weighted)
#             f2_transferred = f2_spatial_weighted
#         else:
#             f1_transferred = f1_spatial_weighted
#             f2_transferred = self.transfer_f2_2_f1(f2_spatial_weighted)

#         f_fusion = f1_transferred+f2_transferred

#         if self.attention:
#             f_fusion = self.attention(f_fusion)

#         return f_fusion


# class FeatureAlignmentModule_only_transfer(nn.Module):
#     def __init__(self, f1_channel = 96,f2_channel = 32, f1_2_f2 = True, attention = True):
#         super().__init__()
#         self.f1_2_f2 = f1_2_f2
#         self.f1_channel = f1_channel
#         self.f2_channel = f2_channel
#         self.attention = attention

#         self.transfer_f1_2_f2 = nn.Conv2d(in_channels=f1_channel,out_channels=f2_channel,kernel_size=1)
#         self.transfer_f2_2_f1 = nn.Conv2d(in_channels=f2_channel,out_channels=f1_channel,kernel_size=1)

#         if self.f1_2_f2:
#             dim = f2_channel
#         else:
#             dim = f1_channel
#         if self.attention:
#             self.attention  = Attention(dim=dim)

#     def forward(self, f1, f2):

#         if self.f1_2_f2:
#             f1_transferred = self.transfer_f1_2_f2(f1)
#             f2_transferred = f2
#         else:
#             f1_transferred = f1
#             f2_transferred = self.transfer_f2_2_f1(f2)

#         f_fusion = f1_transferred+f2_transferred

#         if self.attention:
#             f_fusion = self.attention(f_fusion)

#         return f_fusion

# class FeatureAlignmentModule_connected(nn.Module):
#     def __init__(self, f1_channel = 96,f2_channel = 32, f1_2_f2 = True, attention = True):
#         super().__init__()
#         self.f1_2_f2 = f1_2_f2
#         self.f1_channel = f1_channel
#         self.f2_channel = f2_channel
#         self.attention = attention

#         self.transfer_f1_2_f2 = nn.Conv2d(in_channels=f1_channel+f2_channel,out_channels=f2_channel,kernel_size=1)
#         self.transfer_f2_2_f1 = nn.Conv2d(in_channels=f1_channel+f2_channel,out_channels=f1_channel,kernel_size=1)

#         if self.f1_2_f2:
#             dim = f2_channel
#         else:
#             dim = f1_channel
#         if self.attention:
#             self.attention  = Attention(dim=dim)

#     def forward(self, f1, f2):

#         f = torch.cat((f1,f2),dim=1)

#         if self.f1_2_f2:
#             f_fusion = self.transfer_f1_2_f2(f)
#         else:
#             f_fusion = self.transfer_f2_2_f1(f)

#         if self.attention:
#             f_fusion = self.attention(f_fusion)

#         return f_fusion

# class Attention(nn.Module):
#     def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
#         super(Attention, self).__init__()
#         assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

#         self.dim = dim
#         self.num_heads = num_heads
#         head_dim = dim // num_heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.kv1 = nn.Linear(dim, dim * 3, bias=qkv_bias)

#     def forward(self, x1):

#         B, C, H, W = x1.shape
#         x1 = x1.flatten(2).transpose(1, 2)

#         B, N, C = x1.shape


#         q1,k1, v1 = self.kv1(x1).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

#         ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
#         ctx1 = ctx1.softmax(dim=-2)

#         x1 = (q1 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 

#         x1 = x1.transpose(1, 2).view(B, C, H, W)

#         return x1



        
def unit_test():
    num_minibatch = 2
    rgb = torch.randn(num_minibatch, 32, 128, 128).cuda(1)
    thermal = torch.randn(num_minibatch, 32, 128, 128).cuda(1)
    #skip = Skip_fz(3,2).cuda(0)
    #x = skip(rgb,thermal)
    input = torch.cat((rgb,thermal),dim=1)
    rtf = FeatureAlignmentModule(f1_channel=32,f2_channel=32).cuda(1)
    x = rtf(rgb,thermal)
    print('x: ', x.size())

if __name__ == '__main__':
    unit_test()