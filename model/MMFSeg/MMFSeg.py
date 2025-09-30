import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('.')

from util.init_func import init_weight
from config import config

from model.MMFSeg.net_utils import FeatureAlignmentModule

class EncoderDecoder(nn.Module):
    def __init__(self,cfg=None, Trans_encoder_name='mit_b1', Conv_encoder_name='ConvNeXt_0', decoder_name='outlocal', f1_2_f2=False,  norm_layer=nn.BatchNorm2d,n_class=2):
        super(EncoderDecoder, self).__init__()

        self.norm_layer = norm_layer

        if Trans_encoder_name == 'mit_b5':
            from model.MMFSeg.encoders.dual_segformer import mit_b5 as backbone
            print("!!!!!!!!!!!          mit_b5        !!!!!!!!!!!!!!!!!!!!!!!!!")
            self.trans_channels = [64, 128, 320, 512]
            self.backbone_trans = backbone(norm_fuse=norm_layer)
        elif Trans_encoder_name == 'mit_b4':
            from model.MMFSeg.encoders.dual_segformer import mit_b4 as backbone
            self.trans_channels = [64, 128, 320, 512]
            print("!!!!!!!!!!!          mit_b4        !!!!!!!!!!!!!!!!!!!!!!!!!")
            self.backbone_trans = backbone(norm_fuse=norm_layer)
        elif Trans_encoder_name == 'mit_b3':
            from model.MMFSeg.encoders.dual_segformer import mit_b3 as backbone
            self.trans_channels = [64, 128, 320, 512]
            print("!!!!!!!!!!!          mit_b3        !!!!!!!!!!!!!!!!!!!!!!!!!")
            self.backbone_trans = backbone(norm_fuse=norm_layer)
        elif Trans_encoder_name == 'mit_b2':
            from model.MMFSeg.encoders.dual_segformer import mit_b2 as backbone
            self.trans_channels = [64, 128, 320, 512]
            print("!!!!!!!!!!!          mit_b2        !!!!!!!!!!!!!!!!!!!!!!!!!")
            self.backbone_trans = backbone(norm_fuse=norm_layer)
        elif Trans_encoder_name == 'mit_b1':
            from model.MMFSeg.encoders.dual_segformer import mit_b1 as backbone
            self.trans_channels = [64, 128, 320, 512]
            print("!!!!!!!!!!!          mit_b1        !!!!!!!!!!!!!!!!!!!!!!!!!")
            self.backbone_trans = backbone(norm_fuse=norm_layer)
        elif Trans_encoder_name == 'mit_b0':
            self.trans_channels = [32, 64, 160, 256]
            from model.MMFSeg.encoders.dual_segformer import mit_b0 as backbone
            print("!!!!!!!!!!!          mit_b0        !!!!!!!!!!!!!!!!!!!!!!!!!")
            self.backbone_trans = backbone(norm_fuse=norm_layer)
        else:
            from encoders.dual_segformer import mit_b2 as backbone
            print("zhele?")
            self.trans_channels = [64, 128, 320, 512]
            self.backbone_trans = backbone(norm_fuse=norm_layer)

        if Conv_encoder_name == 'ConvNeXt_4':
            from model.MMFSeg.encoders.ConvNeXt import ConvNeXt_4 as backbone
            self.conv_channels = [256, 512, 1024, 2048]
            self.backbone_conv = backbone()
        elif Conv_encoder_name == 'ConvNeXt_3':
            from model.MMFSeg.encoders.ConvNeXt import ConvNeXt_3 as backbone
            self.conv_channels = [192, 384, 768, 1536]
            self.backbone_conv = backbone()
        elif Conv_encoder_name == 'ConvNeXt_2':
            from model.MMFSeg.encoders.ConvNeXt import ConvNeXt_2 as backbone
            self.conv_channels = [128, 256, 512, 1024]
            self.backbone_conv = backbone()
        elif Conv_encoder_name == 'ConvNeXt_1':
            from model.MMFSeg.encoders.ConvNeXt import ConvNeXt_1 as backbone
            self.conv_channels = [96, 192, 384, 768]
            self.backbone_conv = backbone()
        elif Conv_encoder_name == 'ConvNeXt_0':
            from model.MMFSeg.encoders.ConvNeXt import ConvNeXt_0 as backbone
            self.conv_channels = [96, 192, 384, 768]
            self.backbone_conv = backbone()
        else:
            #logger.info('Using backbone: Segformer-B2')
            from model.MMFSeg.encoders.ConvNeXt import ConvNeXt_2 as backbone
            self.conv_channels = [128, 256, 512, 1024]
            self.backbone_conv = backbone()


        self.aux_head = None

        if decoder_name == 'outlocal':
            #logger.info('Using MLP Decoder')
            from model.MMFSeg.decoders.MY_MLPDecoder_outlocal import DecoderHead
            if f1_2_f2:
                self.channels = self.conv_channels
                print("!!!!!!!!!!!    T2C     !!!!!!!!!!!!!")
            else:
                self.channels = self.trans_channels
                print("!!!!!!!!!!!    C2T     !!!!!!!!!!!!!")
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=n_class, norm_layer=norm_layer, embed_dim=512)
        else:
            #logger.info('No decoder(FCN-32s)')
            
            from decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(in_channels=self.channels[-1], kernel_size=3, num_classes=10, norm_layer=norm_layer)

        self.init_weights(cfg, pretrained=cfg.pretrained_model)

        # if attention:
        #     print("!!!!!!!!!!!     Attention        !!!!!!!!!!!!!")
        # else:
        #     print("!!!!!!!!!!!   No  Attention        !!!!!!!!!!!!!")

        self.fusion1 = FeatureAlignmentModule(self.trans_channels[0],self.conv_channels[0],f1_2_f2=f1_2_f2)
        self.fusion2 = FeatureAlignmentModule(self.trans_channels[1],self.conv_channels[1],f1_2_f2=f1_2_f2)
        self.fusion3 = FeatureAlignmentModule(self.trans_channels[2],self.conv_channels[2],f1_2_f2=f1_2_f2)
        self.fusion4 = FeatureAlignmentModule(self.trans_channels[3],self.conv_channels[3],f1_2_f2=f1_2_f2)


    
    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            self.backbone_trans.init_weights(pretrained=pretrained)
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        x_trans = self.backbone_trans(rgb, modal_x)
        x_trans_1, x_trans_2, x_trans_3, x_trans_4 = x_trans

        x_conv = self.backbone_conv(rgb, modal_x)
        x_conv_1, x_conv_2, x_conv_3, x_conv_4 = x_conv

        x = []
        x1 = self.fusion1(x_trans_1,x_conv_1)
        x.append(x1)
        x2 = self.fusion2(x_trans_2,x_conv_2)
        x.append(x2)
        x3 = self.fusion3(x_trans_3,x_conv_3)
        x.append(x3)
        x4 = self.fusion4(x_trans_4,x_conv_4)
        x.append(x4)

        out= self.decode_head.forward(x)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)

        return out

    def forward(self, input):

        rgb = input[:,:3]
        modal_x = input[:,3:]
        out = self.encode_decode(rgb, modal_x)

        return out

