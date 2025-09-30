import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('.')

from util.init_func import init_weight
from config import config



class EncoderDecoder(nn.Module):
    def __init__(self,cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), encoder_name='ConvNeXt_1', decoder_name='MY_MLPDecoder', n_class=config.num_classes, norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.norm_layer = norm_layer
        
        # import backbone and decoder
        if encoder_name == 'ConvNeXt_4':
            #logger.info('Using backbone: Segformer-B5')
            from model.TCNet.encoders.ConvNeXt import ConvNeXt_4 as backbone
            self.channels = [256, 512, 1024, 2048]
            print("chose ConvNeXt_4")
            self.backbone = backbone()
        elif encoder_name == 'ConvNeXt_3':
            #logger.info('Using backbone: Segformer-B4')
            from model.TCNet.encoders.ConvNeXt import ConvNeXt_3 as backbone
            self.channels = [192, 384, 768, 1536]
            print("chose ConvNeXt_3")
            self.backbone = backbone()
        elif encoder_name == 'ConvNeXt_2':
            #logger.info('Using backbone: Segformer-B4')
            from model.TCNet.encoders.ConvNeXt import ConvNeXt_2 as backbone
            self.channels = [128, 256, 512, 1024]
            print("chose ConvNeXt_2")
            self.backbone = backbone()
        elif encoder_name == 'ConvNeXt_1':
            #logger.info('Using backbone: Segformer-B2')
            from model.TCNet.encoders.ConvNeXt import ConvNeXt_1 as backbone
            self.channels = [96, 192, 384, 768]
            print("chose ConvNeXt_1")
            self.backbone = backbone()
        elif encoder_name == 'ConvNeXt_0':
            #logger.info('Using backbone: Segformer-B1')
            from model.TCNet.encoders.ConvNeXt import ConvNeXt_0 as backbone
            self.channels = [96, 192, 384, 768]
            print("chose ConvNeXt_0")
            self.backbone = backbone()
        else:
            #logger.info('Using backbone: Segformer-B2')
            from model.TCNet.encoders.ConvNeXt import ConvNeXt_2 as backbone
            self.channels = [128, 256, 512, 1024]
            self.backbone = backbone()

        self.aux_head = None

        if decoder_name == 'MY_MLPDecoder':
            #logger.info('Using MLP Decoder')
            from model.TCNet.decoders.MY_MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=n_class, norm_layer=norm_layer, embed_dim=512)
        else:
            #logger.info('No decoder(FCN-32s)')
            from decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(in_channels=self.channels[-1], kernel_size=3, num_classes=10, norm_layer=norm_layer)

        self.voting = nn.Conv2d(in_channels=n_class*2,out_channels=n_class,kernel_size=3,stride=1,padding=1)

        # self.init_weights(cfg, pretrained=cfg.pretrained_model)
    
    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            self.backbone.init_weights(pretrained=pretrained)
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')

    def encode_decode(self, rgb, modal_x):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        orisize = rgb.shape
        x = self.backbone(rgb, modal_x)
        out, c_locals= self.decode_head.forward(x)
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)

        return out,c_locals

    def forward(self, input):

        rgb = input[:,:3]
        modal_x = input[:,3:]
        out,c_locals = self.encode_decode(rgb, modal_x)

        return out,c_locals

