from __future__ import absolute_import

from PIL import Image
import numpy as np
from collections import OrderedDict
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from models.aster import modules as md

class Model(nn.Module):
    """
    This is the integrated model.
    """
    def __init__(self, conf, classes, sDim, attDim, max_len_labels, STN_ON=False):
        super(Model, self).__init__()
        
        self.classes = classes
        self.sDim = sDim
        self.attDim = attDim
        self.max_len_labels = max_len_labels
        self.STN_ON = STN_ON
        self.tps_inputsize = [32, 64]
        self.tps_outputsize = [32, 100]
        self.tps_matgins =  [0.05, 0.05]
        self.num_control_points = 20
        self.stn_activation = None

        #self.encoder = create(self.arch,
        #                  with_lstm=global_args.with_lstm,
        #                  n_group=global_args.n_group)
        self.encoder = md.resnet_aster.ResNet_ASTER(with_lstm=True)
        self.encoder_out_planes = self.encoder.out_planes

        self.decoder = md.attention_recognition_head.AttentionRecognitionHead(
                              num_classes=classes,
                              in_planes=self.encoder_out_planes,
                              sDim=sDim,
                              attDim=attDim,
                              max_len_labels=max_len_labels)
        self.rec_crit = md.sequenceCrossEntropyLoss.SequenceCrossEntropyLoss()
        self.pd = md.prediction.Attention(self.encoder_out_planes, conf["Model"]["hidden_size"], self.classes)
        
        if self.STN_ON:
            self.tps = md.tps_spatial_transformer.TPSSpatialTransformer(
                            output_image_size=self.tps_outputsize,
                            num_control_points=self.num_control_points,
                            margins=self.tps_matgins
            )
            self.stn_head = md.stn_head.STNHead(
                                in_planes=3,
                                num_ctrlpoints=self.num_control_points,
                                activation=self.stn_activation
            )

    def forward(self, img, text, max_batch, is_train=True):
        x, rec_targets, rec_lengths = img, text, max_batch
        
        
        # rectification
        if self.STN_ON:
            # input images are downsampled before being fed into stn_head.
            stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
            stn_img_feat, ctrl_points = self.stn_head(stn_input)
            x, _ = self.tps(x, ctrl_points)

        encoder_feats = self.encoder(x)
        encoder_feats = encoder_feats.contiguous()

        if is_train:
            rec_pred = self.pd(encoder_feats.contiguous(), text, is_train, max_batch-1)
            #rec_pred = self.decoder([encoder_feats, rec_targets, rec_lengths])
            #loss_rec = self.rec_crit(rec_pred, rec_targets, rec_lengths)
        else:
            #rec_pred, rec_pred_scores = self.decoder.beam_search(encoder_feats, global_args.beam_width, self.eos)
            rec_pred_ = self.decoder([encoder_feats, rec_targets, rec_lengths])
            loss_rec = self.rec_crit(rec_pred_, rec_targets, rec_lengths)

        return rec_pred