import torch
import torch.utils
from tqdm import tqdm
import time
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from torch import optim
import yaml
import os
from train import BaseTrainer
import json
import wandb
from vista.pipeline.registry import registry
from vista.pipeline.pipeline_mixin import *
from vista.dataset.data_converter import random_caption_word
from vista.model.vision.basic_modules import get_mlp_head
from vista.model.vision.point_encoder import PointTokenizeEncoder
from vista.model.vision.unified_encoder import UnifiedSpatialCrossEncoderV2
from vista.model.vision.grounding_head import GroundHeadV1
from vista.model.vision.pretrain_head import PretrainHeadV1
from vista.model.language.lang_encoder import get_bert_lang_encoder, get_bert_tokenizer
from vista.optimization.loss_function import get_refer_loss_v1, get_pretrain_loss_v1
from vista.utils.saver import ModelSaver
from transformers.models.bert import BertModel, BertTokenizer
from data.dataloader import VLADataset, dataset_to_dataloader
from utils import *
from arg_parser import parse_args
import argparse



class Grounding3DVistaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # build model
        self.point_encoder = PointTokenizeEncoder(**cfg['point_encoder']['args']).cuda()
        self.lang_encoder = get_bert_lang_encoder(**cfg['lang_encoder']['args']).cuda()
        self.unified_encoder = UnifiedSpatialCrossEncoderV2(**cfg['unified_encoder']['args']).cuda()
        self.ground_head = GroundHeadV1(**cfg['ground_head']['args']).cuda()
        self.pretrain_head = PretrainHeadV1(**cfg['pretrain_head']['args']).cuda()
        self.binary_classifier_head = get_mlp_head(input_size=768, hidden_size=768, output_size=1, dropout=0.3)

    def forward(self, data_dict):
        # prepare data
        # self.prepare_data(data_dict)
        
        # prepare dict
        if 'cur_step' not in data_dict.keys():
            data_dict['cur_step'] = 1
            data_dict['total_steps'] = 1
        # basic feature extracter
        # point_features_pre_spatial is point features before spatial reasoning
        lang_basic_features = self.lang_encoder(data_dict['txt_ids'], data_dict['txt_masks']).last_hidden_state
        point_basic_features, point_features_pre, obj_cls_raw_logits = self.point_encoder(
            data_dict['obj_fts'].float(), 
            data_dict['obj_locs'], 
            data_dict['obj_masks'], 
            data_dict['obj_sem_masks'], 
            data_dict['obj_labels'], 
            data_dict['cur_step'], 
            data_dict['total_steps'])
        
        # unifed language entity transformer
        language_fuse_feature, point_fuse_feature  = self.unified_encoder(lang_basic_features, data_dict['txt_masks'], point_basic_features, data_dict['obj_locs'], data_dict['obj_masks'])
        
        # task head
        txt_cls_logits, obj_cls_post_logits, obj_cls_pre_logits, og3d_logits = self.ground_head(language_fuse_feature, point_fuse_feature, point_features_pre, data_dict['obj_masks'])
        txt_lm_cls_logits, scene_txt_match_logit = self.pretrain_head(language_fuse_feature)
        object_existence_logit = self.binary_classifier_head(language_fuse_feature[0, :])
        
        data_dict['txt_cls_logits'] = txt_cls_logits
        data_dict['obj_cls_post_logits'] = obj_cls_post_logits
        data_dict['obj_cls_pre_logits'] = obj_cls_pre_logits
        data_dict['obj_cls_raw_logits'] = obj_cls_raw_logits
        data_dict['og3d_logits'] = og3d_logits
        data_dict['txt_lm_cls_logits'] = txt_lm_cls_logits
        data_dict['scene_txt_match_logit'] = scene_txt_match_logit
        data_dict['object_existence_logit'] = object_existence_logit
        
        return data_dict

