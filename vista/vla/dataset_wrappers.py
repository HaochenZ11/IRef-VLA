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
from vista.model.vision.point_encoder import PointTokenizeEncoder
from vista.model.vision.unified_encoder import UnifiedSpatialCrossEncoderV2
from vista.model.vision.grounding_head import GroundHeadV1
from vista.model.vision.pretrain_head import PretrainHeadV1
from vista.model.language.lang_encoder import get_bert_lang_encoder, get_bert_tokenizer
from vista.optimization.loss_function import get_refer_loss_v1, get_pretrain_loss_v1
from vista.utils.saver import ModelSaver
from transformers.models.bert import BertModel, BertTokenizer
from data.dataloader import VLADataset, dataset_to_dataloader
from data.referit3d_loader import ReferIt3DDataset
from utils import *
from arg_parser import parse_args
import argparse

class VLADatasetWrapper(VLADataset):

    def __init__(
            self,
            tokenizer=None,
            max_seq_length=80, 
            **kwargs
            ):
        super().__init__(**kwargs)
        tokenizer: BertModel = registry.get_language_model(tokenizer)()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        if self.use_context:
            self.max_obj_len = self.context_size
        else:
            self.max_obj_len = self.max_num_objs
        self.map_from_raw_labels = False

        # Create nyuid to scannet id mapping
        self.int2cat = json.load(open(os.path.join('vista/data/scanfamily', "annotations/meta_data/scannetv2_raw_categories.json"), 'r'))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        with open('vista/data/scanfamily/annotations/meta_data/nyu_id_to_scannet_id.json', 'r') as f:
            self.nyu_to_scannet = json.load(f)
        

    def _map_nyuids_to_scannet(self, ids):

        return np.array([int(self.nyu_to_scannet[str(i)]['scannet_id']) for i in ids])
    
    def _map_raw_labels_to_scannet(self, labels):

        return np.array([self.cat2int[label] for label in labels])
    
    def pad_item(self, ret):
        return ret
    
    def pad_tensors(self, tensors, lens=None, pad=0):
        try:
            assert tensors.shape[0] <= lens
        except:
            print(tensors.shape[0], lens)
            print(tensors.shape)
        if (tensors.shape[0] == lens):
            return tensors
        shape = list(tensors.shape)
        shape[0] = lens - shape[0]
        res = torch.ones(shape, dtype=tensors.dtype) * pad
        res = torch.cat((tensors, res), dim=0)
        return res
    
    def __getitem__(self, idx):

        data_dict = super().__getitem__(idx)
        utterance = data_dict['utterance']
        encoded_input = self.tokenizer(utterance, max_length=self.max_seq_length,
                          add_special_tokens=True, truncation=True,
                          padding='max_length', return_tensors="pt")
        out_dict = {}
        out_dict['utterance'] = data_dict['utterance']
        # out_dict['real'] = data_dict['real']
        # build txt
        out_dict['txt_ids'] = encoded_input['input_ids'].squeeze(0) # L
        out_dict['txt_masks'] = encoded_input['attention_mask'].squeeze(0) # L
        # build object
        obj_locs = torch.cat([torch.from_numpy(data_dict['box_info'][..., :3]), torch.from_numpy(data_dict['axis_aligned_size'])], dim=-1)
        out_dict['obj_masks'] = (torch.arange(self.max_obj_len) < len(obj_locs)) # O
        out_dict['obj_fts'] = self.pad_tensors(torch.from_numpy(data_dict['objects']), lens=self.max_obj_len, pad=1.0).float() # O, 1024, 6
        out_dict['obj_locs']= self.pad_tensors(obj_locs, lens=self.max_obj_len, pad=0.0).float() # O, 3

        if self.map_from_raw_labels:
            obj_labels = self._map_raw_labels_to_scannet(data_dict['raw_labels'])
        else:
            obj_labels = self._map_nyuids_to_scannet(data_dict['class_labels'])
        out_dict['obj_labels'] = self.pad_tensors(torch.from_numpy(obj_labels), lens=self.max_obj_len, pad=-100).long() # O
        # build sem mask, no mask
        out_dict['obj_sem_masks'] = (torch.arange(self.max_obj_len) < len(out_dict['obj_locs']))
        # build label for refer
        if self.map_from_raw_labels:
            tgt_object_label = self._map_raw_labels_to_scannet([data_dict['target_raw_label']])
        else:
            tgt_object_label = self._map_nyuids_to_scannet([data_dict['target_class']])
        out_dict['tgt_object_label'] = torch.LongTensor(tgt_object_label) # 1 or C
        out_dict['tgt_object_id'] = torch.LongTensor([data_dict['target_pos']]) # 1 or O
        if len(out_dict['tgt_object_id']) > 1: # O, pad to max objet length
            out_dict['tgt_object_id'] = self.pad_tensors(data_dict['tgt_object_id'].long(), lens=self.max_obj_len, pad=0).long() # O
        # build target
        if data_dict.get('tgt_object_id_iou25') != None:
            out_dict['tgt_object_id_iou25'] = self.pad_tensors(data_dict['tgt_object_id_iou25'], lens=self.max_obj_len, pad=0).long()
        if data_dict.get('tgt_object_id_iou50') != None:
            out_dict['tgt_object_id_iou50'] = self.pad_tensors(data_dict['tgt_object_id_iou50'], lens=self.max_obj_len, pad=0).long()

        return out_dict


class ReferIt3DDatasetWrapper(VLADatasetWrapper, ReferIt3DDataset):
    def __init__(
            self,
            tokenizer=None,
            max_seq_length=80, 
            **kwargs
            ):
        ReferIt3DDataset.__init__(self, **kwargs)
        tokenizer: BertModel = registry.get_language_model(tokenizer)()
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        if self.use_context:
            self.max_obj_len = self.context_size
        else:
            self.max_obj_len = self.max_num_objs
        self.map_from_raw_labels = False

        # Create nyuid to scannet id mapping
        self.int2cat = json.load(open(os.path.join('vista/data/scanfamily', "annotations/meta_data/scannetv2_raw_categories.json"), 'r'))
        self.cat2int = {w: i for i, w in enumerate(self.int2cat)}
        with open('vista/data/scanfamily/annotations/meta_data/nyu_id_to_scannet_id.json', 'r') as f:
            self.nyu_to_scannet = json.load(f)