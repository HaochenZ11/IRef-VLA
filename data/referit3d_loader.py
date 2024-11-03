import os
import csv
from pathlib import Path
from tqdm import tqdm
import json
import time
import torch
import ast
import random
import numpy as np

from arg_parser import parse_args
from data.dataloader import VLADataset
from utils import *

# subclassed Dataset class to load ReferIt3D data
class ReferIt3DDataset(VLADataset):
    def __init__(
            self, 
            data_path='./', 
            num_obj_pts=1024, 
            num_region_pts=75000, 
            pad_idx=900,
            max_classes=895, 
            metadata_file='metadata.json', 
            split='train', 
            use_color=False, 
            normalize=False, 
            prune_sparse=True, 
            sparsity_thresh=50,
            context_size=50, 
            use_context=False,
            region_splits_torch=False,
            save_scene_pcs=True,
            save_region_pcs=True,
            sr3d=True,
            include_raw_labels=False,
            random_seed=0):

        self.num_obj_pts = num_obj_pts
        self.num_region_pts = num_region_pts
        self.max_num_objs = 0
        self.max_distractors = 0
        self.pad_idx = pad_idx
        self.max_classes = max_classes
        self.data_path = data_path
        self.metadata_file = metadata_file
        self.split = split
        self.prune_sparse = prune_sparse
        self.sparsity_thresh = sparsity_thresh
        self.context_size = context_size
        self.use_context = use_context
        self.region_splits_torch = region_splits_torch
        self.save_scene_pcs = save_scene_pcs
        self.save_region_pcs = save_region_pcs
        self.sr3d = sr3d
        self.include_raw_labels=include_raw_labels
        self.load_false_statements = False

        # set random seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        # take in file listing scenes for split
        scene_file = os.path.join(data_path, split + '.txt')
        self.scene_list = []

        with open(scene_file) as f:
            for line in f:
                # skip empty lines
                if not line or line == '\n':
                    continue
                self.scene_list.append(line.rstrip())

        # only use scans from Scannet
        self.datasets = ['Scannet']

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # call loading functions
        print("Loading scenes...")
        self.all_scenes = self.load_scenes()

        # split into regions and store data per region
        print("Splitting and loading regions...")
        self.all_regions = self.get_regions()

        if self.use_context:
            self.max_context_len = self.context_size
        else:
            self.max_context_len = self.max_num_objs

        # loading language data
        print("Loading referential data...")
        self.referential_data = self.read_refer_csv('../data/sr3d.csv', self.scene_list)
        
        self.filter_data()
        
        print("Split: {}, Number of scenes: {}".format(split, len(self.scene_list)))


    def read_refer_csv(self, csv_path, scene_list):
        '''
        Load language data from referit3d csv
        '''
        refer_data = []
        max_distractors = 0
        with open(csv_path, encoding='utf-8') as csv_file:
            csvReader = csv.DictReader(csv_file)
            for row in csvReader:

                # skip allocentric statements
                if row["scan_id"] in scene_list: #and row["coarse_reference_type"] != "allocentric":
                    if self.sr3d:
                        # for sr3d
                        sample = {
                            "scene": row["scan_id"],
                            "region": "0",
                            "utterance": row["utterance"],
                            "target_label": row["instance_type"],
                            "target_obj_id": str(row["target_id"]),
                            "distractor_ids": ast.literal_eval(row["distractor_ids"]),
                            "relation": row["reference_type"],
                            "anchor_obj_ids": ast.literal_eval(row["anchor_ids"]),
                            "real": True
                        }
                    else:
                        # for nr3d
                        sample = {
                            "scene": row["scan_id"],
                            "region": "0",
                            "utterance": row["utterance"],
                            "target_label": row["instance_type"],
                            "target_obj_id": str(row["target_id"]),
                            "distractor_ids": [], # not in data
                            "relation": "",
                            "anchor_obj_ids": [],
                            "real": True
                        }

                    if len(sample["distractor_ids"]) > max_distractors:
                        max_distractors = len(sample["distractor_ids"])
                    
                    refer_data.append(sample)
        
        # needs to match what if combining datasets
        self.max_distractors = max_distractors # 71 for scannet train
        print("MAX DIST R3D", self.max_distractors)
        return refer_data


    def _check_skipped_obs(self, data, skipped_objs):
        '''
        Prune statements referring to skipped/ignored objects
        '''
        if data["target_obj_id"] in skipped_objs:
            return True
        for i in data["distractor_ids"]:
            if str(i) in skipped_objs:
                return True
        for i in data["anchor_obj_ids"]:
            if str(i) in skipped_objs:
                return True
        
        return False


    def filter_data(self):
        '''
        Filter out referential data involving skipped objects
        '''
        filtered_data = []
        for data in self.referential_data:
            scene_id = data["scene"]
            skipped_objs = self.all_scenes[scene_id]["skipped_objs"]
            if not self._check_skipped_obs(data, skipped_objs):
                filtered_data.append(data)
        
        self.referential_data = filtered_data
    

    def __getitem__(self, idx):
        '''
        Get data sample
        :return:
        '''
        ret = {}
        data = self.referential_data[idx]
        region_id = data["region"]
        scene_id = data["scene"]
        region_data = self.all_regions[scene_id + '_' + region_id]

        objects = region_data["objects"]

        # custom
        if self.save_region_pcs:
            pc = region_data["pc"]
            ret["region_pc"] = pc
        ret["anchor_ids"] = np.array([int(id) for id in data["anchor_obj_ids"]])

        # if get subset of data (not all objects in a region)
        if self.use_context:
            objects = self.get_subset(objects, data["target_obj_id"], [id for id in data["anchor_obj_ids"]])

        # for one sample:
        obj_pcs = np.array([o["object_pc"] for o in objects.values()], dtype=np.float64)
        ret["class_labels"] = self._get_class_labels(objects)
        ret["scan_id"] = region_data["scene_id"]
        ret["context_size"] = len(obj_pcs) # by default, is all objects
        ret["objects"] = obj_pcs
        ret["target_class"] = int(objects[data["target_obj_id"]]["object_class"])

        ret["target_pos"] = int([ind for ind in range(len(objects.keys())) if list(objects.keys())[ind] == data["target_obj_id"]][0])
        ret["target_class_mask"] = np.array(ret["class_labels"] == ret["target_class"])
        ret["tokens"] = data["utterance"] # get rid of punctuation?
        #ret["is_nr3d"] # unused
        ret["box_info"] = self._get_box_info(objects)
        ret["box_corners"] = np.array([o["object_bbox"] for o in objects.values()])
        ret["utterance"] = data["utterance"]
        ret["axis_aligned_size"] = np.array([o["axis_aligned_size"] for o in objects.values()])

        # only if visualization flag set (in MVT)
        stimulus_str = region_data["scene_id"] + '-' + str(data["target_label"]) + '-' + str(len(obj_pcs)) + '-' + data["target_obj_id"]
        for id in data["distractor_ids"]:
            stimulus_str += '-' + str(id)
        ret["stimulus_id"] = stimulus_str

        # get distractor ids
        if len(data["distractor_ids"]) == 0:
            ret["distractor_ids"] = np.array([int(self.max_num_objs+1) for _ in range(self.max_distractors)])
            ret["distractors_pos"] = np.array([int(self.max_num_objs+1) for _ in range(self.max_distractors)])
        else:
            ret["distractor_ids"] = np.array(data["distractor_ids"])
            distractor_pos = []
            for id in ret["distractor_ids"]:
                id_pos = [ind for ind in range(len(objects.keys())) if int(list(objects.keys())[ind]) == id][0]
                distractor_pos.append(int(id_pos))

            ret["distractors_pos"] = np.array(distractor_pos)

        ret["object_ids"] = np.array([int(o) for o in objects.keys()])

        #### TEMP FIX for mismatched obj ids
        for a in range(len(data["anchor_obj_ids"])):
            i = 0
            for o_id, o in objects.items():
                if i == data["anchor_obj_ids"][a]:
                    data["anchor_obj_ids"][a] = o_id
                    break
                i += 1

        # add padding
        ret = self.pad_item(ret)
        #print("ret", ret)

        return ret
