'''
Copyright 2024: Haochen Zhang, Nader Zantout, Pujith Kachana
Parts adapted from: butd_detr, referit3d, MVT-3DVG repositories
https://github.com/nickgkan/butd_detr
https://github.com/referit3d/referit3d
https://github.com/sega-hsj/MVT-3DVG
'''

import numpy as np
import torch
import os
import json
import csv
from pathlib import Path
from torch.utils.data import Dataset
import open3d as o3d
from plyfile import PlyData, PlyElement
from tqdm import tqdm
import time
import random

from utils import *


class VLADataset(Dataset):
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
            include_raw_labels=False,
            load_false_statements=False,
            balance_false=False,
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
        self.include_raw_labels = include_raw_labels
        self.load_false_statements = load_false_statements
        self.balance_false = balance_false

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

        # read metadata
        self.read_metadata()

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
        self.referential_data = self.load_json()
        
        print("Split: {}, Number of scenes: {}".format(split, len(self.scene_list)))


    def read_metadata(self):
        '''
        Reads in stored metadata from file
        '''
        metadata_file = os.path.join(self.data_path, self.metadata_file)

        with open(metadata_file) as f:
            metadata = json.load(f)
            self.all_classes = metadata["class_ids"]
            self.class_name_list = metadata["class_names"]
            self.datasets = metadata["datasets"]


    def __len__(self):
        return len(self.referential_data)


    def _rotz(self, t):
        '''
        Computes rotation matrix for rotation about the z-axis
        '''
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])


    def _get_bbox_coords_heading(self, prefix, row):
        '''
        Get bbox corner points based on center, size, and heading
        '''
        lengths = []
        center = []
        heading = float(row[prefix + '_bbox_heading'])
        # get centers and lengths for each axis
        for ax in ['x', 'y', 'z']:
            ax_c_key = prefix + '_bbox_c' + ax
            ax_c = float(row[ax_c_key])
            ax_l_key = prefix + '_bbox_' + ax + 'length'
            ax_l = float(row[ax_l_key])
            lengths.append(ax_l)
            center.append(ax_c)

        h = float(lengths[2])
        w = float(lengths[1])
        l = float(lengths[0])

        R = self._rotz(1*heading)
        l = l/2
        w = w/2
        h = h/2
        x_corners = [-l,l,l,-l,-l,l,l,-l]
        y_corners = [w,w,-w,-w,w,w,-w,-w]
        z_corners = [h,h,h,h,-h,-h,-h,-h]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0,:] += center[0]
        corners_3d[1,:] += center[1]
        corners_3d[2,:] += center[2]

        coords = []
        for i in range(corners_3d.shape[-1]):
            coords.append(list(corners_3d[:, i]))

        return coords


    def _get_color_vals(self, row):
        '''
        Get rgb color values if color is stored
        '''
        color_vals = []
        for idx in range(1, 4, 1):
            r = row['object_color_r' + str(idx)]
            g = row['object_color_g' + str(idx)]
            b = row['object_color_b' + str(idx)]
            if '_' in [r, g, b] or None in [r, g, b] or '' in [r, g, b]:
                color_vals.append([-1, -1, -1])
            else:
                color_vals.append([float(c) for c in [r, g, b]])

        return color_vals


    def _get_obj_size(self, row):
        '''
        Get size (volume) of an object
        '''
        # compute size of object based on volume of bounding box
        xlen = float(row["object_bbox_xlength"])
        ylen = float(row["object_bbox_ylength"])
        zlen = float(row["object_bbox_zlength"])

        return xlen * ylen * zlen


    def get_regions(self):
        '''
        Split scene into region and store data by region
        :return:
        '''
        all_regions = {}
        max_num_objs = 0
        scene_ind = 0

        pbar = tqdm(self.all_scenes.items())
        for scene, scene_data in pbar:
            dataset = scene_data["source"]
            regions = scene_data["regions"]
            objects = scene_data["objects"]
            skipped_objs = scene_data["skipped_objs"]
            if self.save_scene_pcs and self.save_region_pcs:
                scene_pc = self.all_scenes[scene]["pc"]

                if self.region_splits_torch:
                    region_ids_and_split = torch.load(os.path.join(self.data_path, dataset, scene, f'{scene}_region_split.pt'))
                    region_split = np.cumsum(region_ids_and_split[:, 1].numpy())[:-1]
                    region_ids = region_ids_and_split[:, 0].numpy()
                else:
                    region_ids_and_split = np.load(os.path.join(self.data_path, dataset, scene, f'{scene}_region_split.npy'))
                    region_split = region_ids_and_split[:-1, 1]
                    region_ids = region_ids_and_split[:, 0]

                
                region_pcs_unnormalized = {region_ids[i]: pc for i, pc in enumerate(np.split(scene_pc, region_split))}

            for region, region_data in regions.items():
                pbar.set_description(f'Scene {scene} - Region {region}')
                # pre_load = time.time()
                if self.save_scene_pcs and self.save_region_pcs:
                    if int(region) not in region_pcs_unnormalized:
                        continue
                    region_pc, num_pts, _, _ = self.get_sampled_pc(
                        region_pcs_unnormalized[int(region)], 
                        None, 
                        None, 
                        int(region), 
                        self.num_region_pts, 
                        region=True, 
                        from_separate_files=True
                    )
                # loaded = time.time()
                #print("pc file load time: ", loaded-pre_load)

                region_objects = {object_id: objects[object_id] for object_id in objects.keys() if
                                  objects[object_id]["region_id"] == region and object_id not in skipped_objs}

                #print("num objs", len(region_objects))
                if len(region_objects) > max_num_objs:
                    max_num_objs = len(region_objects)

                region_data = {
                    "region_name": region_data["region_name"],
                    "scene_id": scene,
                    "objects": region_objects,
                    "statements": []
                }

                if self.save_scene_pcs and self.save_region_pcs:
                    region_data["pc"] = region_pc

                scene_region_id = scene + '_' + str(region)
                all_regions.update({scene_region_id: region_data})

                # for debugging
                #self.visualize_region(region_data)

        #print(all_regions)
        self.max_num_objs = max_num_objs
        #print("MAX OBJS", max_num_objs)

        return all_regions


    def _farthest_point_sample(self, xyz, npoint):
        '''
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        '''
        #device = xyz.device
        batchsize, ndataset, dimension = xyz.shape
        centroids = torch.zeros(batchsize, npoint, dtype=torch.long)
        distance = torch.ones(batchsize, ndataset) * 1e10
        farthest = torch.randint(0, ndataset, (batchsize,), dtype=torch.long)
        batch_indices = torch.arange(batchsize, dtype=torch.long)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(batchsize, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1, dtype=torch.float)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids


    def get_sampled_pc(
            self, 
            pc,
            obj_ids,
            region_ids,
            target_id, 
            num_sample_pts, 
            region=False, 
            from_separate_files=False,
            farthest_point_sampling=False):
        '''
        Sample point cloud with either furthest-point sampling or uniform sampling for given object or region
        '''

        if from_separate_files:
            index = np.arange(len(pc))
        else:
            if region:
                index, = torch.where(region_ids.flatten() == target_id)
                #print("num pts in region", len(index))
            else:
                index, = torch.where(obj_ids.flatten() == target_id)
                #print("num pts in object", len(index))

        if len(index) == 0: # region without points (ignore)
            # print(f"{'region' if region else 'object'} has no points: ", target_id)
            return np.zeros((num_sample_pts if region else len(index), 6)), 0, np.zeros(3)

        if type(index) != type(np.zeros(1)):
            index = index.cpu()
        xyz = pc[index, :3]
        colors = pc[index, 3:]

        axis_aligned_bbox_size = xyz.max(axis=-2) - xyz.min(axis=-2)
        pcd_center = xyz.mean(axis=-2)

        if farthest_point_sampling:
            if num_sample_pts < len(index):
                downsampled_index = self._farthest_point_sample(xyz.unsqueeze(0),
                                                          num_sample_pts)  # downsample the object
                downsampled_index = downsampled_index.flatten()
                points = np.hstack((xyz[downsampled_index, :], colors[downsampled_index, :]))

            else:
                copy_times = int((num_sample_pts - 1) / len(index))
                random_index = torch.randperm(len(xyz))
                random_index = random_index[0:num_sample_pts - copy_times * len(index)]
                new_position = torch.cat((xyz.repeat(copy_times, 1), xyz[random_index]))
                new_color = torch.cat((colors.repeat(copy_times, 1), colors[random_index]))
                points = np.hstack((new_position, new_color))
        else:
            resampled_index = np.random.choice(
                len(index), 
                size=num_sample_pts, 
                replace=(len(index) < num_sample_pts))
            points = np.hstack([xyz[resampled_index, :], colors[resampled_index, :]])
        

        norm_points = self._normalize_pc(points[:, :3])
        pc_out = np.hstack((norm_points, points[:, 3:]))
        pc_out = pc_out.astype(np.float32)

        return pc_out, len(index), pcd_center, axis_aligned_bbox_size


    def get_object_pc_o3d(self, path, target_obj_id, num_sample_pts):
        '''
        Get point cloud of an object based on sampling with open3d
        '''
        pcd = o3d.t.io.read_point_cloud(path)

        object_ids = o3d.core.Tensor.numpy(pcd.point.obj_id)
        index, = np.where(object_ids.flatten() == target_obj_id)
        pcd = pcd.select_by_index(index)

        if len(index) < 0.1*num_sample_pts:
            return []

        pcd_1 = o3d.geometry.PointCloud()
        pcd_1.points = o3d.cpu.pybind.utility.Vector3dVector(o3d.core.Tensor.numpy(pcd.point.positions))
        pcd_1.colors = o3d.cpu.pybind.utility.Vector3dVector(o3d.core.Tensor.numpy(pcd.point.colors) / 255)

        if num_sample_pts < len(index):
            pcd_1 = pcd_1.farthest_point_down_sample(num_sample_pts)  # downsample the object
        else:
            copy_times = int((num_sample_pts - 1) / len(index))
            random_index = np.random.permutation(np.arange(len(pcd_1.points)))
            random_index = random_index[0:num_sample_pts - copy_times * len(index)]
            new_position = np.concatenate(
                (np.tile(np.asarray(pcd_1.points), (copy_times, 1)), np.asarray(pcd_1.points)[random_index]))
            new_color = np.concatenate(
                (np.tile(np.asarray(pcd_1.colors), (copy_times, 1)), np.asarray(pcd_1.colors)[random_index]))

            pcd_1 = o3d.geometry.PointCloud()
            pcd_1.points = o3d.cpu.pybind.utility.Vector3dVector(new_position)
            pcd_1.colors = o3d.cpu.pybind.utility.Vector3dVector(new_color)


        points = np.asarray(pcd_1.points)
        norm_points = self._normalize_pc(points)
        pc = np.hstack((norm_points, np.asarray(pcd_1.colors)))

        return pc


    def load_scenes(self):
        '''
        Load scene info for all scenes in dataset
        '''
        all_scenes = {}
        pbar = tqdm(desc="load scenes", total=len(self.scene_list))

        for dataset in self.datasets:
            dataset_folder = os.path.join(self.data_path, dataset)
            for scene in os.listdir(dataset_folder):
                scene_path = os.path.join(dataset_folder, scene)
                
                if scene in self.scene_list and os.path.isdir(scene_path):
                    scene = Path(scene_path).parts[-1]
                    pbar.set_description(f'Scene {scene}')
                    pc_file = os.path.join(scene_path, scene + '_pc_result.ply')

                    pc, region_ids, obj_ids = self.load_pc(pc_file)
                    
                    regions = self.load_regions(scene_path)
                    
                    objects, skipped_objs = self.load_objects(scene_path, pc, region_ids, obj_ids)

                    scene_data = {
                        "source": dataset,
                        "regions": regions,
                        "objects": objects,
                        "region_ids": region_ids,
                        "obj_ids": obj_ids,
                        "statements": [],
                        "skipped_objs": skipped_objs
                    }
                    if self.save_scene_pcs:
                        scene_data["pc"] = pc
                    all_scenes.update({scene:scene_data})
                    pbar.update()

                    # for debugging
                    #self.visualize_scene(scene_data)
                    # print("TIMES")
                    # print("region load time: ", post_region-start_time)
                    # print("pc load time: ", post_pc-post_region)
                    # print("object load time: ", post_objects-post_pc)

        pbar.close()
        #print(all_scenes)
        return all_scenes


    def load_pc(self, path):
        '''
        Load point cloud from .ply file
        :return: point cloud as numpy array
        '''

        plydata = PlyData.read(path)

        xyz = np.vstack((np.asarray(plydata["vertex"]["x"]), np.asarray(plydata["vertex"]["y"]),
                         np.asarray(plydata["vertex"]["z"]))).transpose()
        xyz = torch.from_numpy(xyz)
        colors = np.vstack((np.asarray(plydata["vertex"]["red"]), np.asarray(plydata["vertex"]["green"]),
                            np.asarray(plydata["vertex"]["blue"]))).transpose()
        colors = torch.from_numpy(colors).float()
        if torch.max(colors) > 1:
            colors = colors / 127.5 - 1

        if 'region_id' in plydata["vertex"]:
            region_ids = torch.from_numpy(np.asarray(plydata["vertex"]["region_id"]).copy().astype(np.int32)).to(self.device)
        else:
            region_ids = None

        if 'obj_id' in plydata["vertex"]:
            obj_ids = torch.from_numpy(np.asarray(plydata["vertex"]["obj_id"]).copy().astype(np.int32)).to(self.device)
        else:
            obj_ids = None
        

        # just for debugging
        #o3d.visualization.draw_geometries([pc])

        # should be Nx6
        pc_array = np.hstack([xyz, colors])

        return pc_array, region_ids, obj_ids


    def _check_skipped_obs(self, data, skipped_objs):
        '''
        Prune statements referring to skipped/ignored objects
        '''
        if data["target_index"] in skipped_objs:
            return True
        for i in data["distractor_ids"]:
            if i in skipped_objs:
                return True
        if data["relation_type"] == "ternary":
            if data["anchors"]["anchor_1"]["index"] in skipped_objs or data["anchors"]["anchor_2"]["index"] in skipped_objs:
                return True
        else:
            if data["anchors"]["anchor_1"]["index"] in skipped_objs:
                return True
        
        return False


    def load_json(self):
        '''
        Load json annotation file with statements and bboxes
        :return:
        '''
        #random_seed = 0
        #torch.manual_seed(random_seed)
        #np.random.seed(random_seed)
        #random.seed(random_seed)

        referential_data = []
        max_distractors = 0
        for dataset in self.datasets:
            dataset_folder = os.path.join(self.data_path, dataset)
            for scene in os.listdir(dataset_folder):
                scene_path = os.path.join(dataset_folder, scene)
                
                if scene in self.scene_list and os.path.isdir(scene_path):
                    scene = Path(scene_path).parts[-1]
                    json_file = os.path.join(scene_path, scene + '_referential_statements.json')
                    skipped_objs = self.all_scenes[scene]["skipped_objs"]

                    # store in a scene name/region name dict?
                    with open(json_file) as f:
                        json_data = json.load(f)
                        num_skipped = 0
                        tot_refs = 0
                        for region, region_data in (json_data["regions"].items()):
                            for utt, data in region_data.items():
                                data = data[0]
                                # check for skipped obj references
                                if len(data) <= 1:
                                    continue
                                tot_refs += 1
                                skip = self._check_skipped_obs(data, skipped_objs)
                                if skip:
                                    num_skipped += 1
                                    continue

                                if len(data["distractor_ids"]) > max_distractors:
                                    max_distractors = len(data["distractor_ids"])

                                ref_data = {
                                    "scene": scene,
                                    "region": region,
                                    "utterance": utt,
                                    "target_label": data["target_class"],
                                    #"target_color": data["target_color_used"],
                                    #"target_size": data["target_size_used"],
                                    "target_obj_id": data["target_index"],
                                    "distractor_ids": data["distractor_ids"],
                                    #"anchor_colors": [data["anchors"][key]["color_used"] for key in data["anchors"]],
                                    #"anchor_sizes": [data["anchors"][key]["size_used"] for key in data["anchors"]],
                                    "relation": data["relation"],
                                    "real": True
                                }
                                if data["relation_type"] == "ternary":
                                    ref_data.update({"anchor_obj_ids": [data["anchors"]["anchor_1"]["index"], data["anchors"]["anchor_2"]["index"]]})
                                else:
                                    ref_data.update({"anchor_obj_ids": [data["anchors"]["anchor_1"]["index"]]})
                                '''if data["relation_type"] == "ternary":
                                    ref_data.update({"anchors": [data["anchors"]["anchor_1"]["class"], data["anchors"]["anchor_2"]["class"]]})
                                else:
                                    ref_data.update({"anchors": [data["anchors"]["anchor_1"]["class"]]})'''
                                
                                referential_data.append(ref_data)

                                if self.load_false_statements:
                                    false_statements = []
                                    for key, val in data["false_statements"].items():
                                        if key == "false_anchors":
                                            for anchor, anchor_data in val.items():
                                                for _, false_statement in anchor_data.items():
                                                    false_statements.append(false_statement)
                                        else:
                                            false_statements.append(val)

                                    if self.balance_false:
                                        false_statements = random.sample(false_statements, 1)
                                    
                                    for false_statement in false_statements:
                                        false_ref_data = ref_data.copy()
                                        false_ref_data["utterance"] = false_statement
                                        false_ref_data["real"] = False
                                        referential_data.append(false_ref_data)

                    print("num skipped for scene {}: {}/{}".format(scene, num_skipped, tot_refs))
        
        self.max_distractors = max_distractors
        # print("MAX DISTRACTORS", max_distractors)
        #print("MAX DISTRACTORS", max_distractors)

        return referential_data


    def load_regions(self, scene_path):
        '''
        Load region information for a given scene
        '''
        regions = {}
        scene = Path(scene_path).parts[-1]
        reg_file = os.path.join(scene_path, scene + '_region_result.csv')
        with open(reg_file, encoding='utf-8') as csv_file:
            csvReader = csv.DictReader(csv_file)

            # for each region
            for row in csvReader:
                region_id = row["region_id"]

                region_info = {
                    "region_name": row["region_label"],
                    "region_bbox": self._get_bbox_coords_heading('region', row),
                }
                regions.update({region_id:region_info})

        return regions


    def load_objects(self, scene_path, pc, region_ids, obj_ids):
        '''
        Load all ground-truth bounding boxes in a region
        '''
        # print(scene_path)
        objects = {}

        # load from object csv
        scene = Path(scene_path).parts[-1]
        object_file = os.path.join(scene_path, scene + '_object_result.csv')
        skipped_objs = []

        obj_pc_dir = os.path.join(scene_path, 'objects')
        if not os.path.isdir(obj_pc_dir):
            os.mkdir(obj_pc_dir)

        if self.region_splits_torch:
            object_ids_and_split = torch.load(os.path.join(scene_path, f'{scene}_object_split.pt'))
            object_split = np.cumsum(object_ids_and_split[:, 1].numpy())[:-1]
            object_ids = object_ids_and_split[:, 0].numpy()
        else:
            object_ids_and_split = np.load(os.path.join(scene_path, f'{scene}_object_split.npy'))
            object_split = object_ids_and_split[:-1, 1]
            object_ids = object_ids_and_split[:, 0]

        object_pcs_unnormalized = {object_ids[i]: pc for i, pc in enumerate(np.split(pc, object_split))}

        with open(object_file, encoding='utf-8') as csv_file:
            csvReader = csv.DictReader(csv_file)

            for row in csvReader:
                object_id = row["object_id"]
                region = row["region_id"]
                obj_pc_resampled, num_pts, pcd_center, aligned_size = self.get_sampled_pc(
                    object_pcs_unnormalized[int(object_id)],
                    None,
                    None,
                    int(object_id),
                    self.num_obj_pts,
                    from_separate_files=True)

                # skip object if invalid
                if len(obj_pc_resampled) == 0 or row["nyu_id"] == '0':
                    skipped_objs.append(object_id)
                    continue

                if row["nyu_label"] in ["wall", "ceiling", "floor"]:
                    skipped_objs.append(object_id)
                    continue          

                # skip object if too small or sparse
                if self.prune_sparse and num_pts < self.sparsity_thresh:
                    skipped_objs.append(object_id)
                    continue

                object_info = {
                    "region_id": region,
                    "object_class": row["nyu_id"],
                    "object_label": row["nyu_label"],
                    "raw_label": row["raw_label"],
                    "color_vals": self._get_color_vals(row),
                    #"color_labels": self._get_color_labels(self._get_color_vals(row)),
                    "object_bbox": self._get_bbox_coords_heading('object', row),
                    "center": [float(row["object_bbox_cx"]), float(row["object_bbox_cy"]),
                               float(row["object_bbox_cz"])],
                    # "center": pcd_center.astype(np.float64),
                    "size": float(self._get_obj_size(row)),
                    "axis_aligned_size": aligned_size.astype(np.float64),
                    "object_pc": obj_pc_resampled
                }
                objects.update({object_id:object_info})
                # debugging
                #self.visualize_object(object_info)

        return objects, skipped_objs


    # for debugging
    def visualize_object(self, object_data):
        '''
        Visualize a specific object
        '''
        pc_array = object_data["object_pc"]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_array[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(pc_array[:, 3:])

        geometries = [pcd]

        o3d.visualization.draw_geometries(geometries)


    # for debugging
    def visualize_region(self, region_data):
        '''
        Visualize a specific region with bboxes around objects
        '''
        pc_array = region_data["pc"]
        objects = region_data["objects"]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_array[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(pc_array[:, 3:])

        geometries = [pcd]
        bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]


        for o_id, o in objects.items():
            bbox = o3d.geometry.LineSet()

            color = [1, 0, 0]
            colors = [color for _ in range(len(bbox_lines))]

            bbox.lines = o3d.utility.Vector2iVector(bbox_lines)
            bbox.colors = o3d.utility.Vector3dVector(colors)
            bbox.points = o3d.utility.Vector3dVector(o["object_bbox"])

            geometries.append(bbox)

        o3d.visualization.draw_geometries(geometries)


    # for debugging
    def visualize_scene(self, scene_data):
        '''
        Visualize an entire scene with bboxes around objects and regions
        '''
        pc_array = scene_data["pc"]
        regions = scene_data["regions"]
        objects = scene_data["objects"]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_array[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(pc_array[:, 3:])

        geometries = [pcd]

        bbox_lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]


        for r_id, r in regions.items():
            bbox = o3d.geometry.LineSet()

            color = [0, 0, 1]
            colors = [color for _ in range(len(bbox_lines))]

            bbox.lines = o3d.utility.Vector2iVector(bbox_lines)
            bbox.colors = o3d.utility.Vector3dVector(colors)
            bbox.points = o3d.utility.Vector3dVector(r["region_bbox"])

            geometries.append(bbox)

        for o_id, o in objects.items():
            bbox = o3d.geometry.LineSet()

            color = [1, 0, 0]
            colors = [color for _ in range(len(bbox_lines))]

            bbox.lines = o3d.utility.Vector2iVector(bbox_lines)
            bbox.colors = o3d.utility.Vector3dVector(colors)
            bbox.points = o3d.utility.Vector3dVector(o["object_bbox"])

            geometries.append(bbox)

        o3d.visualization.draw_geometries(geometries)


    def _get_class_labels(self, objects):
        '''
        Get class labels for all objects in a region
        '''
        classes = []

        for o_id, o in objects.items():
            classes.append(int(o["object_class"]))

        return np.array(classes)


    def _get_raw_labels(self, objects):
        '''
        Get class labels for all objects in a region
        '''
        labels = []

        for o_id, o in objects.items():
            labels.append(o["raw_label"])

        return np.array(labels)


    def _get_box_info(self, objects):
        '''
        Get bbox center and size (volume)
        '''
        box_arr = []
        for o in objects.values():
            center = o["center"]
            vol = o["size"]
            box_info = center + [vol]
            box_arr.append(box_info)

        box_arr = np.array(box_arr, dtype=np.float64)
        return box_arr


    def _normalize_color(self, pc):
        '''
        Normalize color based on mean color
        '''
        color = pc[:, 3:]
        mean_rgb = np.array([109.8, 97.2, 83.8]) / 256
        color *= 0.98 + 0.04 * np.random.random((len(color), 3))
        color -= mean_rgb  # normalize color

        return pc


    def _normalize_pc(self, pcd):
        '''
        Normalize given point cloud to the unit sphere
        '''
        pc = pcd[:, :3]
        center_norm = pc - np.expand_dims(np.mean(pc, axis=0), 0)
        max_dist = np.max(np.sqrt(np.sum(center_norm**2, axis=1)), 0)

        pc = center_norm / max_dist  # scale

        return pc


    def augment_pc(self, pc):
        '''
        Augment the point-cloud: rotation, normalization, scaling? jitter colors and boxes
        :param pc:
        :return:
        '''
        augmentations = {}

        # TODO: add option to pick which ones and adjust depending on split
        # Random noise
        noise = np.random.rand(len(pc), 3) * 5e-3
        augmentations["noise"] = noise
        pc[:, :3] = pc[:, :3] + noise

        # Translation
        augmentations["shift"] = np.random.random((3,))[None, :] - 0.5
        pc[:, :3] += augmentations["shift"]

        # Scale (perturb pc)
        augmentations["scale"] = 0.98 + 0.04 * np.random.random()
        pc[:, :3] *= augmentations["scale"]

        # Color
        self._normalize_color(pc)

        # Flip
        augmentations["yz_flip"] = np.random.random() > 0.5
        if augmentations["yz_flip"]:
            pc[:, 0] = -pc[:, 0]

        augmentations["xz_flip"] = np.random.random() > 0.5
        if augmentations["xz_flip"]:
            pc[:, 1] = -pc[:, 1]

        # Rotation (only for view-independence)

        # TODO: match augmentations for bounding boxes and object pcs

        return pc, augmentations

    def get_subset(self, objects, target_id, anchor_ids):
        '''
        Get subset of objects in a region as "context"
        :return: Dict, subset of objects
        '''
        context_size = self.context_size
        if len(objects) < context_size:
            return objects
        
        distractor_ids = [o_id for o_id in objects.keys() if (o_id != target_id and objects[o_id]["object_class"] == objects[target_id]["object_class"])]
        keep_ids = distractor_ids + [target_id] + anchor_ids

        # print("dist ids", distractor_ids)
        # print("target_id", target_id)
        # print("anchor_ids", anchor_ids)

        if len(keep_ids) < context_size:
            random.shuffle(keep_ids)
            add_ids = [o_id for o_id in objects.keys() if o_id not in keep_ids]
            num_add = context_size - len(keep_ids)
            keep_ids += add_ids[:num_add]
        else:
            distractor_num = context_size - len(anchor_ids) - 1
            distractor_ids = distractor_ids[:distractor_num]
            keep_ids = distractor_ids + [target_id] + anchor_ids
            random.shuffle(keep_ids)

        #print(objects.keys())
        subset_objects = {k: objects[str(k)] for k in keep_ids}

        return subset_objects

    def pad_item(self, ret):
        '''
        Pad relevant data features to max_num_objs length (max objects per region)
        :return:
        '''

        obj_diff = self.max_context_len - len(ret["class_labels"])

        if obj_diff > 0:
            pad_class_labels = np.array([self.pad_idx for _ in range(obj_diff)])
            ret["class_labels"] = np.concatenate([ret["class_labels"], pad_class_labels])
            #ret["raw_labels"] = np.concatenate([ret["raw_labels"], pad_class_labels])

            pad_objects = np.zeros((obj_diff, self.num_obj_pts, 6))
            ret["objects"] = np.concatenate([ret["objects"], pad_objects], axis=0)
            pad_size = np.zeros((obj_diff, 3))
            ret["axis_aligned_size"] = np.concatenate([ret["axis_aligned_size"], pad_size], axis=0)

            target_class_mask = np.array(ret["class_labels"] == ret["target_class"])
            ret["target_class_mask"] = target_class_mask

            pad_box_info = np.zeros((obj_diff, 4))
            ret["box_info"] = np.concatenate([ret["box_info"], pad_box_info], axis=0)

            pad_box_corners = np.zeros((obj_diff, 8, 3))
            ret["box_corners"] = np.concatenate([ret["box_corners"], pad_box_corners], axis=0)

            pad_object_ids = np.array([self.pad_idx for _ in range(obj_diff)]) # NOTE: ensure pad_idx is an invalid object id
            ret["object_ids"] = np.concatenate([ret["object_ids"], pad_object_ids])

        distractor_diff = self.max_distractors - len(ret["distractor_ids"])
        if distractor_diff > 0:
            pad_distractors_pos = np.array([int(self.max_context_len+1) for _ in range(distractor_diff)])
            ret["distractors_pos"] = np.concatenate([ret["distractors_pos"], pad_distractors_pos])
            pad_distractor_ids = np.array([self.pad_idx for _ in range(distractor_diff)])
            ret["distractor_ids"] = np.concatenate([ret["distractor_ids"], pad_distractor_ids])

        # anchors (max 2) # TODO: change to auto detect max anchors
        if len(ret["anchor_ids"]) < 2:
            ret["anchor_ids"] = np.concatenate([ret["anchor_ids"], [self.pad_idx]])

        return ret


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

        # call get_pc/other functions to fetch loaded data and augment
        '''aug_pc, augmentations = self.augment_pc(pc)
        region["pc"] = aug_pc

        # for debugging
        self.visualize_region(region)'''

        # for one sample:
        obj_pcs = np.array([o["object_pc"] for o in objects.values()], dtype=np.float64)
        ret["class_labels"] = self._get_class_labels(objects)

        if self.include_raw_labels:
            ret["raw_labels"] = self._get_raw_labels(objects)

        ret["scan_id"] = region_data["scene_id"]
        ret["context_size"] = len(obj_pcs) # by default, is all objects
        ret["objects"] = obj_pcs
        ret["target_class"] = int(objects[data["target_obj_id"]]["object_class"])
        #ret["target_raw_label"] = objects[data["target_obj_id"]]["raw_label"]

        ret["target_pos"] = int([ind for ind in range(len(objects.keys())) if list(objects.keys())[ind] == data["target_obj_id"]][0])
        ret["target_class_mask"] = np.array(ret["class_labels"] == ret["target_class"])
        ret["tokens"] = data["utterance"] # get rid of punctuation?
        #ret["is_nr3d"] # unused
        ret["box_info"] = self._get_box_info(objects)
        ret["box_corners"] = np.array([o["object_bbox"] for o in objects.values()])
        ret["utterance"] = data["utterance"]
        ret["axis_aligned_size"] = np.array([o["axis_aligned_size"] for o in objects.values()])

        if self.load_false_statements:
            ret["real"] = data["real"]

        # only if visualization flag set (in MVT)
        stimulus_str = region_data["scene_id"] + '-' + str(data["target_label"]) + '-' + str(len(obj_pcs)) + '-' + data["target_obj_id"]
        for id in data["distractor_ids"]:
            stimulus_str += '-' + str(id)
        ret["stimulus_id"] = stimulus_str


        if len(data["distractor_ids"]) == 0:
            ret["distractor_ids"] = np.array([int(self.max_num_objs+1) for _ in range(self.max_distractors)])
            ret["distractors_pos"] = np.array([int(self.max_num_objs+1) for _ in range(self.max_distractors)])
        else:
            # calculate distractor ids in case not all are listed in data
            target_id = data["target_obj_id"]
            ret["distractor_ids"] = np.array([int(o_id) for o_id in objects.keys() if (o_id != target_id and objects[o_id]["object_class"] == objects[target_id]["object_class"])])
            distractor_pos = []

            for id in ret["distractor_ids"]:
                id_pos = [ind for ind in range(len(objects.keys())) if int(list(objects.keys())[ind]) == id][0]
                distractor_pos.append(int(id_pos))

            ret["distractors_pos"] = np.array(distractor_pos)

            '''ret["distractor_ids"] = np.array(data["distractor_ids"])
            distractor_pos = []
            for id in ret["distractor_ids"]:
                id_pos = [ind for ind in range(len(objects.keys())) if int(list(objects.keys())[ind]) == id][0]
                distractor_pos.append(int(id_pos))

            ret["distractors_pos"] = np.array(distractor_pos)'''

        ret["object_ids"] = np.array([int(o) for o in objects.keys()])

        #### TEMP FIX for mismatched obj ids
        for a in range(len(data["anchor_obj_ids"])):
            i = 0
            for o_id, o in objects.items():
                if i == data["anchor_obj_ids"][a]:
                    data["anchor_obj_ids"][a] = o_id
                    break
                i += 1
        # this can be obtained w anchor_obj_ids
        # ret["anchor_bboxes"] = np.array([objects[i]["object_bbox"] for i in data["anchor_obj_ids"]])

        # add padding
        ret = self.pad_item(ret)
        #print("ret", ret)

        return ret