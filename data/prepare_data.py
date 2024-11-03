import os
import torch
import json
import argparse
from pathlib import Path
# import pandas as pd
from utils import * 

def split_data_separate(args):
    '''
    Split scenes into train and text sets stored in text files for each of the datasets
    '''
    # get list of all scenes
    for source in os.listdir(args.data_path):
        print(source)

        scenes = []

        data_folder = os.path.join(args.data_path, source)
        if os.path.isdir(data_folder):
            print(data_folder)
        
            for scene in os.listdir(data_folder):
                scene_path = os.path.join(data_folder, scene)
                if os.path.isdir(scene_path):
                    scenes.append(scene)
        else:
            continue
    
        
        # split scenes based on train/test size
        train_size = int(args.train_size * len(scenes))
        if len(scenes) == 1:
            train_size = 1
            
        test_size = len(scenes) - train_size
        train_scenes, test_scenes = torch.utils.data.random_split(scenes, [train_size, test_size])
        train_scenes = list(train_scenes)
        test_scenes = list(test_scenes)

        # save to files
        train_file = os.path.join(args.data_path, f'{source.lower()}_train.txt')
        test_file = os.path.join(args.data_path, f'{source.lower()}_test.txt')

        with open(train_file, 'w') as f:
            for scene in train_scenes:
                f.write(scene + '\n')

        with open(test_file, 'w') as f:
            for scene in test_scenes:
                f.write(scene + '\n')

    return train_scenes, test_scenes


def split_data(args):
    '''
    Split scenes into train and text sets stored in text files
    '''
    scenes = []
    # get list of all scenes
    for source in os.listdir(args.data_path):
        data_folder = os.path.join(args.data_path, source)
        if os.path.isdir(data_folder):
        
            for scene in os.listdir(data_folder):
                scene_path = os.path.join(data_folder, scene)
                if os.path.isdir(scene_path):
                    scenes.append(scene)
    
    print(scenes)
    
    # split scenes based on train/test size
    train_size = int(args.train_size * len(scenes))
    if len(scenes) == 1:
        train_size = 1
        
    test_size = len(scenes) - train_size
    train_scenes, test_scenes = torch.utils.data.random_split(scenes, [train_size, test_size])
    train_scenes = list(train_scenes)
    test_scenes = list(test_scenes)

    # save to files
    train_file = os.path.join(args.data_path, 'train.txt')
    test_file = os.path.join(args.data_path, 'test.txt')

    with open(train_file, 'w') as f:
        for scene in train_scenes:
            f.write(scene + '\n')

    with open(test_file, 'w') as f:
        for scene in test_scenes:
            f.write(scene + '\n')

    return train_scenes, test_scenes


# TODO: add more metadata for max objects in a region, etc.
def save_metadata(args):
    '''
    Save metadata in a json file
    '''
    metadata = {}
    idx_to_class = get_nyu_classes(args.class_file, args.pad_idx)

    datasets = []
    all_classes = []
    max_obj_per_region = 0
    min_obj_per_region = np.inf
    max_distractors = 0
    max_obj_region_scene = None
    min_obj_region_scene = None
    max_distractor_scene = None
    num_objects = 0
    num_scenes = 0
    num_regions = 0
    scenes = []
    num_scene_objects = []
    for source in os.listdir(args.data_path):
        data_folder = os.path.join(args.data_path, source)
        if os.path.isdir(data_folder):
            print("source: ", source)
            datasets.append(source)
        
            for scene in os.listdir(data_folder):
                scene_path = os.path.join(data_folder, scene)

                if os.path.isdir(scene_path):
                    scene = Path(scene_path).parts[-1]
                    scenes.append(scene)
                    object_file = os.path.join(scene_path, scene + '_object_result.csv')
                    df = pd.read_csv(object_file)

                    num_obj = df['object_id'].count()
                    regions = df['region_id'].nunique()
                    num_regions += regions

                    # get object classes
                    classes = list(df['nyu_id'])
                    classes = [x for x in classes if x != 0]
                    all_classes += classes
                    num_objects += len(classes)

                    df = df.drop(df[df['nyu_label'] == 'unknown'].index)
                    region_counts = df['region_id'].value_counts()
                    max_num = region_counts.iloc[0]
                    min_num = region_counts.iloc[-1]

                    # count max distractors in a region
                    region_class_counts = df.groupby(['region_id'])['nyu_id'].value_counts()
                    num_distractors = region_class_counts.iloc[0]
                    if num_distractors > max_distractors:
                        max_distractors = int(num_distractors)
                        max_distractor_scene = scene

                    # get min and max number of objects per region
                    if max_num > max_obj_per_region:
                        max_obj_per_region = int(max_num)
                        max_obj_region_scene = scene
                    
                    if min_num < min_obj_per_region:
                        min_obj_per_region = int(min_num)
                        min_obj_region_scene = scene
                    
                    num_scene_objects.append(int(num_obj))

                    num_scenes += 1
        
    # get min and max number of objects in a scene
    max_obj_per_scene = int(np.max(num_scene_objects))
    min_obj_per_scene = int(np.min(num_scene_objects))
    max_obj_scene = scenes[np.argmax(num_scene_objects)]
    min_obj_scene = scenes[np.argmin(num_scene_objects)]

    metadata.update({'datasets':datasets})
    metadata.update({'total_scenes':num_scenes})
    metadata.update({'total_regions':num_regions})
    metadata.update({'total_objects':num_objects})
    metadata.update({'max_distractors':max_distractors})
    metadata.update({'max_distractor_scene':max_distractor_scene})
    metadata.update({'max_obj_per_scene':max_obj_per_scene})
    metadata.update({'min_obj_per_scene':min_obj_per_scene})
    metadata.update({'max_obj_per_region':max_obj_per_region})
    metadata.update({'min_obj_per_region':min_obj_per_region})
    metadata.update({'max_obj_scene':max_obj_scene})
    metadata.update({'min_obj_scene':min_obj_scene})
    metadata.update({'max_obj_region_scene':max_obj_region_scene})
    metadata.update({'min_obj_region_scene':min_obj_region_scene})

    all_classes = list(set(all_classes))
    metadata.update({'class_ids':all_classes})

    n_classes = len(all_classes) # does not include pad class
    class_name_list = [idx_to_class[int(idx)] for idx in all_classes]
    metadata.update({'class_names':class_name_list})

    metadata_file = os.path.join(args.data_path, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='data/')
    parser.add_argument('--train_size', default=0.8)
    parser.add_argument('--class_file', default='NYU_Object_Classes.csv')
    parser.add_argument('--pad_idx', default=900)
    parser.add_argument('--split_separately', default=False, action='store_true')

    args = parser.parse_args()

    if args.split_separately:
        split_data_separate(args)
    else:
        split_data(args)
    # save_metadata(args)