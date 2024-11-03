import random
import os

def combine_splits(input_path: str, splits: list, output_name: str, seed: int):

    all_scenes = []

    for split in splits:
        with open(os.path.join(input_path, f'{split}.txt')) as f:
            for line in f:
                # skip empty lines
                if not line or line == '\n':
                    continue
                all_scenes.append(line.rstrip())
            
    random.Random(seed).shuffle(all_scenes)

    with open(os.path.join(input_path, f'{output_name}.txt'), 'w') as f:
        for scene in all_scenes:
            f.write(scene + '\n')

if __name__ == "__main__":

    # Arguments:

    input_path = '../VLA_Dataset_v3'

    splits = [
        'referit3d',
        '3rscan',
        'arkitscenes',
        'matterport',
        'hm3d',
        'unity'
    ]

    suffix = 'test'

    output_name = 'full'

    seed = 4

    #####

    splits = [f'{split}_{suffix}' for split in splits]

    combine_splits(input_path, splits, f'{output_name}_{suffix}', seed)