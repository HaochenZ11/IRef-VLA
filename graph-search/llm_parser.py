'''
Copyright 2024: Haochen Zhang
'''

import json
import os
import re
import csv
import ast
import argparse
from pathlib import Path
import random
import time
from openai import OpenAI


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# few-shot prompt prefix
PROMPT_PREFIX = '''
I will give a sentence, please give me the target object, anchor object(s), and relation. Also give me the attributes (color or size) for each target and anchor object, if any. Make sure colors are valid and size should be big, small, bigger, or smaller. Relations should only be one of: between, near, in, on, above, below, closest, farthest, second closest, second farthest, third closest, third farthest. Give the output in json format and only one response per sentence. Only use the specified relations.
Here are some examples:
Sentence: "the blue chair closest to the refridgerator"
{
target: 'chair',
target_color: 'blue',
target_size: '',
anchors: ['refridgerator'],
anchor_colors: [''],
anchor_sizes = ['']
relation: 'closest'
}

Sentence: the bed between the bigger nightstand and the wall
{
target: 'bed',
target_color: '',
target_size: ''
anchors: ['nightstand', 'wall'],
anchor_colors: ['', ''],
anchor_sizes: ['bigger', ''],
relation: 'between'
}

Sentence: the small lamp in the middle of the potted plant and other lamp
{
target: 'lamp',
target_color: '',
target_size: 'small',
anchors: ['potted plant', 'lamp'],
anchor_colors: ['', ''],
anchor_sizes: ['', ''],
relation: 'between'
}

Sentence: choose the desk with a laptop on its top
{
target: 'desk',
target_color: '',
target_size: '',
anchors: ['laptop'],
anchor_colors: [''],
anchor_sizes: [''],
relation: 'below'
}
'''

def call_llm(utt):
    '''
    Send llm request
    '''
    text = PROMPT_PREFIX + '\n\nSentence: ' + utt
    #print(text)
    tries = 5
    while True:
        tries -= 1
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": text}
                ]
            )
            break
        except (ConnectionError) as err:
            if tries == 0:
                raise err
            else:
                time.sleep(10)

    print("response: ", response)

    #print("parsed json", json_response)

    return response.choices[0].message.content


def get_statements(args):
    '''
    Load referential statements from IRef-VLA
    '''
    # take in file listing scenes for split
    scene_file = os.path.join(args.data_path, args.split + '.txt')
    metadata_file = os.path.join(args.data_path, 'metadata.json')
    scene_list = []
    datasets = []
    load_false = args.load_false
    balance_false = True

    # set random seed
    random_seed = 0
    random.seed(random_seed)

    with open(scene_file) as f:
        for line in f:
            # skip empty lines
            if not line or line == '\n':
                continue
            scene_list.append(line.rstrip())
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    datasets = metadata["datasets"]

    statements = []
    referential_data = []
    tot_statements = 0
    # per scene basis
    for dataset in datasets:
        dataset_folder = os.path.join(args.data_path, dataset)
        for scene in os.listdir(dataset_folder):
            scene_path = os.path.join(dataset_folder, scene)
            
            if scene in scene_list and os.path.isdir(scene_path):
                scene = Path(scene_path).parts[-1]
                json_file = os.path.join(scene_path, scene + '_referential_statements.json')
                # read referential statement files
                with open(json_file) as f:
                    json_data = json.load(f)
                    for region, region_data in (json_data["regions"].items()):
                        for utt, data in region_data.items():
                            data = data[0]

                            if len(data) <= 1:
                                continue
                            
                            ref_data = {
                                "scene": scene,
                                "region": region,
                                "utterance": utt,
                                "target_label": data["target_class"],
                                "target_obj_id": data["target_index"],
                                "distractor_ids": data["distractor_ids"],
                                "relation": data["relation_type"],
                                "real": True
                            }
                            tot_statements += 1
                            statements.append(utt)
                            # false statement loading
                            if load_false:
                                false_statements = []
                                for key, val in data["false_statements"].items():
                                    if key == "false_anchors":
                                        for anchor, anchor_data in val.items():
                                            for _, false_statement in anchor_data.items():
                                                false_statements.append(false_statement)
                                    else:
                                        false_statements.append(val)

                                if balance_false:
                                    false_statements = random.sample(false_statements, 1)
                                
                                for false_statement in false_statements:
                                    false_ref_data = ref_data.copy()
                                    false_ref_data["utterance"] = false_statement
                                    false_ref_data["real"] = False
                                    false_ref_data["target_obj_id"] = -1
                                    referential_data.append(false_ref_data)
    
    return statements, referential_data


def get_refer_statements(args):
    '''
    Load language data from referit3d csv
    '''
    refer_data = []
    # only test statements (previously v2, scannetv2_val.txt)
    scene_file = os.path.join(args.data_path, args.split+'.txt')
    scene_list = []
    with open(scene_file) as f:
        for line in f:
            # skip empty lines
            if not line or line == '\n':
                continue
            scene_list.append(line.rstrip())
    
        csv_path = 'sr3d.csv'
        with open(csv_path, encoding='utf-8') as csv_file:
            csvReader = csv.DictReader(csv_file)
            for row in csvReader:
                if row["scan_id"] in scene_list: #and row["coarse_reference_type"] != "allocentric":
                    sample = {
                        "scene": row["scan_id"],
                        "region": "0",
                        "utterance": row["utterance"],
                        #"target_label"
                        "target_obj_id": row["target_id"],
                        "distractor_ids": ast.literal_eval(row["distractor_ids"]),
                        "relation": row["reference_type"],
                        "anchor_obj_ids": ast.literal_eval(row["anchor_ids"])
                    }
                    refer_data.append(row["utterance"])
        
        return refer_data


def process_statements(args):
    '''
    Send datasets through llm to parse and save outputs
    '''
    statements, referential_data = get_statements(args)
    #statements = get_refer_statements(args)

    results_dict = {}
    with open(args.output_file, 'w') as file:
        json.dump(results_dict, file, indent=4)
    
    start = 0
    for i in range(start, len(referential_data)):
        s = referential_data[i]["utterance"]
        scene_region_id = referential_data[i]["scene"] + '_' + referential_data[i]["region"]
        print(i)
        print("statement: ", s)

        # get llm parsing
        output = call_llm(s)
        with open(args.output_file, 'r+') as file:
            # write output to file
            data = json.load(file)
            if scene_region_id in data.keys():
                if s not in data[scene_region_id].keys():
                    data[scene_region_id].update({s:output})
            else:
                data.update({scene_region_id:{s:output}})
            #data.update({s:output})
            file.seek(0)
            json.dump(data, file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', help-'path to dataset')
    parser.add_argument('--output_file', default='', help='output json file to save llm requests to')
    parser.add_argument('--split', default='test')
    parser.add_argument('--load_false', action='store_true', default='True', help='whether to load false statements')

    args = parser.parse_args()
    
    process_statements(args)