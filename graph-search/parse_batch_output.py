'''
Copyright 2024: Haochen Zhang
Script to parse LLM output files into JSON format
'''

import json
import os
import argparse
import re
import csv
import ast


# referenced: https://www.reddit.com/r/Bard/comments/18mmszg/comment/kz99su9/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
def parse_json_from_str(json_str: str):
    """
    Parses JSON-like string into dict object
    """

    try:
        # Remove potential leading/trailing whitespace
        json_str = json_str.strip()

        # Extract JSON content from triple backticks and "json" language specifier
        json_match = re.search(r"```json\s*(.*?)\s*```", json_str, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)

        return json.loads(json_str)
    except (json.JSONDecodeError, AttributeError):
        print("Cannot decode JSON: ")
        print(json_str)
        return None


def get_refer_statements(args):
    '''
    Load language data from referit3d csv
    '''
    refer_data = []
    id_2_data = {}
    # ONLY test statements (previously v2, scannetv2_val.txt)
    scene_file = os.path.join(args.data_path, 'referit3d_test.txt')
    scene_list = []

    with open(scene_file) as f:
        for line in f:
            # skip empty lines
            if not line or line == '\n':
                continue
            scene_list.append(line.rstrip())
    
        csv_path = 'sr3d.csv' # modify csv path
        i = 0
        with open(csv_path, encoding='utf-8') as csv_file:
            csvReader = csv.DictReader(csv_file)
            for row in csvReader:
                if row["scan_id"] in scene_list:
                    sample = {
                        "scene": row["scan_id"],
                        "region": "0",
                        "utterance": row["utterance"],
                        "target_obj_id": row["target_id"],
                        "distractor_ids": ast.literal_eval(row["distractor_ids"]),
                        "relation": row["reference_type"],
                        "anchor_obj_ids": ast.literal_eval(row["anchor_ids"])
                    }
                    id_2_data.update({str(i):sample})
                    i += 1
        
        print(id_2_data)

        return id_2_data


def read_files(args):
    id_2_data = get_refer_statements(args)
    id_2_input = {}
    scene_utt_output = {}

    with open(args.batch_file, 'r') as file:
        input_list = list(file)
    
    for json_str in input_list:
        input_dict = json.loads(json_str)
        text = input_dict["body"]["messages"][0]["content"]
        #print(text)
        last_line = text.splitlines()[-1]
        utt = last_line.split('Sentence: ')[-1]
        #print(utt)
        id_2_input.update({input_dict["custom_id"]:utt})
    
    with open(args.llm_output_file, 'r') as file:
        output_list = list(file)

    for json_str in output_list:
        response_dict = json.loads(json_str)
        output = response_dict["response"]["body"]["choices"][0]["message"]["content"]
        output_dict = parse_json_from_str(output)
        sample = id_2_data[response_dict["custom_id"]]
        scene = sample["scene"]
        utt = sample["utterance"]
        if scene in scene_utt_output.keys():
            if utt in scene_utt_output[scene].keys():
                print(utt)
            scene_utt_output[scene].update({utt:output_dict})
        else:
            scene_utt_output.update({scene:{utt:output_dict}})
    
    # write parsed json output to file
    with open(args.output_file, 'w') as file:
        json.dump(scene_utt_output, file, indent=4)
    


def read_files_iref(args):
    """
    Read LLM batch output files and process string output to JSON
    """
    scene_utt_output = {}

    with open(args.llm_output_file, 'r') as file:
        result_dict = json.load(file)
    
    for scene, request in result_dict.items():
        for utt, text in request.items():
            output_dict = parse_json_from_str(text)

            if scene in scene_utt_output.keys():
                if utt in scene_utt_output[scene].keys():
                    print(utt)
                scene_utt_output[scene].update({utt:output_dict})
            else:
                scene_utt_output.update({scene:{utt:output_dict}})
    
    with open(args.output_file, 'w') as file:
        json.dump(scene_utt_output, file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='')
    parser.add_argument('--llm_output_file')
    parser.add_argument('--batch_file', help='input request file', default='')
    parser.add_argument('--output_file')
    parser.add_argument('--split', default='test')

    args = parser.parse_args()
    
    #read_files(args)
    read_files_iref(args)
