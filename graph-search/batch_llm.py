import json
import os
import math
import csv
import ast
import argparse
from pathlib import Path
import openai
from openai import OpenAI
import google.generativeai as genai

from llm_parser import *


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# few-shot prompt prefix
PROMPT_PREFIX = '''
I will give a sentence, please give me the target object, anchor object(s), and relation. Also give me the attributes (color or size) for each target and anchor object, if any. Relations should only be one of: between, near, in, on, above, below, closest, farthest, second closest, second farthest, third closest, third farthest. Give the output in json format and only one response per sentence. Only use the specified relations.
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

def upload_file(args):
    '''
    Upload batch request file to client
    '''
    batch_input_file = client.files.create(
        file=open(args.batch_file_name, "rb"),
        purpose="batch"
    )
    return batch_input_file


def send_batch(batch_input_file):
    '''
    Send batch request to API
    '''
    batch_input_file_id = batch_input_file.id

    metadata = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "referit3d sr3d test"
        }
    )

    #print(metadata)
    return metadata


def create_batch_file(args, statements):
    '''
    Generate batch request file for OpenAI API
    '''
    max_num = 5000 # token queue limit / 400 tokens 
    i = 0
    num_files = math.ceil(len(statements) / max_num)

    for n in range(0, num_files):
        file_name = args.batch_file_name + '_' + str(n) + '.jsonl'
        start_idx = n*max_num
        end_idx = min(start_idx+max_num, len(statements)-1)
        subset = statements[start_idx:end_idx]
        #print(file_name)
        #print(start_idx, end_idx)
        
        with open(file_name, 'w') as file:
            for utt in subset:
                text = PROMPT_PREFIX + '\n\nSentence: ' + utt
                message_id = str(i)
                message_dict = {
                    "custom_id": message_id, 
                    "method": "POST", "url": "/v1/chat/completions", 
                    "body": {
                        "model": "gpt-4o-mini", 
                        "messages": [{"role": "user", "content": text}],
                        "max_tokens": 65
                    }
                }
                i += 1 
                json.dump(message_dict, file)
                file.write('\n')
        


def process_statements(args):
    '''
    Get statements and send in batches
    '''
    statements = get_refer_statements(args)
    create_batch_file(args, statements)
    batch_input_file = upload_file(args)
    metadata = send_batch(batch_input_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path')
    parser.add_argument('--batch_file_name', default='batch')
    parser.add_argument('--output_file', default='')
    parser.add_argument('--split', default='test')

    args = parser.parse_args()
    
    process_statements(args)