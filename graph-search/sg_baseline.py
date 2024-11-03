'''
Copyright 2024: Haochen Zhang
'''

import json
import os
import csv
from colour import Color
from textblob import TextBlob
from collections import defaultdict
import argparse
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd

from scene_graph import *
from utils import *
from data.dataloader import VLADataset
from data.referit3d_loader import ReferIt3DDataset
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

class SGDataset(VLADataset):
    def __init__(self, data_path='./', split='train', metadata_file='metadata.json', load_false=True):

        self.data_path = data_path
        self.split = split
        self.metadata_file = metadata_file
        self.load_false = load_false
        self.balance_false = True

        # read metadata
        self.read_metadata()

        # take in file listing scenes for split
        scene_file = os.path.join(data_path, split + '.txt')
        self.scene_list = []

        with open(scene_file) as f:
            for line in f:
                # skip empty lines
                if not line or line == '\n':
                    continue
                self.scene_list.append(line.rstrip())
        
        self.all_scenes = {}
        for scene in self.scene_list:
            self.all_scenes.update({scene:{"skipped_objs":[]}})

        ##### temp
        csvfile = 'NYUv2_ChatGPT_Affordances.csv'
        self.aff_df = pd.read_csv(csvfile)
        #####

        self.class_2_aff = self.map_class_aff()

        print("Loading scene graphs...")
        self.scene_graphs, self.class_groups, self.aff_groups = self.load_scene_graphs(data_path, split)

        print("Loading statements...")
        self.referential_data = self.load_json()

        # loading language data from referit3d
        #print("Loading referential data...")
        #self.referential_data = self.read_refer_csv('sr3d.csv', self.scene_list)

        self.scene_refs = self.group_refs()

    def __len__(self):
        return len(self.referential_data)
    
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
    
    def map_class_aff(self):
        class_2_aff = {}
        
        for ind, row in self.aff_df.iterrows():
            class_2_aff.update({row['nyuClass'].lower().lstrip():row['affordance'].split(',')})
        
        return class_2_aff
    
    def _group_by_class(self, region):
        # same as getting distractors
        class_2_id = {}
        objects = region["objects"]
        for o in objects:
            nyu_id = o["nyu_id"]
            if nyu_id not in class_2_id.keys():
                class_2_id[nyu_id] = [o["object_id"]]
            else:
                class_2_id[nyu_id].append(o["object_id"])
        
        return class_2_id


    def _group_by_affordance(self, region):
        aff_2_id = {}
        objects = region["objects"]
        for o in objects:
            # in case affordances not in scene graph
            if "affordances" not in o.keys():
                aff_list = []
            else:
                aff_list = o["affordances"]
            nyu_id = o["nyu_id"]
            nyu_class = o["nyu_label"]
            #aff = self.aff_df.loc[self.aff_df['nyuId'] == int(nyu_id), 'affordance'].item()
            for aff in aff_list:
                if aff not in aff_2_id.keys():
                    aff_2_id[nyu_class] = [o["object_id"]]
                else:
                    aff_2_id[nyu_class].append(o["object_id"])
        
        return aff_2_id
    

    def group_refs(self):
        scene_refs = {}
        for ref in self.referential_data:
            if ref["real"]:
                scene_region_id = ref["scene"] + '_' + ref["region"]
                utt = ref["utterance"]
                if scene_region_id in scene_refs.keys():
                    scene_refs[scene_region_id].append(ref)
                else:
                    scene_refs.update({scene_region_id:[ref]})
        
        return scene_refs
    

    # take in info from scene graph json
    def create_graph(self, region):
        graph = SceneGraph()
        
        for obj in region["objects"]:
            nyu_id = obj["nyu_id"]
            #affordance = df.loc[df['nyuId'] == int(nyu_id), 'affordance'].item()
            if "affordances" not in obj.keys():
                affordance = ['']
            else:
                affordance = obj["affordances"]
            obj_node = Node(obj["nyu_label"], obj["color_labels"], size_val=obj["volume"], class_id=obj["nyu_id"], affordance=affordance, idx=obj["object_id"], raw_label=obj["raw_label"])
            graph.nodes[obj["object_id"]] = obj_node
        
        print("Number of nodes: ", len(graph.nodes))
        for rel in region["relationships"]:
            rels = region["relationships"][rel]
            if rel == "between":
                for target, anchor_list in rels.items():
                    for anchor in anchor_list:
                        graph.add(graph.nodes[target], [graph.nodes[a] for a in anchor], rel)
            else:
                if '_' in rel:
                    rel = ' '.join(rel.split('_'))
                
                for anchor, target_list in rels.items():
                    for target in target_list:
                        graph.add(graph.nodes[target], [graph.nodes[anchor]], rel)

        return graph


    def load_scene_graphs(self, data_path, split):
        # take in file listing scenes for split
        scene_graphs = {}
        class_groups = {}
        aff_groups = {}

        for dataset in self.datasets:
            dataset_folder = os.path.join(self.data_path, dataset)
            for scene in os.listdir(dataset_folder):
                scene_path = os.path.join(dataset_folder, scene)
                if scene in self.scene_list and os.path.isdir(scene_path):
                    scene = Path(scene_path).parts[-1]
                    sg_file = os.path.join(scene_path, scene + '_scene_graph.json')
                    
                    with open(sg_file) as f:
                        scene_data = json.load(f)
                        for region, region_data in (scene_data["regions"].items()):
                            scene_graph = self.create_graph(region_data)
                            scene_region_id = scene + '_' + region
                            scene_graphs.update({scene_region_id:scene_graph})
                            
                            #check_statements(args, scene, region, scene_graph)

                            # pre-grouping for alternatives
                            class_2_id = self._group_by_class(region_data)
                            aff_2_id = self._group_by_affordance(region_data)
                            class_groups.update({scene_region_id:class_2_id})
                            aff_groups.update({scene_region_id:aff_2_id})


        return scene_graphs, class_groups, aff_groups

    
    def __getitem__(self, idx):
        data = self.referential_data[idx]
        scene = data["scene"]
        region = data["region"]
        scene_region_id = scene + '_' + region
        graph = self.scene_graphs[scene_region_id]
        data.update({"scene_graph":[graph]})

        return data


class SGSearcher():
    def __init__(self, args, class_2_aff):
        self.data_path = ''
        self.class_2_aff = class_2_aff

        print("Loading llm outputs for SGSearcher...")
        self.llm_outputs = self.load_llm_output(args.llm_output_file)

    def load_llm_output(self, llm_output_file):
        with open(llm_output_file, 'r') as f:
            outputs = json.load(f)
        
        return outputs

    def _get_tags(self, text):
        blob = TextBlob(text)
        return blob.tags

    def _is_color(self, word):
        try: 
            Color(word)
            return True
        except ValueError:
            return False

    def parse_output(self, scene, region, utt):
        try:
            output = self.llm_outputs[scene+'_'+region][utt]
            if not output:
                return None
            # post-processing here?
        except KeyError:
            return "ignore"
        
        if not output: 
            return None
        anchor_objs = output["anchors"]

        # get relation without filler words
        relation_tags = self._get_tags(output["relation"])
        relation_words = []
        for word, tag in relation_tags:
            if tag != 'TO':
                relation_words.append(word)
        relation = " ".join(relation_words)

        # create nodes
        try:
            target_aff = self.class_2_aff[output['target'].lower()]
        except KeyError:
            target_aff = []
        
        target_node = Node(output["target"], output["target_color"], size=output["target_size"])
        anchor_nodes = []
        for i in range(len(anchor_objs)):
            name = 'anchor_' + str(i+1)
            try:
                anchor_aff = self.class_2_aff[anchor_objs[i].lower()]
            except KeyError:
                anchor_aff = []
            if i < len(output["anchor_colors"]) and i < len(output["anchor_sizes"]):
                anchor_node = Node(anchor_objs[i], output["anchor_colors"][i], size=output["anchor_sizes"][i])
                anchor_nodes.append(anchor_node)

        # create edge/subgraph to search for
        edge = Edge(target_node, anchor_nodes, relation)

        return edge 


    def forward(self, data):
        utt = data["utterance"]
        graph = data["scene_graph"][0]
        scene = data["scene"]
        region = data["region"]

        # parse statement into subgraph with llm
        edge = self.parse_output(scene, region, utt)
        if edge == "ignore":
            return False, "ignore"

        if not edge:
            return False, None

        # search through scene graph for match
        found, node = graph.search(edge)

        return found, node
    

    def get_candidates(self, scene_region_id, parsed_output, scene_refs):
        classes = parsed_output["anchors"] + [parsed_output["target"]]
        all_scene_statements = scene_refs[scene_region_id]
        candidates = []
        # go through existing candidates in scene
        for data in all_scene_statements:
            keep = False
            # filter candidates by classes mentioned
            if data["target_label"] == parsed_output["target"]:
                keep = True
            if set(data["anchors"]) == set(parsed_output["anchors"]):
                keep = True
            if keep:
                candidates.append(data)
        
        #print("cand", candidates)
        #print(len(candidates))

        return candidates
    
    
    def llm_request(self, utt, candidates):
        prompt_prefix = """I will give you a sentence of an object I'm looking for. Then I will give you a list of sentences describing what objects are actually available. Please give me the sentence closest to what I'm looking for. Return ONLY the sentence number. \n Here is an example:

    Sentence: the cabinet near the blue guitar
    
    Choices:
    1 the cabinet farthest from the window
    2 the bookshelf between the desk and guitar
    3 the brown stool near the big night stand
    4 the cabinet near the brown guitar

    Answer: 4
        """
        prompt = prompt_prefix + '\nSentence: ' + utt
        for i in range(1, len(candidates)+1):
            prompt += '\n' + str(i) + ' ' + candidates[i-1]
        
        prompt += '\nAnswer: '
        #print(prompt)

        tries = 5
        while True:
            tries -= 1
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                break
            except (ConnectionError) as err:
                if tries == 0:
                    raise err
                else:
                    time.sleep(5)
        
        output = response.choices[0].message.content
        print("llm output: ", output)
        
        return response.choices[0].message.content


    def score_alternative(self, ref, alternative):
        num_sim = 0
        total_aspects = 0
        priority_keys = ["target", "anchors", "relation"]
        alternative.update({"target":alternative["target_label"]})
        #print("ref", ref)
        #print("chosen: ", alternative)
        
        # weighted sum based on number and type of matching aspects
        for key, val in ref.items():
            if key == 'scene_graph':
                continue

            if key in priority_keys:
                weight = 2
            else:
                weight = 1

            if type(val) == str:
                if val != "":
                    total_aspects += weight*1
                    if val == alternative[key]:
                        num_sim += weight*1
            elif type(val) == type([]):
                empty = True
                for i in val:
                    if i != "":
                        empty = False
                        total_aspects += weight*1
                if set(val) == set(alternative[key]) and not empty:
                    num_sim += weight*len(val)
        
        print("num sim", num_sim)
        print("tot ", total_aspects)
        
        return num_sim/total_aspects


    def choose_alternative(self, ref, scene_refs):
        scene_region_id = ref["scene"] + '_' + ref["region"]
        utt = ref["utterance"]
        # get llm output
        parsed_output = self.llm_outputs[scene_region_id][utt]
        candidates = self.get_candidates(scene_region_id, parsed_output, scene_refs)
        cand_statements = [c["utterance"] for c in candidates]
        #print(cand_statements)
        scores = []
        # get score for chosen alternative
        if len(candidates) > 0:
            idx = self.llm_request(ref["utterance"], cand_statements)
        
            answer = candidates[int(idx)-1]
            score = self.score_alternative(parsed_output, answer)
            #print("score", score)
            scores.append(score)
        
            return score
        
        return None
    

def eval(args):
    real_corr = 0
    tot_real = 0
    tot_fake = 0
    tot = 0
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    scores = []
    no_alt = 0

    print("starting eval")
    dataset = SGDataset(args.data_path, split=args.split, load_false=True)
    scene_graphs = dataset.scene_graphs
    class_groups = dataset.class_groups
    aff_groups = dataset.aff_groups
    class_2_aff = dataset.class_2_aff
    scene_refs = dataset.scene_refs
    #test_loader = dataset_to_dataloader(dataset, args, False)

    model = SGSearcher(args, class_2_aff)

    # loop over utterances
    i = 0
    for ref in dataset:
        if i % 10 == 0:
            #print("ref", ref)
            utt = ref["utterance"]
            target_idx = ref["target_obj_id"] # negative if statement false
            real = ref["real"]

            found, node = model.forward(ref)
            if not found:
                if type(node) == type("ignore"):
                    continue

            if found and int(target_idx) >= 0: # true positive
                print("Target found", node.idx, target_idx)
                if node.idx == target_idx:
                    tp += 1
            elif not found and int(target_idx) < 0: # true negative
                tn += 1
                score = model.choose_alternative(ref, scene_refs)
                if score != None:
                    scores.append(score)
                else:
                    no_alt += 1
            elif found and int(target_idx) < 0: # false positive
                fp += 1
            elif not found and int(target_idx) >= 0: # false negative
                fn += 1
            
            tot += 1
            i += 10

    
    print("Total average score for alternatives: ", sum(scores)/len(scores))
    print("correct (TP): ", tp)
    print("TN: ", tn)
    print("FP: ", fp)
    print("FN: ", fn)
    print("Total statements: ", tot)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='VLA_Dataset', help='path to dataset source')
    parser.add_argument('--llm_output_file', help='path to json file output from LLM')
    parser.add_argument('--output_path', default='')
    parser.add_argument('--split', default='test')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size during training')
    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()
    
    eval(args)
    