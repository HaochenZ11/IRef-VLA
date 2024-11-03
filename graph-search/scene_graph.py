'''
Copyright 2024: Haochen Zhang
'''

import csv
import json
import os
from collections import defaultdict
import argparse
import numpy as np
import random
from pathlib import Path

class Node(object):
    def __init__(self, label, colors, size='', size_val=0, class_id='', affordance='', idx='', raw_label=''):
        self.label = label
        self.color = colors
        self.size = size
        self.size_val = size_val
        self.class_id = class_id
        self.idx = idx
        self.aff = affordance
        self.raw_label = raw_label

class Edge(object):
    def __init__(self, target, anchors, rel):
        self.target = target # node
        self.anchors = anchors # list of nodes
        self.relation = rel

class SceneGraph(object):
    def __init__(self):
        self._graph = defaultdict(set)
        self.nodes = {}

    def add(self, node1, node2, rel):
        # add edge connection to graph
        edge = Edge(node1, node2, rel)
        if node1 in self._graph.keys():
            self._graph[node1].append(edge)
        else:
            self._graph[node1] = [edge]

    def remove(self, node):
        del self._graph[node]
        for id, n in self.nodes.items():
            if n == node:
                del self.nodes[id]

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, dict(self._graph))
    
    def search_node(self, target_node):
        # Note: currently ignores relative size
        target_color = False
        target_size = False
        if target_node.color:
            target_color = True
        nodes = []
        
        for node in self.nodes.values():
            if node.label == target_node.label:
                
                if target_color: # if target color used
                    if target_node.color in node.color:
                        nodes.append(node)
                else: # both objects reference color
                    nodes.append(node)
        
        if len(nodes) > 0:
            return True, nodes
        return False, None
    
    def search(self, rel):
        # searching through all edges would be too costly
        target_node = rel.target
        anchor_node_list = rel.anchors
        anchor_classes = set([n.label for n in anchor_node_list])
        
        target_color = False
        anchor_color = False
        if target_node.color:
            target_color = True
        if any(n.color for n in anchor_node_list):
            anchor_color = True
        
        target_size = False
        anchor_size = False
        if target_node.size:
            target_size = True
        if any(n.size for n in anchor_node_list):
            anchor_size = True
        
        potential_matches = []

        for node in self.nodes.values():
            # matching target class
            if node.label == target_node.label or node.label == target_node.raw_label:
                node_edges = self._graph[node]
                
                for edge in node_edges:
                    # matching relation
                    if edge.relation == rel.relation:
                        current_anchors = set([n.label for n in edge.anchors])
                        
                        # matching anchor class(es)
                        if anchor_classes == current_anchors:              
                            # check for color
                            if target_color and not anchor_color: # if only target color used
                                if target_node.color in node.color:
                                    potential_matches.append([node, edge.anchors])
                            elif not target_color and anchor_color: # if only anchor color used
                                colors = [n.color for n in anchor_node_list]
                                curr_colors = [n.color for n in edge.anchors]
                                color_match = True
                                if len(colors) > 1:
                                    for c in colors:
                                        if c in curr_colors[0] or curr_colors[1]:
                                            color_match = True
                                        else:
                                            color_match = False
                                else:
                                    if colors[0] not in curr_colors[0]:
                                        color_match = False
                                if color_match:
                                    potential_matches.append([node, edge.anchors])
                            elif not target_color and not anchor_color: # colors not used
                                potential_matches.append([node, edge.anchors])
                            else: # both target and anchor reference color
                                target_color_match = False
                                if target_node.color in node.color:
                                    target_color_match = True

                                colors = [n.color for n in anchor_node_list]
                                curr_colors = [n.color for n in edge.anchors]
                                anchor_color_match = True
                                if len(colors) > 1:
                                    for c in colors:
                                        if c in curr_colors[0] or curr_colors[1]:
                                            anchor_color_match = True
                                        else:
                                            anchor_color_match = False
                                else:
                                    if colors[0] not in curr_colors[0]:
                                        anchor_color_match = False
                                
                                if anchor_color_match and target_color_match:
                                    potential_matches.append([node, edge.anchors])

        # if size needed to disambiguate matches
        if len(potential_matches) == 1:
            return True, potential_matches[0][0]
        elif len(potential_matches) > 1:
            # check for relative size
            if target_size and not anchor_size: # only target size referenced
                target_sizes = [item[0].size_val for item in potential_matches]
                if "big" in target_node.size:
                    idx = np.argmax(target_sizes)
                else:
                    idx = np.argmin(target_sizes)
                return True, potential_matches[idx][0]
            elif not target_size and not anchor_size: # this case shouldn't happen due to unique statements
                return True, random.choice([item[0] for item in potential_matches])
            elif not target_size and anchor_size: # only anchor size referenced
                updated_matches = potential_matches
                for i in range(len(anchor_node_list)):
                    anchor_sizes = [item[1][i].size_val for item in updated_matches]
                    if anchor_node_list[i].size:
                        if "big" in anchor_node_list[i].size:
                            idx = np.argwhere(anchor_sizes == np.max(anchor_sizes))
                        else:
                            idx = np.argwhere(anchor_sizes == np.min(anchor_sizes))
                    if len(idx.flatten().tolist()) == 1:
                        return True, potential_matches[idx.flatten()[0]][0]
                return True, updated_matches[idx.flatten()[0]][0] # could also return false if unique match needed
            else: # both sizes used
                updated_matches = potential_matches
                target_sizes = [item[0].size_val for item in potential_matches]
                if "big" in target_node.size:
                    target_idx = np.argwhere(target_sizes == np.max(target_sizes))
                else:
                    target_idx = np.argwhere(target_sizes == np.min(target_sizes))

                if len(target_idx.flatten()) == 1:
                    return True, potential_matches[target_idx.flatten()[0]][0]
                
                updated_matches = [potential_matches[i] for i in target_idx.flatten()]
                for i in range(len(anchor_node_list)):
                    anchor_sizes = [item[1][i].size_val for item in updated_matches]
                    if anchor_node_list[i].size:
                        if "big" in anchor_node_list[i].size:
                            idx = np.argwhere(anchor_sizes == np.max(anchor_sizes))
                        else:
                            idx = np.argwhere(anchor_sizes == np.min(anchor_sizes))
                    if len(idx.flatten().tolist()) == 1:
                        return True, potential_matches[idx.flatten()[0]][0]
                    updated_matches = potential_matches[idx.flatten()]
                
                return True, updated_matches[idx.flatten()[0]][0] # could also return false if unique match needed
        else:
            return False, None