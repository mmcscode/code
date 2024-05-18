import gzip
import codecs
import os
import json
from subprocess import Popen, PIPE
from tqdm import tqdm
import networkx as nx
from collections import defaultdict
from spacy.tokens import Doc
import multiprocess as mp
import pickle
import re
import spacy
import multiprocessing
from pathlib import Path
import pickle
NLP = spacy.load('en_core_web_sm')
# 构件图数据
def build_desc_graph(desc, file=None):
    try:
        if str(desc).endswith('.'):
            desc = desc[0: len(desc)-1]
        desc = ' '.join(desc.split())
        doc = NLP(desc)
        g_features = []
        dep_tree = defaultdict(list)
        boundary_nodes = []
        for sent in doc.sents:
            boundary_nodes.append(sent[-1].i)
            for each in sent:
                g_features.append(each.text)
                if each.i != each.head.i:
                    dep_tree[each.head.i].append({'node': each.i, 'edge': each.dep_})

        for i in range(len(boundary_nodes) - 1):
            dep_tree[boundary_nodes[i]].append({'node': boundary_nodes[i] + 1, 'edge': 'neigh'})
            dep_tree[boundary_nodes[i] + 1].append({'node': boundary_nodes[i], 'edge': 'neigh'})
        edges = []
        for key, values in dep_tree.items():
            for value in values:
                edges.append((value['edge'], key, value['node']))
        if edges:
            des_graph = {'backbone_sequence': g_features, 'edges': edges}
        else:
            des_graph = {}
    except:
        des_graph = {}
    return des_graph


def normalize_des_graph(des_graph):
    new_tokens = []
    new_edges = []
    nodes_to_subtokenize = {}
    mapping = {}
    count = 0
    for index, token in enumerate(des_graph['backbone_sequence']):
        subtokens = subtokenizer(token)
        if len(subtokens) > 1:
            for subtoken in subtokens:
                if index not in nodes_to_subtokenize.keys():
                    nodes_to_subtokenize[index] = [count]
                    mapping[index] = count
                else:
                    nodes_to_subtokenize[index].append(count)
                new_tokens.append(subtoken)
                count += 1
        else:
            mapping[index] = count
            new_tokens.extend(subtokens)
            count += 1
    for edge in des_graph['edges']:
        new_edges.append((edge[0].upper(), mapping[edge[1]], mapping[edge[2]]))
    for key in nodes_to_subtokenize.keys():
        for index in range(1, len(nodes_to_subtokenize[key])):
            new_edges.append(('SUB_TOKEN', nodes_to_subtokenize[key][index - 1], nodes_to_subtokenize[key][index]))
    for index in range(len(new_tokens) - 1):
        new_edges.append(('NEXT_TOKEN', index, index + 1))
    return {'backbone_sequence': new_tokens, 'edges': new_edges}

def subtokenizer(identifier):
    if identifier == 'MONKEYS_AT':
        return [identifier]
    splitter_regex = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    identifiers = re.split('[._\-]', identifier)
    subtoken_list = []
    for identifier in identifiers:
        matches = splitter_regex.finditer(identifier)
        for subtoken in [m.group(0) for m in matches]:
            subtoken_list.append(subtoken)
    return subtoken_list