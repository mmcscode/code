from ast import parse

import os
import gzip
import json
from tqdm import tqdm
import re
import pickle
import multiprocess as mp
from spacy.tokens import Doc
import spacy
from pathlib import Path
from collections import defaultdict
import codecs
import sys
sys.path.append(".")


NLP = spacy.load('en_core_web_sm')



home_path = str(Path.home())

sys.path.append(home_path + '/autodl-tmp/project/mmcs_nn/parsers/sourcecode/python/')

from .ast_graph_generator import AstGraphGenerator
# 数据集存放位置
RAW_FILE = os.path.join(home_path, 'autodl-tmp/data/python/final/jsonl')
PYTHON_BASE = os.path.join(home_path,
                           'autodl-tmp/data/python_dedupe_definitions_v2.pkl')
BASE_GRAPH_FILE = os.path.join(home_path, 'autodl-tmp/data/python_base_gz')

NLP = spacy.load('en')

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(' ')
        # 分词
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

NLP.tokenizer = WhitespaceTokenizer(NLP.vocab)



def post_process():
    all_samples = load_data(RAW_FILE)
    process_list = []
    for key in all_samples.keys():
        process = mp.Process(target=single_post_process,
                             args=(key, all_samples[key]))
        process_list.append(process)
        process.start()
    for process in process_list:
        process.join()


def single_post_process(key, chunk):
    file_graph_name = os.path.join(os.path.join(RAW_FILE, key), 'graph_' + key + '_gnn.jsonl.gz')
    count = 0
    
    with gzip.GzipFile(file_graph_name, 'wb') as gnn_file:
        for raw_sample in tqdm(chunk):
            try:
                graph = build_python_graph(raw_sample['code'])
                if graph and len(graph['node_labels']) <= 200:
                        raw_sample['code_graph'] = normalize_graph(graph)

                else:
                    continue
                if raw_sample['docstring_tokens']:
                    doc_summary = ' '.join(re.sub(r'[^A-Za-z ]+', ' ',
                                                  ' '.join(raw_sample['docstring_tokens'])).split())
                    if doc_summary:
                        doc_graph = build_desc_graph(doc_summary)
                        raw_sample['doc_graph'] = normalize_des_graph(doc_graph)
                else:
                    raw_sample['doc_graph'] = {}
                if raw_sample['code_graph'] and raw_sample['doc_graph']:
                    save_sample_to_jsonl_gz(raw_sample, gnn_file)
                    count += 1


                del raw_sample
            except:
                continue
        print('there are %d samples have code graph and doc graph in %s' % (count, key))

        gnn_file.close()


def build_python_graph(code):
    try:
        visitor = AstGraphGenerator()
        visitor.visit(parse(code))
        edge_list = [(t, origin, destination)
                     for (origin, destination), edges
                     in visitor.graph.items() for t in edges]
        graph_node_labels = [label.strip() for (_, label) in sorted(visitor.node_label.items())]
        graph = {"edges": edge_list, "backbone_sequence": visitor.terminal_path, "node_labels": graph_node_labels}
    except:
        # print('sample %s build code graph failed' % index)
        graph = {}
    return graph

# 创建子边
def subtokenizer(identifier):
    splitter_regex = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    identifiers = re.split('[._\-]', identifier)
    subtoken_list = []

    for identifier in identifiers:
        matches = splitter_regex.finditer(identifier)
        for subtoken in [m.group(0) for m in matches]:
            subtoken_list.append(subtoken)

    return subtoken_list


def normalize_graph(graph):
    node_labels = graph['node_labels']
    backbone_sequence = graph['backbone_sequence']
    formatted_node_labels = []
    for index, node_label in enumerate(node_labels):
        formatted_node_labels.append({'id': index, 'contents': node_label, 'type': node_label})
    index = 0
    mapping = {}
    nodes_to_subtokenize = {}
    token_sequential_list = []
    seq_token_in_node = []
    method_nodes = []
    method_edges = []
    for i, sorted_node in enumerate(formatted_node_labels):
        if i in backbone_sequence:
            subtokens = subtokenizer(sorted_node['contents'])
            if len(subtokens) > 1:
                for subtoken in subtokens:
                    dummy_node = sorted_node.copy()
                    dummy_node['contents'] = subtoken
                    dummy_node['id_sorted'] = index
                    dummy_node['subtoken'] = True
                    dummy_node['ori_token'] = sorted_node['contents']
                    token_sequential_list.append(subtoken)
                    seq_token_in_node.append(index)
                    method_nodes.append(dummy_node)
                    if sorted_node['id'] not in mapping.keys():
                        mapping[sorted_node['id']] = index
                    if sorted_node['id'] not in nodes_to_subtokenize.keys():
                        nodes_to_subtokenize[sorted_node['id']] = [index]
                    else:
                        nodes_to_subtokenize[sorted_node['id']].append(index)
                    index += 1
            else:
                sorted_node['id_sorted'] = index
                sorted_node['subtoken'] = False
                method_nodes.append(sorted_node)
                mapping[sorted_node['id']] = index
                seq_token_in_node.append(index)
                token_sequential_list.append(sorted_node['contents'])
                index += 1
        else:
            sorted_node['id_sorted'] = index
            method_nodes.append(sorted_node)
            mapping[sorted_node['id']] = index
            index += 1
    edge_label_dict = {'child': 'AST_CHILD', 'NextToken': 'NEXT_TOKEN', 'computed_from': 'COMPUTED_FROM',
                       'last_use': 'LAST_USE', 'last_write': 'LAST_WRITE'}
    for edge in graph['edges']:
        if edge[0] in edge_label_dict.keys():
            method_edges.append({'type': edge_label_dict[edge[0]], 'sourceId': mapping[edge[1]], 'destinationId': mapping[edge[2]]})
    for key in nodes_to_subtokenize.keys():
        for index in range(1, len(nodes_to_subtokenize[key])):
            edge = {}
            edge['type'] = 'SUB_TOKEN'
            edge['sourceId'] = nodes_to_subtokenize[key][index-1]
            edge['destinationId'] = nodes_to_subtokenize[key][index]
            method_edges.append(edge)
    if not len(seq_token_in_node) == len(token_sequential_list):
        code_graph = {}
    else:
        code_graph = {'nodes': method_nodes, 'edges': method_edges, 'seq_token_in_node': seq_token_in_node,
                      'tokens': token_sequential_list}
    return code_graph


def load_data(file_folder):
    partition_folders = os.listdir(file_folder)
    all_samples = {}
    for folder in partition_folders:
        samples = []
        files = os.listdir(os.path.join(file_folder, folder))
        for file in files:
            if file.startswith('python') and file.endswith('.gz'):
                with gzip.open(os.path.join(file_folder, folder, file)) as f:
                    for line in f:
                        samples.append(json.loads(line.decode('utf-8')))
        all_samples[folder] = samples
    return all_samples


def build_python_base_graphs():
    definitions = pickle.load(open(PYTHON_BASE, 'rb'))
    chunk_samples = list(chunks(definitions, 100000))
    print('there are %d chunks' % len(chunk_samples))
    process_list = []
    for index in range(len(chunk_samples)):
        process = mp.Process(target=single_post_base_process,
                             args=(chunk_samples[index], index))
        process_list.append(process)
        process.start()
    for process in process_list:
        process.join()


def single_post_base_process(raw_samples, index):
    count = 0
    if not os.path.exists(BASE_GRAPH_FILE):
        os.makedirs(BASE_GRAPH_FILE)
    with gzip.GzipFile(os.path.join(BASE_GRAPH_FILE, 'base_graph_gnn_' + str(index) + '.jsonl.gz'), 'wb') as gnn_file:
        for raw_sample in tqdm(raw_samples):
            function = raw_sample['function']
            graph = build_python_graph(function)
            if graph and len(graph['node_labels']) <= 200:
                if raw_sample['code_tokens']:
                    code_summary = ' '.join(re.sub(r'[^A-Za-z0-9 ]+', ' ',
                                                   ' '.join(raw_sample['code_tokens'])).split())
                    new_tokens = []

                    for index, token in enumerate(code_summary):
                        subtokens = subtokenizer(token)
                        if len(subtokens) > 1:
                            for subtoken in subtokens:
                                new_tokens.append(subtoken)

                        else:

                            new_tokens.append(subtokens)

                    new_tokens = ' '.join(re.sub(r'[^A-Za-z0-9 ]+', ' ',
                                                 ' '.join(new_tokens)).split())
                raw_sample['code_graph'] = normalize_graph(graph, new_tokens)

                if raw_sample['code_graph']:
                    save_sample_to_jsonl_gz(raw_sample, gnn_file)
                    count += 1
                del raw_sample
            else:
                continue
        print('there are %d samples in chunk %d' % (count, index))
        gnn_file.close()


def check_data(files):
    for file in files:
        samples = []
        with gzip.open(file) as f:
            for index, line in enumerate(f):
                print(index)
                raw_sample = json.loads(line.decode('utf-8'))
                if raw_sample['docstring_tokens']:
                    doc_summary = ' '.join(re.sub(r'[^A-Za-z0-9 ]+', ' ',
                                                  ' '.join(raw_sample['docstring_tokens'])).split())
                    if doc_summary:
                        doc_graph = build_desc_graph(doc_summary)
                        if doc_graph:
                            raw_sample['doc_graph'] = normalize_des_graph(doc_graph)
                            samples.append(raw_sample)
                        else:
                            continue
                else:
                    continue
        save_to_jsonl_gz(samples, file)


def save_to_jsonl_gz(functions, file_name):
    with gzip.GzipFile(file_name, 'wb') as out_file:
        writer = codecs.getwriter('utf-8')
        for entry in functions:
            writer(out_file).write(json.dumps(entry))
            writer(out_file).write('\n')

def normalize_graph(graph, func_nodes):
    method_nodes = []
    method_edges = []
    sorted_nodes = sorted(graph['node'], key=lambda i: int(i['id']))
    mapping = {}
    index = 0
    nodes_to_subtokenize = {}
    token_sequential_list = []
    seq_token_in_node = []
    try:
        for sorted_node in sorted_nodes:
            if sorted_node['id'] in func_nodes:
                if 'TOKEN' in sorted_node['type']:
                    subtokens = subtokenizer(sorted_node['contents'])
                    if len(subtokens) > 1:
                        for subtoken in subtokens:
                            dummy_node = sorted_node.copy()
                            dummy_node['contents'] = subtoken
                            dummy_node['id_sorted'] = index
                            dummy_node['subtoken'] = True
                            dummy_node['ori_token'] = sorted_node['contents']
                            token_sequential_list.append(subtoken)
                            seq_token_in_node.append(index)
                            method_nodes.append(dummy_node)
                            if sorted_node['id'] not in mapping.keys():
                                mapping[sorted_node['id']] = index
                            if sorted_node['id'] not in nodes_to_subtokenize.keys():
                                nodes_to_subtokenize[sorted_node['id']] = [index]
                            else:
                                nodes_to_subtokenize[sorted_node['id']].append(index)
                            index += 1
                    else:
                        sorted_node['id_sorted'] = index
                        sorted_node['subtoken'] = False
                        method_nodes.append(sorted_node)
                        mapping[sorted_node['id']] = index
                        seq_token_in_node.append(index)
                        token_sequential_list.append(sorted_node['contents'])
                        index += 1
                else:
                    sorted_node['id_sorted'] = index
                    method_nodes.append(sorted_node)
                    mapping[sorted_node['id']] = index
                    index += 1
        for edge in graph['edge']:
            if edge['sourceId'] in func_nodes and edge['destinationId'] in func_nodes and edge['type'] in \
                    ['AST_CHILD', 'NEXT_TOKEN', 'COMPUTED_FROM', 'LAST_USE', 'LAST_WRITE']:
                edge['sourceId'] = mapping[edge['sourceId']]
                edge['destinationId'] = mapping[edge['destinationId']]
                method_edges.append(edge)
        for key in nodes_to_subtokenize.keys():
            for index in range(1, len(nodes_to_subtokenize[key])):
                edge = {}
                edge['sourceId'] = nodes_to_subtokenize[key][index - 1]
                edge['destinationId'] = nodes_to_subtokenize[key][index]
                edge['type'] = 'SUB_TOKEN'
                method_edges.append(edge)
        if not len(seq_token_in_node) == len(token_sequential_list):
            code_graph = {}
        else:
            code_graph = {'nodes': method_nodes, 'edges': method_edges, 'seq_token_in_node': seq_token_in_node,
                          'tokens': token_sequential_list}
    except:
        code_graph = {}
    return code_graph

def build_desc_graph(desc, file=None):
    try:
        if str(desc).endswith('.'):
            desc = desc[0: len(desc) - 1]
        desc = ' '.join(desc.split())
        doc = NLP(desc)
        g_features = []
        dep_tree = defaultdict(list)
        boundary_nodes = []
        for sent in doc.sents:
            boundary_nodes.append(sent[-1].i)
            for each in sent:
                g_features.append(each.text)
                if each.i != each.head.i:  # Not a root
                    dep_tree[each.head.i].append({'node': each.i, 'edge': each.dep_})
        # 在抽象语法树图上构建邻边
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


def save_sample_to_jsonl_gz(function, out_file):
    writer = codecs.getwriter('utf-8')
    writer(out_file).write(json.dumps(function))
    writer(out_file).write('\n')

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

if __name__ == "__main__":
    post_process()
    # build_python_base_graphs()
