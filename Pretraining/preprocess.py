import json
import torch
from transformers import *
import argparse
import numpy as np
from tqdm import tqdm
# from fuzzywuzzy import fuzz
import os
import pickle
from Pretraining.utils import *

tokenizer = BertTokenizer.from_pretrained('/data/csl/resources/Bert/bert-base-cased', do_lower_case = False)



            

def get_vocab(args, vocab):
    kb = json.load(open(os.path.join(args.input_dir, 'kb.json')))
    entities = kb['entities']
    for eid in entities:
        relations = entities[eid]['relations']
        for relation in relations:
            r = relation['predicate']
            if relation['direction'] == 'backward':
                r = '[inverse] ' + r
            if not r in vocab['relation2id']:
                vocab['relation2id'][r] = len(vocab['relation2id'])
    vocab['id2relation'] = [relation for relation, id in vocab['relation2id'].items()]

    concepts = kb['concepts']
    for cid in concepts:
        concept = concepts[cid]['name']
        if not concept in vocab['concept2id']:
            vocab['concept2id'][concept] = len(vocab['concept2id'])
    vocab['id2concept'] = [concept for concept, id in vocab['concept2id'].items()]

    train = [json.loads(line.strip()) for line in open(os.path.join(args.input_dir, 'train.json'))]
    for item in train:
        program = item['program']
        for f in program:
            function = f['function']
            if not function in vocab['function2id']:
                vocab['function2id'][function] = len(vocab['function2id'])
    vocab['id2function'] = [function for function, id in vocab['function2id'].items()]

    
def get_relation_dataset(args, vocab):
    # train = json.load(open(os.path.join(args.input_dir, 'train.json')))
    # dev = json.load(open(os.path.join(args.input_dir, 'val.json')))
    train = [json.loads(line.strip()) for line in open(os.path.join(args.input_dir, 'train.json'))]
    dev = [json.loads(line.strip()) for line in open(os.path.join(args.input_dir, 'val.json'))]

    for name, raw_data in zip(['train', 'dev'], [train, dev]):
        dataset = []
        for item in tqdm(raw_data):
            text = item['question']
            program = item['program']
            data = []
            relations = []
            for idx, f in enumerate(program):
                function = f['function']
                if function == 'Relate':
                    inputs = f['inputs']
                    r = inputs[0]
                    if inputs[1] == 'backward':
                        r = '[inverse] ' + r
                    if not r in vocab['relation2id']:
                        continue
                    r = vocab['relation2id'][r]
                    relations.append([idx + 1, r])
                function_id = vocab['function2id'][function]
                data.append({'function': function_id})
            if len(relations) == 0:
                relations.append([0, vocab['relation2id']['<PAD>']])
            dataset.append({'question': text, 'program': data, 'relations': relations})
        # verbose = True
        # if verbose:
        #     for idx in range(100):
        #         print('*'*10)
        #         text = dataset[idx]['question']
        #         print(text)
        #         text = tokenizer.tokenize(text)
        #         for f in dataset[idx]['program']:
        #             function_id = f['function']
        #             print(vocab['id2function'][function_id])
        #         for pos, r in dataset[idx]['relations']:
        #             print(pos, vocab['id2relation'][r])


        with open(os.path.join(args.output_dir, 'relation', '%s.json'%(name)), 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')

def get_concept_dataset(args, vocab):
    train = [json.loads(line.strip()) for line in open(os.path.join(args.input_dir, 'train.json'))]
    dev = [json.loads(line.strip()) for line in open(os.path.join(args.input_dir, 'val.json'))]
    for name, raw_data in zip(['train', 'dev'], [train, dev]):
        dataset = []
        for item in tqdm(raw_data):
            text = item['question']
            program = item['program']
            data = []
            concepts = []
            for idx, f in enumerate(program):
                function = f['function']
                if function == 'FilterConcept':
                    inputs = f['inputs']
                    c = inputs[0]
                    if not c in vocab['concept2id']:
                        continue
                    c = vocab['concept2id'][c]
                    concepts.append([idx + 1, c])
                function_id = vocab['function2id'][function]
                data.append({'function': function_id})
            if len(concepts) == 0:
                concepts.append([0, vocab['concept2id']['<PAD>']])
            dataset.append({'question': text, 'program': data, 'concepts': concepts})
        # verbose = True
        # if verbose:
        #     for idx in range(100):
        #         print('*'*10)
        #         text = dataset[idx]['question']
        #         print(text)
        #         text = tokenizer.tokenize(text)
        #         for f in dataset[idx]['program']:
        #             function_id = f['function']
        #             print(vocab['id2function'][function_id])
        #         for pos, r in dataset[idx]['concepts']:
        #             print(pos, vocab['id2concept'][r])


        with open(os.path.join(args.output_dir, 'concept', '%s.json'%(name)), 'w') as f:
            for item in dataset:
                f.write(json.dumps(item) + '\n')

def encode_relation(args, vocab):
    encoded_inputs = tokenizer(vocab['id2relation'], padding = True)
    print(encoded_inputs.keys())
    print(len(encoded_inputs['input_ids'][0]))
    print(len(encoded_inputs['token_type_ids'][0]))
    print(len(encoded_inputs['attention_mask'][0]))
    print(tokenizer.decode(encoded_inputs['input_ids'][0]))
    max_seq_length = len(encoded_inputs['input_ids'][0])
    input_ids_list = encoded_inputs['input_ids']
    token_type_ids_list = encoded_inputs['token_type_ids']
    attention_mask_list = encoded_inputs['attention_mask']
    input_ids_list = np.array(input_ids_list, dtype=np.int32)
    token_type_ids_list = np.array(token_type_ids_list, dtype=np.int32)
    attention_mask_list = np.array(attention_mask_list, dtype=np.int32)
    return input_ids_list, token_type_ids_list, attention_mask_list

def encode_concept(args, vocab):
    encoded_inputs = tokenizer(vocab['id2concept'], padding = True)
    print(encoded_inputs.keys())
    print(len(encoded_inputs['input_ids'][0]))
    print(len(encoded_inputs['token_type_ids'][0]))
    print(len(encoded_inputs['attention_mask'][0]))
    print(tokenizer.decode(encoded_inputs['input_ids'][0]))
    max_seq_length = len(encoded_inputs['input_ids'][0])
    input_ids_list = encoded_inputs['input_ids']
    token_type_ids_list = encoded_inputs['token_type_ids']
    attention_mask_list = encoded_inputs['attention_mask']
    input_ids_list = np.array(input_ids_list, dtype=np.int32)
    token_type_ids_list = np.array(token_type_ids_list, dtype=np.int32)
    attention_mask_list = np.array(attention_mask_list, dtype=np.int32)
    return input_ids_list, token_type_ids_list, attention_mask_list

def encode_relation_dataset(args, vocab, dataset):
    def get_function_ids(program):
        function_ids = [f['function'] for f in program]
        return function_ids

    tmp = []
    for item in dataset:
        question = item['question']
        program = item['program']
        relations = item['relations']
        for relation in relations:
            tmp.append({'question': question, 'program': program, 'relation': relation})
    print('dataset size: {}'.format(len(dataset)))
    dataset = tmp
    print('new dataset size: {}'.format(len(dataset)))
    questions = []
    for item in dataset:
        question = item['question']
        questions.append(question)
    encoded_inputs = tokenizer(questions, padding = True)
    # print(encoded_inputs.keys())
    # print(len(encoded_inputs['input_ids'][0]))
    # print(len(encoded_inputs['token_type_ids'][0]))
    # print(len(encoded_inputs['attention_mask'][0]))
    # print(tokenizer.decode(encoded_inputs['input_ids'][0]))
    max_seq_length = len(encoded_inputs['input_ids'][0])
    function_ids_list = []
    for item in tqdm(dataset):
        program = item['program']
        program = [{'function': vocab['function2id']['<START>']}] + program + [{'function': vocab['function2id']['<END>']}]
        function_ids = get_function_ids(program)
        function_ids_list.append(function_ids)
    max_func_len = max([len(function_ids) for function_ids in function_ids_list])
    print('max_func_len: {}'.format(max_func_len))
    for function_ids in function_ids_list:
        while len(function_ids) < max_func_len:
            function_ids.append(vocab['function2id']['<PAD>'])
        assert len(function_ids) == max_func_len
    relation_pos_list = []
    relation_id_list = []
    for item in dataset:
        relation = item['relation']
        relation_pos_list.append([relation[0]])
        relation_id_list.append([relation[1]])

    input_ids_list = encoded_inputs['input_ids']
    token_type_ids_list = encoded_inputs['token_type_ids']
    attention_mask_list = encoded_inputs['attention_mask']
    # verbose = False
    # if verbose:
    #     for idx in range(10):
    #         question = tokenizer.decode(input_ids_list[idx])
    #         functions = [vocab['id2function'][id] for id in function_ids_list[idx]]
    #         relation_pos = relation_pos_list[idx][0]
    #         relation_id = vocab['id2relation'][relation_id_list[idx][0]]
    #         print(question, functions, relation_pos, relation_id)

    input_ids_list = np.array(input_ids_list, dtype=np.int32)
    token_type_ids_list = np.array(token_type_ids_list, dtype=np.int32)
    attention_mask_list = np.array(attention_mask_list, dtype=np.int32)
    function_ids_list = np.array(function_ids_list, dtype=np.int32)
    relation_pos_list = np.array(relation_pos_list, dtype=np.int32)
    relation_id_list = np.array(relation_id_list, dtype=np.int32)
    
    return input_ids_list, token_type_ids_list, attention_mask_list, function_ids_list, relation_pos_list, relation_id_list


def encode_concept_dataset(args, vocab, dataset):
    def get_function_ids(program):
        function_ids = [f['function'] for f in program]
        return function_ids

    tmp = []
    for item in dataset:
        question = item['question']
        program = item['program']
        concepts = item['concepts']
        for concept in concepts:
            tmp.append({'question': question, 'program': program, 'concept': concept})
    print('dataset size: {}'.format(len(dataset)))
    dataset = tmp
    print('new dataset size: {}'.format(len(dataset)))
    questions = []
    for item in dataset:
        question = item['question']
        questions.append(question)
    encoded_inputs = tokenizer(questions, padding = True)
    # print(encoded_inputs.keys())
    # print(len(encoded_inputs['input_ids'][0]))
    # print(len(encoded_inputs['token_type_ids'][0]))
    # print(len(encoded_inputs['attention_mask'][0]))
    # print(tokenizer.decode(encoded_inputs['input_ids'][0]))
    max_seq_length = len(encoded_inputs['input_ids'][0])
    function_ids_list = []
    for item in tqdm(dataset):
        program = item['program']
        program = [{'function': vocab['function2id']['<START>']}] + program + [{'function': vocab['function2id']['<END>']}]
        function_ids = get_function_ids(program)
        function_ids_list.append(function_ids)
    max_func_len = max([len(function_ids) for function_ids in function_ids_list])
    print('max_func_len: {}'.format(max_func_len))
    for function_ids in function_ids_list:
        while len(function_ids) < max_func_len:
            function_ids.append(vocab['function2id']['<PAD>'])
        assert len(function_ids) == max_func_len
    relation_pos_list = []
    relation_id_list = []
    for item in dataset:
        relation = item['concept']
        relation_pos_list.append([relation[0]])
        relation_id_list.append([relation[1]])

    input_ids_list = encoded_inputs['input_ids']
    token_type_ids_list = encoded_inputs['token_type_ids']
    attention_mask_list = encoded_inputs['attention_mask']
    verbose = False
    if verbose:
        for idx in range(10):
            question = tokenizer.decode(input_ids_list[idx])
            functions = [vocab['id2function'][id] for id in function_ids_list[idx]]
            relation_pos = relation_pos_list[idx][0]
            relation_id = vocab['id2concept'][relation_id_list[idx][0]]
            print(question, functions, relation_pos, relation_id)

    input_ids_list = np.array(input_ids_list, dtype=np.int32)
    token_type_ids_list = np.array(token_type_ids_list, dtype=np.int32)
    attention_mask_list = np.array(attention_mask_list, dtype=np.int32)
    function_ids_list = np.array(function_ids_list, dtype=np.int32)
    relation_pos_list = np.array(relation_pos_list, dtype=np.int32)
    relation_id_list = np.array(relation_id_list, dtype=np.int32)
    
    return input_ids_list, token_type_ids_list, attention_mask_list, function_ids_list, relation_pos_list, relation_id_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required = True, type = str)
    parser.add_argument('--output_dir', required = True, type = str)
    args = parser.parse_args()
    print(args)
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    if not os.path.isdir(os.path.join(args.output_dir, 'relation')):
        os.makedirs(os.path.join(args.output_dir, 'relation'))
    if not os.path.isdir(os.path.join(args.output_dir, 'concept')):
        os.makedirs(os.path.join(args.output_dir, 'concept'))
    vocab = {
        'relation2id': {
            '<PAD>': 0
        },
        'concept2id': {
            '<PAD>': 0
        },
        'function2id':{
            '<PAD>': 0,
            '<START>': 1,
            '<END>':2
        }
    }
    get_vocab(args, vocab)

    for k in vocab:
        print('{}:{}'.format(k, len(vocab[k])))
    fn = os.path.join(args.output_dir, 'vocab.json')
    print('Dump vocab to {}'.format(fn))
    with open(fn, 'w') as f:
        json.dump(vocab, f, indent=2)

    outputs = encode_relation(args, vocab)
    with open(os.path.join(args.output_dir, 'relation', 'relation.pt'), 'wb') as f:
        for o in outputs:
            print(o.shape)
            pickle.dump(o, f)

    outputs = encode_concept(args, vocab)
    with open(os.path.join(args.output_dir, 'concept', 'concept.pt'), 'wb') as f:
        for o in outputs:
            print(o.shape)
            pickle.dump(o, f)
            
    get_relation_dataset(args, vocab)
    get_concept_dataset(args, vocab)
    # vocab = json.load(open(os.path.join(args.output_dir, 'vocab.json')))
    for name in ['train', 'dev']:
        dataset = []
        with open(os.path.join(args.output_dir, 'relation', '%s.json'%(name))) as f:
            for line in f:
                dataset.append(json.loads(line.strip()))
        outputs = encode_relation_dataset(args, vocab, dataset)
        assert len(outputs) == 6
        print('shape of input_ids, token_type_ids, attention_mask, function_ids， relation_pos, relation_id:')
        with open(os.path.join(args.output_dir, 'relation', '{}.pt'.format(name)), 'wb') as f:
            for o in outputs:
                print(o.shape)
                pickle.dump(o, f)
    
    for name in ['train', 'dev']:
        dataset = []
        with open(os.path.join(args.output_dir, 'concept', '%s.json'%(name))) as f:
            for line in f:
                dataset.append(json.loads(line.strip()))
        outputs = encode_concept_dataset(args, vocab, dataset)
        assert len(outputs) == 6
        print('shape of input_ids, token_type_ids, attention_mask, function_ids， relation_pos, relation_id:')
        with open(os.path.join(args.output_dir, 'concept', '{}.pt'.format(name)), 'wb') as f:
            for o in outputs:
                print(o.shape)
                pickle.dump(o, f)



if __name__ == "__main__":
    main()
