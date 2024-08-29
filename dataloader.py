# -*- coding: utf-8 -*-

import os
import json
import random
from collections import OrderedDict, Collection
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from utils import get_aug_instance

class BaseDataset(Dataset):
    def __init__(self, 
                 dataset_path,
                 max_length, 
                 tokenize,
                 N, K, Q, O, use_BIO=True):
        self.raw_data = json.load(open(dataset_path, "r", encoding='utf-8'))
        self.classes = self.raw_data.keys()
        
        self.max_length = max_length - 2
        self.tokenize = tokenize
        
        self.N = N
        self.K = K
        self.Q = Q
        self.O = O
        self.use_BIO = use_BIO
        
    def __len__(self):
        return 99999999
    
    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)

        label2id, id2label = self.build_dict(target_classes)

        support_set = {'tokens': [], 'entity_label': [], 'B-mask': [], 'I-mask': [], 'att-mask': [], 'text-mask': []}
        query_set = {'tokens': [], 'entity_label': [], 'B-mask': [], 'I-mask': [], 'att-mask': [], 'text-mask': []}
        
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.raw_data[class_name]))), 
                    self.K + self.Q, False)

            count = 0
            for j in indices:
                if count < self.K:
                    instance = self.preprocess(self.raw_data[class_name][j], [class_name])
                    token_ids, label_ids, B_mask, I_mask, att_mask, text_mask = self.tokenize(instance, label2id)
                    
                    support_set['tokens'].append(token_ids)
                    support_set['entity_label'].append(label_ids)
                    support_set['B-mask'].append(B_mask)
                    support_set['I-mask'].append(I_mask)
                    support_set['att-mask'].append(att_mask)
                    support_set['text-mask'].append(text_mask)
                else:
                    instance = self.preprocess(self.raw_data[class_name][j], target_classes)
                    token_ids, label_ids, B_mask, I_mask, att_mask, text_mask = self.tokenize(instance, label2id)
                    
                    query_set['tokens'].append(token_ids)
                    query_set['entity_label'].append(label_ids)
                    query_set['B-mask'].append(B_mask)
                    query_set['I-mask'].append(I_mask)
                    query_set['att-mask'].append(att_mask)
                    query_set['text-mask'].append(text_mask)
                count += 1
        
        for k, v in support_set.items():
            support_set[k] = torch.stack(v)
        
        for k, v in query_set.items():
            query_set[k] = torch.stack(v)
        
        return support_set, query_set, id2label
    
    def preprocess(self, instance, event_type_list):
        raise NotImplementedError
    
    def build_dict(self, rel_type_list):
        label2id = OrderedDict()
        id2label = OrderedDict()

        label2id['O'] = 0
        id2label[0] = 'O'
        label2id['PAD'] = -100
        id2label[-100] = 'PAD'
        
        for i, rel_type in enumerate(rel_type_list):
            label2id['I-' + rel_type + ':HEAD'] = 2 * i + 1
            label2id['I-' + rel_type + ':TAIL'] = 2 * i + 2
            id2label[2 * i + 1] = 'I-' + rel_type + ':HEAD'
            id2label[2 * i + 2] = 'I-' + rel_type + ':TAIL'
        
        return label2id, id2label


class FewEventDataset(BaseDataset):

    def __getitem__(self, index):
        target_classes = random.sample(self.classes, self.N)
        label2id, id2label = self.build_dict(target_classes)

        # print('target_classes', target_classes)

        support_set = {'tokens': [], 'entity_label': [], 'B-mask': [], 'I-mask': [], "att-mask": [], 'text-mask': []}
        query_set = {'tokens': [], 'entity_label': [], 'B-mask': [], 'I-mask': [], "att-mask": [], 'text-mask': []}

        # N--Way
        for i, class_name in enumerate(target_classes):
            # (K + Q)--Shot
            indices = np.random.choice(
                    list(range(len(self.raw_data[class_name]))), 
                    self.K + self.Q, False)
            count = 0
            # 查询集中样本的实体
            query_h_entitys, query_t_entitys = '', ''
            for j in indices:
                if count == 0:
                    # print('query_set_class_name', class_name)
                    # print('query_set_self.raw_data[class_name][j]', self.raw_data[class_name][j])

                    instance = self.preprocess(self.raw_data[class_name][j], class_name, target_classes)
                    token_ids, label_ids, B_mask, I_mask, att_mask, text_mask = self.tokenize(instance, label2id)

                    query_set['tokens'].append(token_ids)
                    query_set['entity_label'].append(label_ids)
                    query_set['B-mask'].append(B_mask)
                    query_set['I-mask'].append(I_mask)
                    query_set['att-mask'].append(att_mask)
                    query_set['text-mask'].append(text_mask)

                    query_h_entitys = self.raw_data[class_name][j]['h'][0]
                    query_t_entitys = self.raw_data[class_name][j]['t'][0]
                else:
                    # print('class_name', class_name)
                    # print('self.raw_data[class_name][j]', self.raw_data[class_name][j])

                    instance = self.preprocess(self.raw_data[class_name][j], class_name, [class_name])
                    token_ids, label_ids, B_mask, I_mask, att_mask, text_mask = self.tokenize(instance, label2id)
                    support_set['tokens'].append(token_ids)
                    support_set['entity_label'].append(label_ids)
                    support_set['B-mask'].append(B_mask)
                    support_set['I-mask'].append(I_mask)
                    support_set['att-mask'].append(att_mask)
                    support_set['text-mask'].append(text_mask)
                    # 支撑集进行实体数据增强-增强的实体不能出现在查询集中
                    aug_num = 4
                    for i in range(aug_num):
                        aug_instance = get_aug_instance(class_name, self.raw_data[class_name][j], query_h_entitys, query_t_entitys)
                        # print('support_aug_instance', aug_instance)
                        aug_instance = self.preprocess(aug_instance, class_name, [class_name])
                        token_ids, label_ids, B_mask, I_mask, att_mask, text_mask = self.tokenize(aug_instance, label2id)
                        support_set['tokens'].append(token_ids)
                        support_set['entity_label'].append(label_ids)
                        support_set['B-mask'].append(B_mask)
                        support_set['I-mask'].append(I_mask)
                        support_set['att-mask'].append(att_mask)
                        support_set['text-mask'].append(text_mask)
                count += 1

                # if count < self.K:
                #     # print('class_name', class_name)
                #     # print('self.raw_data[class_name][j]', self.raw_data[class_name][j])
                #
                #     instance = self.preprocess(self.raw_data[class_name][j], class_name, [class_name])
                #     token_ids, label_ids, B_mask, I_mask, att_mask, text_mask = self.tokenize(instance, label2id)
                #     support_set['tokens'].append(token_ids)
                #     support_set['entity_label'].append(label_ids)
                #     support_set['B-mask'].append(B_mask)
                #     support_set['I-mask'].append(I_mask)
                #     support_set['att-mask'].append(att_mask)
                #     support_set['text-mask'].append(text_mask)
                #
                #     aug_num = 4
                #     for i in range(aug_num):
                #         aug_instance = get_aug_instance(class_name, self.raw_data[class_name][j])
                #         # print('aug_instance', aug_instance)
                #
                #         aug_instance = self.preprocess(aug_instance, class_name, [class_name])
                #         token_ids, label_ids, B_mask, I_mask, att_mask, text_mask = self.tokenize(aug_instance, label2id)
                #         support_set['tokens'].append(token_ids)
                #         support_set['entity_label'].append(label_ids)
                #         support_set['B-mask'].append(B_mask)
                #         support_set['I-mask'].append(I_mask)
                #         support_set['att-mask'].append(att_mask)
                #         support_set['text-mask'].append(text_mask)
                #
                # else:
                #     instance = self.preprocess(self.raw_data[class_name][j], class_name, target_classes)
                #     token_ids, label_ids, B_mask, I_mask, att_mask, text_mask = self.tokenize(instance, label2id)
                #
                #     query_set['tokens'].append(token_ids)
                #     query_set['entity_label'].append(label_ids)
                #     query_set['B-mask'].append(B_mask)
                #     query_set['I-mask'].append(I_mask)
                #     query_set['att-mask'].append(att_mask)
                #     query_set['text-mask'].append(text_mask)
                # count += 1
        
        for k, v in support_set.items():
            support_set[k] = torch.stack(v)
        
        for k, v in query_set.items():
            query_set[k] = torch.stack(v)
        
        return support_set, query_set, id2label

    def preprocess(self, instance, relation_type, rel_type_list):
        result = {'tokens': [], 'entity_label': [], 'B-mask': [], 'I-mask': []}

        sentence = instance['tokens']
        result['tokens'] = sentence

        entity_label = ['O'] * len(sentence)
        B_mask = [0] * len(sentence)
        I_mask = [0] * len(sentence)

        h_entity_start = instance['h'][2][0][0]
        h_entity_end = instance['h'][2][0][-1] + 1

        t_entity_start = instance['t'][2][0][0]
        t_entity_end = instance['t'][2][0][-1] + 1

        #  头实体的标签序列
        for i in range(h_entity_start, h_entity_end):
            # 头实体所在位置，B_mask标记为1
            entity_label[i] = f"I-{relation_type}:HEAD"
            B_mask[i] = 1

        # 尾实体的标签序列
        for i in range(t_entity_start, t_entity_end):
            # 尾实体所在位置，I_mask标记为1
            entity_label[i] = f"I-{relation_type}:TAIL"
            I_mask[i] = 1

        result['entity_label'] = entity_label
        result['B-mask'] = B_mask
        result['I-mask'] = I_mask

        return result


def collate_fn(data):
    batch_support = {'tokens': [], 'entity_label': [], 'B-mask': [], 'I-mask': [], "att-mask": [], 'text-mask': []}
    batch_query = {'tokens': [], 'entity_label': [], 'B-mask': [], 'I-mask': [], "att-mask": [], 'text-mask': []}
    batch_id2label = []
    
    support_sets, query_sets, id2labels = zip(*data)

    # print('support_sets', support_sets)

    for i in range(len(support_sets)):
        for k in support_sets[i]:
            batch_support[k].append(support_sets[i][k])
        for k in query_sets[i]:
            batch_query[k].append(query_sets[i][k])
        batch_id2label.append(id2labels[i])

    for k in batch_support:
        batch_support[k] = torch.cat(batch_support[k], 0)
    for k in batch_query:
        batch_query[k] = torch.cat(batch_query[k], 0)

    return batch_support, batch_query, batch_id2label


def split_json_data(json_data: dict):
    train_data, dev_data, test_data = {}, {}, {}
    event_types = list(json_data.keys())
    random.shuffle(event_types)
    train_types = event_types[:80]
    dev_types = event_types[80: 90]
    test_types = event_types[90: 100]
    for k in train_types:
        train_data[k] = json_data[k]
    for k in dev_types:
        dev_data[k] = json_data[k]
    for k in test_types:
        test_data[k] = json_data[k]
    return train_data, dev_data, test_data


def get_loader(dataset_name,
               mode,
               max_length,
               tokenize,
               N, K, Q, O,
               batch_size,
               use_BIO=False,
               num_workers=0,
               collate_fn=collate_fn):
    root_data_dir = "data"

    if mode == "TRAIN":
        data_file = "meta_train_dataset.json"
    elif mode == "DEV":
        data_file = "meta_dev_dataset.json"
    elif mode == "TEST":
        data_file = "meta_test_dataset.json"
    else:
        raise ValueError("Error mode!")

    dataset_path = os.path.join(root_data_dir, "FewRel", data_file)
    dataset = FewEventDataset(dataset_path,
                              max_length,
                              tokenize,
                              N, K, Q, O, use_BIO)

    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=num_workers,
                            collate_fn=collate_fn)
    return iter(dataloader)

