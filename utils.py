# -*- coding: utf-8 -*-
import json
import os
from tqdm import tqdm
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json


entityPath = './data/FewRel/entity_data.json'
entityDic = json.load(open(entityPath, "r", encoding='utf-8'))


def get_entitys(entitys, entity_len):
    entity_list = []
    for entity in entitys:
        if len(entity.split()) == entity_len:
            entity_list.append(entity)
    return entity_list


def get_aug_instance(class_name, instance, qyery_h_entitys, qyery_t_entitys):

    if len(instance['h'][2][0]) == 1:
        h_entity_len = 1
    else:
        h_entity_len = instance['h'][2][0][-1] - instance['h'][2][0][0] + 1

    if len(instance['t'][2][0]) == 1:
        t_entity_len = 1
    else:
        t_entity_len = instance['t'][2][0][-1] - instance['t'][2][0][0] + 1

    head_entity_list = entityDic[class_name]['head_entity']
    tail_entity_list = entityDic[class_name]['tail_entity']

    if qyery_h_entitys in head_entity_list:
        head_entity_list.remove(qyery_h_entitys)

    if qyery_t_entitys in tail_entity_list:
        tail_entity_list.remove(qyery_t_entitys)

    head_len_entitys = get_entitys(head_entity_list, h_entity_len)
    tail_len_entitys = get_entitys(tail_entity_list, t_entity_len)

    if head_len_entitys == [] or tail_len_entitys == []:
        return instance

    headAugEntity = np.random.choice(head_len_entitys, 1, False)
    tailAugEntity = np.random.choice(tail_len_entitys, 1, False)

    headAugEntity = headAugEntity[0]
    tailAugEntity = tailAugEntity[0]

    head_list, tail_list = headAugEntity.split(), tailAugEntity.split()
    h_entity_len, t_entity_len = len(head_list), len(tail_list)

    for i in range(h_entity_len):
        index = instance['h'][2][0][0] + i
        instance['tokens'][index] = head_list[i]

    for i in range(t_entity_len):
        index = instance['t'][2][0][0] + i
        instance['tokens'][index] = tail_list[i]

    instance['h'][0] = headAugEntity
    instance['t'][0] = tailAugEntity

    return instance



def statistics_of_fewevent(data_dir):
    """
    统计数据集的各种信息
    绘制直方图以表示数据集分布
    :param data_dir:
    :return:
    """
    train_set_path = os.path.join(data_dir, "meta_train_dataset.json")
    dev_set_path = os.path.join(data_dir, "meta_dev_dataset.json")
    test_set_path = os.path.join(data_dir, "meta_test_dataset.json")
    all_samples = []
    all_length = []
    max_len = 0
    sum_len = 0
    multi_token_trigger_num = 0
    all_trigger_num = 0
    sent_longer_than_128 = 0
    sent_longer_than_64 = 0
    statistics = {
        "max_len": 0,
        "avg_len": 0,
        "multi_token_trigger_percentage": 0.0,
        "sent_longer_than_128_percentage": 0.0,
        "sent_longer_than_64_percentage": 0.0,
        "trigger_num": 0,
        "sent_num": 0
    }
    for file_path in [train_set_path, dev_set_path, test_set_path]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for event_type, event_instances in data.items():
                all_samples.extend(event_instances)
                for sample in event_instances:
                    if len(sample["tokens"]) <= 9:
                        print(sample["tokens"], sample["trigger"], event_type)
    for sample in tqdm(all_samples):
        if len(sample["tokens"]) < 128:
            all_length.append(len(sample["tokens"]))
        if len(sample["tokens"]) > max_len:
            max_len = len(sample["tokens"])
        if len(sample["tokens"]) > 128:
            sent_longer_than_128 += 1
        if len(sample["tokens"]) > 64:
            sent_longer_than_64 += 1
        sum_len += len(sample["tokens"])
        if len(sample["trigger"]) > 1:
            multi_token_trigger_num += 1
        all_trigger_num += len(sample["trigger"])
    statistics["max_len"] = max_len
    statistics["avg_len"] = sum_len / len(all_samples)
    statistics["multi_token_trigger_percentage"] = multi_token_trigger_num / all_trigger_num
    statistics["sent_longer_than_128_percentage"] = sent_longer_than_128 / len(all_samples)
    statistics["sent_longer_than_64_percentage"] = sent_longer_than_64 / len(all_samples)
    statistics["trigger_num"] = all_trigger_num
    statistics["sent_num"] = len(all_samples)
    # 绘制直方图，统计句子长度的分布情况
    all_length = np.array(all_length, dtype=np.int64)
    fig, ax = plt.subplots(1, 1)
    sns.distplot(all_length, kde=False)
    plt.xlabel("length")
    plt.ylabel("Number")
    plt.savefig("img/length_distribution.svg", format="svg", bbox_inches="tight")
    plt.show()
    plt.close()
    return statistics


def t_test(x1, x2):
    t_score, p_value = ttest_ind(x1, x2, equal_var=True)
    return {"t_score": t_score, "p_value": p_value}


if __name__ == "__main__":
    statistics = statistics_of_fewevent("data/FewEvent")
    print(statistics)
    our_scores = np.array([64.88, 61.31, 63.48, 61.97, 65.84])
    our_v2_scores = np.array([65.59, 65.30, 64.32, 64.59, 65.34])
    print(t_test(our_scores, our_v2_scores))
