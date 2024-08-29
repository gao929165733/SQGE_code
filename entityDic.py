import json

train_path = 'data/FewRel/meta_train_dataset.json'
test_path = 'data/FewRel/meta_test_dataset.json'
dev_path = 'data/FewRel/meta_dev_dataset.json'

train_data = json.load(open(train_path, "r", encoding='utf-8'))
test_data = json.load(open(test_path, "r", encoding='utf-8'))
dev_data = json.load(open(dev_path, "r", encoding='utf-8'))

train_dic = {}
for rel, values in train_data.items():
    train_dic[rel] = {}
    train_dic[rel]['head_entity'] = []
    train_dic[rel]['tail_entity'] = []
    for singleData in train_data[rel]:
        train_dic[rel]['head_entity'].append(singleData['h'][0])
        train_dic[rel]['tail_entity'].append(singleData['t'][0])

test_dic = {}
for rel, values in test_data.items():
    test_dic[rel] = {}
    test_dic[rel]['head_entity'] = []
    test_dic[rel]['tail_entity'] = []
    for singleData in test_data[rel]:
        test_dic[rel]['head_entity'].append(singleData['h'][0])
        test_dic[rel]['tail_entity'].append(singleData['t'][0])

dev_dic = {}
for rel, values in dev_data.items():
    dev_dic[rel] = {}
    dev_dic[rel]['head_entity'] = []
    dev_dic[rel]['tail_entity'] = []
    for singleData in dev_data[rel]:
        dev_dic[rel]['head_entity'].append(singleData['h'][0])
        dev_dic[rel]['tail_entity'].append(singleData['t'][0])

# 字典合并
all_entity_dic = {**train_dic, **test_dic, **dev_dic}

# 写入文件
json_entity = json.dumps(all_entity_dic)
with open('./data/FewRel/entity_data.json', 'w') as json_file:
    json_file.write(json_entity)






