import json

dataset_path = 'data/FewRel/meta_train_dataset.json'

new_event_train = 'data/eventTest/meta_train_dataset.json'

data = json.load(open(dataset_path, "r", encoding='utf-8'))

all_data = {}
for key, values in data.items():
    all_data[key] = {}
    for i in range(len(values)):
        print('values', values[i])
        sing_data = {}
        sing_data['tokens'] = values[i]['tokens']
        sing_data['trigger'] = sing_data['tokens'][values[i]['h'][2][0][0]]
        sing_data['position'] = [values[i]['h'][2][0][0], values[i]['h'][2][0][0]+1]
        print('sing_data', sing_data)

    # sing_data = {}
    # sing_data['tokens'] = data[key][]
    # print('111111111', all_data)
    break


