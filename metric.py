# -*- coding: utf-8 -*-
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F


class Metric:
    def __init__(self):        
        self.pred = []
        self.true = []

    def update_state(self, preds, trues, id2labels, query_emb, prototype):
        batch_size = len(id2labels)
        
        _, seq_len = trues.shape
        preds = preds.view(batch_size, -1, seq_len)
        trues = trues.view(batch_size, -1, seq_len)

        preds = preds.cpu().tolist()
        trues = trues.cpu().tolist()
        
        for pred, true, id2label in zip(preds, trues, id2labels):
            pred = self.decode(pred, id2label)
            true = self.decode(true, id2label)

            # [PAD]部分不进行预测
            for i in range(len(pred)):
                for j in range(len(pred[i])):
                    if true[i][j] == 'PAD':
                        true[i][j] = 'O'
                        pred[i][j] = 'O'

            pred_triplet = self.extract(pred, query_emb, prototype)
            true_triplet = self.extract(true, query_emb, prototype)

            self.pred.extend(pred_triplet)
            self.true.extend(true_triplet)

    def result(self):
            return self.score(self.pred, self.true)

    def reset(self):
        self.pred = []
        self.true = []
        self.cnt = 0

    def decode(self, ids, id2label):
        labels = []
        for ins in ids:
            ins_labels = list(map(lambda x: id2label[x], ins))
            labels.append(ins_labels)
        return labels

    def extract(self, label_sequences, query_emb, prototype):

        results = []
        for i, instance_label in enumerate(label_sequences):
            spans = self.get_span(instance_label)
            result = []

            relations = {}
            for span in spans:
                relation, entity_type = span[0].split(":")

                if relation not in relations:
                    relations[relation] = {"HEAD": [], "TAIL": []}

                relations[relation][entity_type].append(span)

            for relation, entities in relations.items():
                heads = entities["HEAD"]
                tails = entities["TAIL"]

                if heads != [] and tails != []:
                    heads_embedding = []
                    tails_embedding = []

                    for head in heads:
                        star = head[1]
                        end = head[-1]+1
                        heads_embedding.append(query_emb[0][i][star:end].sum(0))
                    for tail in tails:
                        star = tail[1]
                        end = tail[-1]+1
                        tails_embedding.append(query_emb[0][i][star:end].sum(0))

                    heads_embedding = torch.stack(heads_embedding)
                    entity_prototype = prototype[0]
                    heads_embedding = heads_embedding.unsqueeze(1).expand(-1, entity_prototype.size(0), -1)
                    sim = 1e3 * F.cosine_similarity(heads_embedding, entity_prototype, dim=-1)
                    max_index = torch.argmax(sim)
                    max_index, _ = divmod(max_index.item(), sim.size()[1])
                    head = heads[max_index]

                    tails_embedding = torch.stack(tails_embedding)
                    tails_embedding = tails_embedding.unsqueeze(1).expand(-1, entity_prototype.size(0), -1)
                    sim = 1e3 * F.cosine_similarity(tails_embedding, entity_prototype, dim=-1)
                    max_index = torch.argmax(sim)
                    max_index, _ = divmod(max_index.item(), sim.size()[1])
                    tail = tails[max_index]

                    triplet = (self.cnt, i, head, tail)
                    result.append(triplet)
            results.extend(result)

            #     for head in heads:
            #         for tail in tails:
            #             triplet = (self.cnt, i, head, tail)
            #             result.append(triplet)
            # results.extend(result)

        return results

    def score(self, pred_tags, true_tags):

        pred_triplets = set(self.pred)
        true_triplets = set(self.true)

        pred_correct = len(true_triplets & pred_triplets)
        pred_all = len(pred_triplets)
        true_all = len(true_triplets)

        p = pred_correct / pred_all if pred_all > 0 else 0
        r = pred_correct / true_all if true_all > 0 else 0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0
        ti_p, ti_r, ti_f1 = 0.0, 0.0, 0.0

        return p, r, f1, ti_p, ti_r, ti_f1

    def get_span(self, seq):
        if any(isinstance(s, list) for s in seq):
            seq = [item for sublist in seq for item in sublist + ['O']]
        
        prev_tag = 'O'
        prev_type = ''
        begin_offset = 0
        chunks = []
        for i, chunk in enumerate(seq + ['O']):
            tag = chunk[0]
            type_ = chunk.split('-')[-1]
    
            if self.end_of_span(prev_tag, tag, prev_type, type_):
                chunks.append((prev_type, begin_offset, i-1))
            if self.start_of_span(prev_tag, tag, prev_type, type_):
                begin_offset = i
            prev_tag = tag
            prev_type = type_
    
        return chunks
    
    def start_of_span(self, prev_tag, tag, prev_type, type_):
        chunk_start = False
    
        if tag == 'B': chunk_start = True
        if tag == 'S': chunk_start = True
    
        if prev_tag == 'E' and tag == 'E': chunk_start = True
        if prev_tag == 'E' and tag == 'I': chunk_start = True
        if prev_tag == 'S' and tag == 'E': chunk_start = True
        if prev_tag == 'S' and tag == 'I': chunk_start = True
        if prev_tag == 'O' and tag == 'E': chunk_start = True
        if prev_tag == 'O' and tag == 'I': chunk_start = True
    
        if tag != 'O' and tag != '.' and prev_type != type_:
            chunk_start = True
    
        return chunk_start
    
    def end_of_span(self, prev_tag, tag, prev_type, type_):
        chunk_end = False
    
        if prev_tag == 'E': chunk_end = True
        if prev_tag == 'S': chunk_end = True
    
        if prev_tag == 'B' and tag == 'B': chunk_end = True
        if prev_tag == 'B' and tag == 'S': chunk_end = True
        if prev_tag == 'B' and tag == 'O': chunk_end = True
        if prev_tag == 'I' and tag == 'B': chunk_end = True
        if prev_tag == 'I' and tag == 'S': chunk_end = True
        if prev_tag == 'I' and tag == 'O': chunk_end = True
    
        if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
            chunk_end = True
    
        return chunk_end
