# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from contrastive_loss import SupConLoss, ProtoConLoss, distance, gaussian


def dropout_augmentation(token_ids):
    return NotImplementedError


class ProtoDot(nn.Module):

    def __init__(self, encoder, opt):
        super(ProtoDot, self).__init__()

        self.feature_size = opt.feature_size
        self.max_len = opt.max_length
        self.distance_metric = opt.distance_metric

        self.encoder = encoder
        # self.encoder = nn.DataParallel(self.encoder)
        self.projection_head = nn.Sequential(
            nn.Linear(self.feature_size, self.feature_size, bias=False),
            nn.ReLU(),
            nn.Linear(self.feature_size, self.feature_size, bias=False)
        )
        self.similarity_mlp = nn.Sequential(
            nn.Linear(opt.trainN + 1, opt.trainN + 1),
            nn.ReLU(),
            nn.Linear(opt.trainN + 1, opt.trainN + 1),
            nn.Sigmoid()
        )

        # self attention
        self.Wk = nn.Linear(self.feature_size, self.feature_size)
        self.Wq = nn.Linear(self.feature_size, self.feature_size)
        self.Wv = nn.Linear(self.feature_size, self.feature_size)

        with torch.no_grad():
            pad_embedding = self.encoder(torch.LongTensor([[0]]))[0].view(self.feature_size)
            pad_embedding = pad_embedding.repeat(opt.O, 1)
        # self.other_prototype = nn.Parameter(torch.randn(opt.O, self.feature_size))

        self.cost = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
        self.contrastive_cost = SupConLoss(temperature=opt.temperature_alpha)
        self.proto_contrastive_cost = ProtoConLoss(temperature=opt.temperature_beta)

        self.dropout_rate = opt.dropout
        self.drop = nn.Dropout(self.dropout_rate)
        self.use_BIO = opt.use_BIO
        self.O = opt.O
        self.contrastive = opt.contrastive
        self.alpha = opt.alpha
        self.beta = opt.beta
        self.threshold = opt.threshold

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=5,
                out_channels=8,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=2
            ),
            nn.Dropout(self.dropout_rate),

            nn.Conv1d(
                in_channels=8,
                out_channels=4,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool1d(
                kernel_size=2
            ),
            nn.Dropout(self.dropout_rate)
        )
        self.fc = nn.Linear(768, 1)

    def get_embedding(self, support_set, query_set):
        # encode
        support_emb = self.encoder(support_set['tokens'], attention_mask=support_set["att-mask"])[
            0]  # B*N*K, max_len, feature_size
        query_emb = self.encoder(query_set['tokens'], attention_mask=query_set['att-mask'])[
            0]  # B*N*K, max_len, feature_size

        # dropout
        support_emb = self.drop(support_emb)  # B*N*K, max_len, feature_size
        query_emb = self.drop(query_emb)  # B*N*K, max_len, feature_size
        return support_emb, query_emb

    def forward(self, support_set, query_set, N, K, Q, O, epoch):
        support_emb, query_emb = self.get_embedding(support_set, query_set)

        aug_k = 5 * K
        support_emb = support_emb.view(-1, N, aug_k, self.max_len, self.feature_size)  # B, N, K, max_len, feature_size
        query_emb = query_emb.view(-1, N * Q, self.max_len, self.feature_size)  # B, N*Q, max_len, feature_size

        B_mask = support_set['B-mask'].view(-1, N, aug_k, self.max_len)  # B, N, K, max_len
        I_mask = support_set['I-mask'].view(-1, N, aug_k, self.max_len)  # B, N, K, max_len

        text_mask = support_set['text-mask'].view(-1, N, aug_k, self.max_len)  # B, N, K, max_len

        # prototype
        prototype = self.proto(support_emb, B_mask, I_mask, text_mask)  # B, 2*N+1, feature_size

        # print(epoch)
        if epoch >= 0:
            # pre_logits
            logits, other_max_sim_index = self.similarity(prototype, query_emb, N, K, Q, O)  # B, N*Q, max_len, 2*N+1
            _, pred = torch.max(logits.view(-1, logits.shape[-1]), 1)  # B*N*Q*max_len
            # get query_mask
            pred = pred.view(-1, self.max_len)
            # print('mask_pred1', pred)
            mask_pred, query_text_mask = self.get_pred_mask(pred, query_set['entity_label'])
            query_prototype = self.get_query_proto(query_emb, mask_pred, query_text_mask, prototype)
            prototype = prototype + query_prototype

            # a = prototype.unsqueeze(1)
            # b = query_prototype.unsqueeze(2)
            # true_query_prototype = self.get_query_proto(query_emb, query_set['entity_label'], query_text_mask, prototype)
            # result1 = F.cosine_similarity(true_query_prototype, prototype, dim=-1)
            # result2 = F.cosine_similarity(true_query_prototype, query_prototype, dim=-1)
            # result3 = F.cosine_similarity(true_query_prototype, prototype + query_prototype, dim=-1)
            # # result_data = result1[0]
            # print('result111', result1[0])
            # print('result222', result2[0])
            # print('result333', result3[0])

            # result = F.cosine_similarity(prototype, query_prototype, dim=-1)
            # result_data = result[0]
            # res = torch.where(result_data > 0)[0]
            # res = res.tolist()
            # # print('res', res)
            # # print('result', result)
            # if res != []:
            #     for i in res:
            #         prototype[0][i] = 1e-8
            # result3 = F.cosine_similarity(true_query_prototype, prototype + query_prototype, dim=-1)
            # print('result333', result3[0])

            # prototype = prototype + query_prototype

        # classification
        logits, other_max_sim_index = self.similarity(prototype, query_emb, N, K, Q, O)  # B, N*Q, max_len, 2*N+1
        _, pred = torch.max(logits.view(-1, logits.shape[-1]), 1)  # B*N*Q*max_len

        outputs = (logits, pred, support_emb, query_emb, prototype)

        # loss
        if query_set['entity_label'] is not None:
            loss = self.loss(logits, query_set['entity_label'])

            # 计算对比损失
            if self.contrastive != "None":
                # 使用线性层进行特征变换
                support_emb_for_contrast = self.projection_head(support_emb).view(-1, self.feature_size)
                query_emb_for_contrast = self.projection_head(query_emb).view(-1, self.feature_size)

                prototype_for_contrast = self.projection_head(prototype)

                # support_emb_aug, query_emb_aug = self.get_embedding(support_set, query_set)
                # support_emb_for_contrast = self.projection_head(torch.cat([support_emb.view(support_emb_aug.shape), support_emb_aug]))
                # query_emb_for_contrast = self.projection_head(torch.cat([query_emb, query_emb_aug]))
                prototype = prototype.view(-1, self.feature_size)
                l2_prototype = F.normalize(prototype, p=2, dim=1)

                # label_similarity = 1 - torch.matmul(l2_prototype, l2_prototype.T)
                # label_similarity = torch.exp(label_similarity * label_similarity / -2)
                # label_similarity = torch.matmul(l2_prototype, l2_prototype.T)
                # label_similarity = self.similarity_mlp(label_similarity)
                label_similarity = torch.ones(N + 1, N + 1).to(prototype)  # N+1, N+1
                label_similarity[0, :] = 2.0
                label_similarity[:, 0] = 2.0
                label_similarity[0][0] = 1.0

                # print("label_similarity: ", label_similarity)
                contrastive_loss = self.contrastive_loss(torch.cat([support_emb_for_contrast]),
                                                         torch.cat([support_set["entity_label"]]),
                                                         label_similarity, self.contrastive)
                proto_contrastive_loss = self.prototypical_contrastive_loss(query_emb_for_contrast,
                                                                            query_set["entity_label"],
                                                                            prototype_for_contrast, label_similarity,
                                                                            other_max_sim_index,
                                                                            self.contrastive)
                outputs = (loss + self.alpha * contrastive_loss + self.beta * proto_contrastive_loss,) + outputs

            else:
                outputs = (loss,) + outputs

        return outputs

    def get_pred_mask(self, pred, true):

        # 将entity_label中值为-100的位置找出来
        query_false_mask = (true == -100)
        query_true_mask = (true != -100)
        # 使用torch.where替换对应位置的值
        pred = torch.where(query_false_mask, true, pred)
        return pred, query_true_mask

    def get_span_index(self, input_index):
        # 寻找连续的子张量
        sub_tensors = []
        start_idx = 0

        for i in range(1, len(input_index)):
            if input_index[i] != input_index[i - 1] + 1:
                sub_tensors.append(input_index[start_idx:i])
                start_idx = i

        # 添加最后一个子张量
        sub_tensors.append(input_index[start_idx:])
        return sub_tensors

    def similarity(self, prototype, query, N, K, Q, O):
        '''
        inputs:
            prototype_h: B, 2*N+1, feature_size或者B, N+1, feature_size
            prototype_t: B, 2*N+1, feature_size或者B, N+1, feature_size
            query: B, N*Q, max_len, feature_size
        outputs:
            sim: B, N*Q, max_len, 2*N+1或者B, N*Q, max_len, N+1
        '''
        tag_num = prototype.shape[1]
        query_num = query.shape[1]

        query = query.unsqueeze(-2)  # B, N*Q, max_len, 1, feature_size
        query = query.expand(-1, -1, -1, tag_num, -1)  # B, N*Q, max_len, 2*N+1/N+1, feature_size

        prototype = prototype.unsqueeze(1)  # B, 1, 2*N+1, feature_size
        prototype = prototype.unsqueeze(2)  # B, 1, 1, 2*N+1, feature_size
        prototype = prototype.expand(-1, query_num, self.max_len, -1, -1)  # B, N*Q, max_len, 2*N+1/N+1, feature_size

        if self.distance_metric == "dot":
            sim = (prototype * query).sum(-1)  # B, N*Q, max_len, 2*N+O/N+O

        elif self.distance_metric == "match":
            sim = 1e3 * F.cosine_similarity(prototype, query, dim=-1)  # B, N*Q, max_len, 2*N+O/N+O

        elif self.distance_metric == "conv":
            # conv relnet
            minus = (query - prototype).abs()  # B, NQ, N, feature_size
            add = query + prototype
            mul = query * prototype

            inputs = torch.stack([prototype, query, minus, add, mul], dim=-2)  # B, NQ, max_len, 2*N+O, 5, feature_size
            original_shape = inputs.shape
            inputs = inputs.view(-1, 5, self.feature_size)                      # B*NQ*max_len*(2*N+O), 5, feature_size
            out = self.conv(inputs)                                             # B*NQ*max_len*(2*N+O), 4, 192
            out = out.view(*original_shape[:4], -1)                             # B, NQ, max_len, 2*N+O, 768
            sim = self.fc(out).squeeze(-1)                                      # B, NQ, max_len, 2*N+O

        else:
            sim = -(torch.pow(query - prototype, 2)).sum(-1)  # B, N*Q, max_len, 2*N+O/N+O

        # 对于前O个Other类的prototype，选择最高的sim作为logit
        B, NQ, max_len, _ = sim.shape
        new_sim = torch.zeros(B, NQ, max_len, 2 * N + 1)
        new_sim = new_sim.to(sim)

        if self.O > 0:
            new_sim[:, :, :, 0] = torch.max(torch.mean(sim[:, :, :, :O + 1].view(-1, O + 1), dim=0))
            other_max_sim_index = torch.argmax(sim[:, :, :, :O + 1])
        else:
            if self.threshold == "mean":
                new_sim[:, :, :, 0] = torch.mean(sim[:, :, :, 0])
            elif self.threshold == "max":
                new_sim[:, :, :, 0] = torch.max(sim[:, :, :, 0])
            elif self.threshold == "med":
                new_sim[:, :, :, 0] = torch.median(torch.stack(sorted(sim[:, :, :, 0])))
            else:
                new_sim[:, :, :, 0] = sim[:, :, :, 0]

            other_max_sim_index = 0
        new_sim[:, :, :, 1:] = sim[:, :, :, O + 1:]

        return new_sim, other_max_sim_index

    def get_query_proto(self, query_emb, query_mask, text_mask, prototype):
        fill_num = 1e-8

        B, N, _, _ = query_emb.shape
        query_emb = query_emb.unsqueeze(2)
        query_emb = query_emb.view(B, N, -1, self.max_len, self.feature_size)  # B, N, Q, max_len, feature_size
        prototype = prototype.view(-1, self.feature_size)
        query_prototype = torch.full((B, 2 * N + self.O + 1, self.feature_size), fill_value=fill_num).to(query_emb)  # B, 2N+1, feature_size

        # print('query_mask', query_mask)

        query_mask = query_mask.unsqueeze(0)
        query_mask = query_mask.unsqueeze(2)
        query_mask = query_mask.unsqueeze(-1)
        query_mask = query_mask.expand(-1, -1, -1, -1, self.feature_size)
        query_mask = query_mask.to(query_emb)  # B, N, Q, max_len, feature_size

        text_mask = text_mask.unsqueeze(0)
        text_mask = text_mask.unsqueeze(2)
        text_mask = text_mask.unsqueeze(-1)
        text_mask = text_mask.expand(-1, -1, -1, -1, self.feature_size)
        text_mask = text_mask.to(query_emb)  # B, N, Q, max_len, feature_size

        for i in range(B):

            for j in range(N):
                head_prototype = prototype[2 * j + 1]
                tail_prototype = prototype[2 * j + 1 + 1]
                query_mask_h = (query_mask == 2 * j + self.O + 1)
                query_mask_t = (query_mask == 2 * j + self.O + 1 + 1)
                # 头实体
                # sum_B_fea = (query_emb[i] * query_mask_h[i]).view(-1, self.feature_size).sum(0)
                # num_B_fea = query_mask_h[i].sum() / self.feature_size + fill_num
                # query_prototype[i, 2 * j + self.O + 1] = sum_B_fea / num_B_fea
                sum_H_fea = (query_emb[i] * query_mask_h[i]).view(-1, self.feature_size)  # N, Q, max_len, feature_size
                sim_h = F.cosine_similarity(head_prototype, sum_H_fea, dim=-1)
                the_index = torch.where(sim_h > 0.0)[0]
                if len(the_index) > 0:
                    theP_h = torch.median(torch.stack(sorted(sim_h[the_index])))
                    sim_h_indices = torch.where(sim_h > 0)
                    # print('theP', theP)
                    # print('sim_h', sim_h)
                    # print('sim_h_indices', sim_h_indices)

                    # 存在符合阈值的表示
                    if len(sim_h_indices[0]) > 0:
                        sum_H_fea = sum_H_fea[sim_h_indices].sum(0)
                        num_H_fea = len(sim_h_indices[0])
                        # print('sum_H_fea', sum_H_fea.size())
                        # print('num_H_fea', num_H_fea)
                        query_prototype[i, 2 * j + self.O + 1] = sum_H_fea / num_H_fea

                # 尾实体
                # sum_I_fea = (query_emb[i] * query_mask_t[i]).view(-1, self.feature_size).sum(0)
                # num_I_fea = query_mask_t[i].sum() / self.feature_size + fill_num
                # query_prototype[i, 2 * j + self.O + 1 + 1] = sum_I_fea / num_I_fea

                sum_T_fea = (query_emb[i] * query_mask_t[i]).view(-1, self.feature_size)
                sim_t = F.cosine_similarity(tail_prototype, sum_T_fea, dim=-1)
                the_index = torch.where(sim_t > 0)[0]
                if len(the_index) > 0:
                    theP_t = torch.median(torch.stack(sorted(sim_t[the_index])))
                    sim_t_indices = torch.where(sim_t > 0)

                    # 存在符合阈值的表示
                    if len(sim_t_indices[0]) > 0:
                        sum_T_fea = sum_T_fea[sim_t_indices].sum(0)
                        num_T_fea = len(sim_t_indices[0])
                        query_prototype[i, 2 * j + self.O + 1 + 1] = sum_T_fea / num_T_fea

            O_mask = (query_mask == 0)  # N, K, max_len, feature_size
            O_mask = O_mask * text_mask[i]
            sum_O_fea = (query_emb[i] * O_mask).reshape(-1, self.feature_size).sum(0)
            num_O_fea = O_mask.sum() / self.feature_size + 1e-8
            query_prototype[i, 0] = sum_O_fea / num_O_fea

        return query_prototype

    def proto(self, support_emb, B_mask, I_mask, text_mask):
        '''
        input:
            support_emb : B, N, K, max_len, feature_size
            B_mask : B, N, K, max_len
            I_mask : B, N, K, max_len
            att_mask: B, N, K, max_len
        output:
            prototype : B, 2*N+1, feature_size # (class_num -> 2N + 1)
        '''
        B, N, K, _, _ = support_emb.shape
        prototype = torch.empty(B, 2 * N + self.O + 1, self.feature_size).to(support_emb)  # B, 2N+1, feature_size

        text_mask = text_mask.unsqueeze(-1)
        text_mask = text_mask.expand(-1, -1, -1, -1, self.feature_size)
        text_mask = text_mask.to(support_emb)  # B, N, K, max_len, feature_size
        # 头实体和尾实体mask
        B_mask = B_mask.unsqueeze(-1)
        B_mask = B_mask.expand(-1, -1, -1, -1, self.feature_size)
        B_mask = B_mask.to(support_emb)  # B, N, K, max_len, feature_size
        I_mask = I_mask.unsqueeze(-1)
        I_mask = I_mask.expand(-1, -1, -1, -1, self.feature_size)
        I_mask = I_mask.to(support_emb)  # B, N, K, max_len, feature_size

        # 实体原型计算
        for i in range(B):
            O_mask = torch.ones_like(B_mask[i]).to(B_mask)  # N, K, max_len, feature_size
            O_mask -= B_mask[i] + I_mask[i]
            O_mask = O_mask * text_mask[i]
            for j in range(N):
                # 头实体
                sum_B_fea = (support_emb[i, j] * B_mask[i, j]).view(-1, self.feature_size).sum(0)
                num_B_fea = B_mask[i, j].sum() / self.feature_size + 1e-8
                prototype[i, 2 * j + self.O + 1] = sum_B_fea / num_B_fea

                # 尾实体
                sum_I_fea = (support_emb[i, j] * I_mask[i, j]).view(-1, self.feature_size).sum(0)
                num_I_fea = I_mask[i, j].sum() / self.feature_size + 1e-8
                prototype[i, 2 * j + self.O + 1 + 1] = sum_I_fea / num_I_fea

            # 先用Other类的embedding计算一个prototype
            sum_O_fea = (support_emb[i] * O_mask).reshape(-1, self.feature_size).sum(0)
            num_O_fea = O_mask.sum() / self.feature_size + 1e-8
            prototype[i, 0] = sum_O_fea / num_O_fea
            # prototype[i, 0] = 1e-8
            # 对Other类做聚类
            # 如果存在辅助向量，则加到prototype后面
            if self.O >= 1:
                raise NotImplementedError("O must be set to 0")
        return prototype

    def proto_interaction(self, prototype):
        # self attention
        K = self.Wk(prototype)  # B, 2*N+1, feature_size
        Q = self.Wq(prototype)  # B, 2*N+1, feature_size
        V = self.Wv(prototype)  # B, 2*N+1, feature_size

        att_score = torch.matmul(K, Q.transpose(-1, -2))  # B, 2*N+1, 2*N+1
        att_score /= torch.sqrt(torch.tensor(self.feature_size).to(K))  # B, 2*N+1, 2*N+1
        att_score = att_score.softmax(-1)  # B, 2*N+1, 2*N+1

        prototype = torch.matmul(att_score, V)  # B, 2*N+1, feature_size
        return prototype

    def loss(self, logits, label):
        logits = logits.view(-1, logits.shape[-1])
        label = label.view(-1)

        loss_weight = torch.ones_like(label).float()
        loss = self.cost(logits, label)
        loss = (loss_weight * loss).mean()
        return loss

    def contrastive_loss(self, support_features, support_labels, label_similarity, contrastive="Normal"):
        """
        compute contrastive loss
        :param ignore_padding: 无视padding
        :param features: B, N, K, max_len, feature_size
        :param labels: B*N*K*max_len
        :return: loss
        """
        support_features = support_features.view(-1, 1, self.feature_size)
        support_labels = support_labels.view(-1)
        # query_features = support_features.view(-1, 1, self.feature_size)
        # query_labels = support_labels.view(-1)
        # delete Other and PAD label
        support_features = support_features[support_labels != -100].view(-1, 1, self.feature_size)
        support_labels = support_labels[support_labels != -100].view(-1)

        # support_features = support_features[support_labels != 0].view(-1, 1, self.feature_size)
        # support_labels = support_labels[support_labels != 0].view(-1)
        # query_features = query_features[query_labels != -100].view(-1, 1, self.feature_size)
        # query_labels = query_labels[query_labels != -100].view(-1)
        # query_features = query_features[query_labels != 0].view(-1, 1, self.feature_size)
        # query_labels = query_labels[query_labels != 0].view(-1)
        # L2 Normalize
        support_features = F.normalize(support_features, p=2, dim=2)
        # query_features = F.normalize(query_features, p= 2, dim=2)
        contrastive_loss = self.contrastive_cost(label_similarity, support_features, support_labels,
                                                 contrastive=contrastive)
        # print("contrastive_loss: ", contrastive_loss.item())
        return contrastive_loss

    def prototypical_contrastive_loss(self, query_features, query_labels, prototypes, label_similarity,
                                      other_max_sim_index,
                                      contrastive="Normal"):
        # Other类取相似度最高的那个prototype参与计算
        # print("other_max_sim_index: ", other_max_sim_index)
        prototypes = torch.cat([prototypes[:, 0: self.O + 1, :], prototypes[:, self.O + 1:, :]], dim=1)
        query_features = query_features.view(-1, self.feature_size)
        prototypes = prototypes.view(-1, self.feature_size)
        query_labels = query_labels.view(-1)
        query_features = query_features[query_labels != -100].view(-1, self.feature_size)
        query_labels = query_labels[query_labels != -100].view(-1)
        # L2 norm
        query_features = F.normalize(query_features, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=1)
        proto_contrastive_loss = self.proto_contrastive_cost(query_features, query_labels, prototypes, self.O + 1,
                                                             contrastive=contrastive)
        return proto_contrastive_loss
