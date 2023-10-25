import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from classifier.base import BASE


class Contrast_Loss(nn.Module):
    def __init__(self, temperature=0.07):
        super(Contrast_Loss, self).__init__()

        self.cross_entropy = nn.CrossEntropyLoss()
        self.tao = temperature

    def forward(self, feat1, feat2):
        scores1 = torch.bmm(feat1.unsqueeze(1), feat2.unsqueeze(-1)).squeeze(-1)
        scores2 = F.cosine_similarity(feat1.unsqueeze(1), feat1.unsqueeze(0), dim=-1)
        diagonal = torch.diag(scores2)
        scores2 = scores2 - torch.diag(diagonal)

        scores = torch.cat([scores1, scores2], dim=1) / self.tao

        labels = torch.zeros(scores.shape[0], device=feat1.device).long()

        loss = self.cross_entropy(scores, labels)

        return loss


class ProtoMulHeadAttn(nn.Module):
    def __init__(self, args, dim):
        super(ProtoMulHeadAttn, self).__init__()

        self.way = args.way
        self.shot = args.shot

        self.head_num = args.head_num
        self.head_dim = dim // self.head_num

        self.q_fc = nn.Linear(dim, dim)
        self.k_fc = nn.Linear(dim, dim)
        self.v_fc = nn.Linear(dim, dim)

        self.norm = nn.LayerNorm(dim)

        # self.proto_fc1 = nn.Linear(2 * dim, dim)
        self.proto_fc2 = nn.Linear(dim, 1)

        self.output = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )

    def forward(self, x, flag='XQ', mask=None):

        if flag == 'XS':
            x = x.reshape(self.way, self.shot, -1, x.size(-1))
            way, shot, l, h = x.size()

            Q = self.q_fc(x).view(way, shot, l, self.head_num, self.head_dim).permute(0, 3, 1, 2,
                                                                                      4)  # [way, head_num, shot, l, h]
            K = self.k_fc(x).view(way, shot, l, self.head_num, self.head_dim).permute(0, 3, 1, 2,
                                                                                      4)  # [way, head_num, shot, l, h]
            V = self.v_fc(x).view(way, shot, l, self.head_num, self.head_dim).permute(0, 3, 1, 2,
                                                                                      4)  # [way, head_num, shot, l, h]

            scores1 = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
            scores2 = torch.matmul(Q.permute(0, 1, 4, 2, 3), K.permute(0, 1, 4, 3, 2)) / (self.head_dim ** 0.5)

            if mask is not None:
                scores1 = scores1.masked_fill(mask.bool(), float('-inf'))
                scores2 = scores2.masked_fill(mask.bool(), float('-inf'))

            weights1 = F.softmax(scores1, dim=-1)
            weights2 = F.softmax(scores2, dim=-1)

            attn_out1 = torch.matmul(weights1, V).permute(0, 2, 3, 1, 4).contiguous().view(way, shot, l, -1).transpose(
                -2, -1)
            attn_out2 = torch.matmul(weights2, V.permute(0, 1, 4, 2, 3)).permute(0, 3, 4, 1, 2).contiguous().view(way,
                                                                                                                  shot,
                                                                                                                  l,
                                                                                                                  -1).transpose(
                -2, -1)

            attn_out1 = F.max_pool1d(attn_out1.view(way * shot, h, l), l).squeeze(-1).view(way, shot, h)
            attn_out2 = F.max_pool1d(attn_out2.view(way * shot, h, l), l).squeeze(-1).view(way, shot, h)

            attn_out1 = self.norm(attn_out1)
            attn_out2 = self.norm(attn_out2)
            # attn_out = torch.cat([attn_out1, attn_out2], dim=-1) # [way, shot, 2h]

            attn_out = (attn_out1 + attn_out2) / 2

            proto = torch.tanh(attn_out)
            proto_weight = F.softmax(self.proto_fc2(proto).squeeze(-1), dim=-1).unsqueeze(-1)
            proto = torch.mul(proto_weight, attn_out)
            proto = self.output(proto)
            proto = torch.sum(proto, dim=1)
            return proto

        else:

            b, l, h = x.size()

            q = self.q_fc(x).view(b, l, self.head_num, self.head_dim).transpose(1, 2)
            k = self.k_fc(x).view(b, l, self.head_num, self.head_dim).transpose(1, 2)
            v = self.v_fc(x).view(b, l, self.head_num, self.head_dim).transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

            if mask is not None:
                scores = scores.masked_fill(mask.bool(), float('-inf'))

            weights = F.softmax(scores, dim=-1)

            attn_out = torch.matmul(weights, v).transpose(1, 2).contiguous().view(b, l, -1)
            attn_out = self.norm(attn_out)

            output = self.output(attn_out)
            output = torch.sum(output, dim=1)

            return output


class ProtoAttn(BASE):
    def __init__(self, ebd, args):
        super(ProtoAttn, self).__init__(args)

        self.ebd = ebd

        self.dim = self.ebd.embedding_dim

        self.model = ProtoMulHeadAttn(args, self.dim)

        self.dropout = nn.Dropout(0.2)

        self.cl_norm = Contrast_Loss()

    def forward(self, support, query):
        YS = support['label']
        YQ = query['label']

        YS, YQ = self.reidx_y(YS, YQ)

        XS = self.ebd(support)
        XQ = self.ebd(query)

        sorted_YS, indices = torch.sort(YS)
        XS = XS[indices]

        proto = self.model(XS, flag='XS')
        XS = self.dropout(XS)
        proto1 = self.model(XS, flag='XS')
        norm_loss1 = self.cl_norm(proto, proto1)

        XQ1 = self.model(XQ)
        XQ2 = self.dropout(XQ)
        XQ2 = self.model(XQ2)
        norm_loss2 = self.cl_norm(XQ1, XQ2)

        pred = F.cosine_similarity(proto.unsqueeze(0), XQ1.unsqueeze(1), dim=-1) * 10

        loss = F.cross_entropy(pred, YQ) + norm_loss1 + norm_loss2

        acc = BASE.compute_acc(pred, YQ)

        return acc, loss

