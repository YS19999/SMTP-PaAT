import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertForMaskedLM, BertTokenizer
from learner.TE_utils import TE_Loss

class smtp_bert(nn.Module):

    def __init__(self, args):
        super(smtp_bert, self).__init__()

        self.bert = BertForMaskedLM.from_pretrained(args.bert)

        self.te_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.bert.config.hidden_size, 256)
        )

        self.te_loss = TE_Loss()

    def get_te_feat(self, feat):
        te_feat = self.te_head(feat)
        te_feat = F.normalize(te_feat, dim=1)
        return te_feat

    def forward(self, embed1, embed2, embed3):

        bert_output1 = self.bert(**embed1, output_hidden_states=True)
        bert_output2 = self.bert(**embed1, output_hidden_states=True)
        bert_output3 = self.bert(**embed1, output_hidden_states=True)

        feat1 = self.get_mean_embeddings(bert_output1.hidden_states[-1], embed1['attention_mask'])
        feat2 = self.get_mean_embeddings(bert_output2.hidden_states[-1], embed2['attention_mask'])
        feat3 = self.get_mean_embeddings(bert_output3.hidden_states[-1], embed3['attention_mask'])

        # TE
        te_feat1 = self.get_te_feat(feat1)
        te_feat2 = self.get_te_feat(feat2)
        te_feat3 = self.get_te_feat(feat3)

        te_loss = self.te_loss(te_feat1, te_feat2, te_feat3)

        return te_loss

    def forwardMLM(self, embed, labels):

        outputs = self.bert(**embed, labels=labels)

        return outputs.loss

    def get_mean_embeddings(self, bert_output, attention_mask):

        mean_output = torch.sum(bert_output[0] * attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask.unsqueeze(-1), dim=1)

        return mean_output

    def save_model(self, save_path):

        self.bert.save_pretrained(save_path)




