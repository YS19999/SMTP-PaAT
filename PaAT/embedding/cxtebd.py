import datetime

import torch
import torch.nn as nn
from transformers import BertModel
# from pytorch_transformers import BertModel
import dataset.stats as stats

class CXTEBD(nn.Module):
    """
        An embedding layer directly returns precomputed BERT
        embeddings.
    """
    def __init__(self, args, return_seq=False):
        """
            pretrained_model_name_or_path, cache_dir: check huggingface's codebase for details
            finetune_ebd: finetuning bert representation or not during
            meta-training
            return_seq: return a sequence of bert representations, or [cls]
        """
        super(CXTEBD, self).__init__()

        self.args = args
        self.return_seq = return_seq

        print("{}, Loading pretrainedModel bert".format(datetime.datetime.now().strftime('%y/%m/%d %H:%M:%S')), flush=True)

        self.model = BertModel.from_pretrained("module/bert-base-uncased") # BERT after SMTP 
        self.unfreeze_layers = ['layer.11', 'pooler.']
        for name, param in self.model.named_parameters():
            param.requires_grad = False
            for ele in self.unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break

        self.embedding_dim = self.model.config.hidden_size
        self.ebd_dim = self.model.config.hidden_size

    def get_bert(self, bert_id, mask, data):
        """
            Return the last layer of bert's representation
            @param: bert_id: batch_size * max_text_len+2
            @param: text_len: text_len

            @return: last_layer: batch_size * max_text_len
        """

        # need to use smaller batches
        out = self.model(input_ids=bert_id, attention_mask=mask)

        # return seq of bert ebd, dim: batch, text_len, ebd_dim
        # return last_layer[:,1:-1,:]

        if self.return_seq:
            return out[0]
        else:
            return out[0][:, 0, :]

    def forward(self, data, weight=None):

        text = data['text']
        attn_mask = data['attn_mask']
        with torch.no_grad():
            return self.get_bert(text, attn_mask, data)

