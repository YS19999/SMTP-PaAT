import torch
from tqdm import tqdm

from model.transformer import smtp_bert

class SMTP:
    def __init__(self, args):
        self.args = args

        self.model = smtp_bert(self.args)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.model.to(self.device)

    def training(self, dataloader, tokenizer):

        total_loss_train = 0

        for data in tqdm(dataloader):

            self.model.zero_grad()
            data1, data2, data3 = data

            data1 = self.get_device(data1)
            data2 = self.get_device(data2)
            data3 = self.get_device(data3)

            te_loss = self.model(data1, data2, data3)
            data1['input_ids'], labels = mask_tokens(inputs=data1['input_ids'].cpu(), tokenizer=tokenizer, mlm_probability=0.25)
            data1['input_ids'] = data1['input_ids'].to(self.device)
            mlm_loss = self.model.forwardMLM(data1, labels.to(self.device))

            loss = mlm_loss + te_loss
            total_loss_train += loss.item()

            torch.autograd.set_detect_anomaly = True
            loss.backward()
            self.optimizer.step()

        return total_loss_train

    def get_device(self, data):
        new_data = {'input_ids': data['input_ids'].squeeze(1).to(self.device),
                    'attention_mask': data['attention_mask'].squeeze(1).to(self.device),
                    'token_type_ids': data['token_type_ids'].squeeze(1).to(self.device)}
        return new_data

def mask_tokens(inputs, tokenizer, special_tokens_mask=None, mlm_probability=0.15):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    probability_matrix[torch.where(inputs == 0)] = 0.0
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

