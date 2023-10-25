import argparse

from transformers import BertTokenizer

from dataset.mtp_loader import loader_dataset
from smtp_trainer import SMTP

def parse_args():

    parser = argparse.ArgumentParser(description="SMTP")

    parser.add_argument("--bert", type=str, default='../module/bert-base-uncased')

    parser.add_argument("--dataname", type=list, default=['BANKING77', 'HWU64', 'Clinic150', 'Liu'])
    parser.add_argument("--save_path", type=str, default='module')

    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)

    return parser.parse_args()

def smtp_main():
    args = parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.bert)

    dataloader = loader_dataset(args, tokenizer)
    trainer = SMTP(args)

    for ep in range(args.epochs):
        trainer.model.train()
        total_loss_train = trainer.training(dataloader, tokenizer)

        print("train loss: {:.7f}".format(total_loss_train / (len(dataloader) * args.batch_size)))

    trainer.model.save_model(args.save_path)

if __name__ == '__main__':
    smtp_main()
