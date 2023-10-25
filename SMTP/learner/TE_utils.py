import torch
import torch.nn as nn
import torch.nn.functional as F

class TE_Loss(nn.Module):
    def __init__(self):
        super(TE_Loss, self).__init__()
        self.loss_fcun = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, anchor, positive, negative):

        loss = self.loss_fcun(anchor, positive, negative)

        return loss

