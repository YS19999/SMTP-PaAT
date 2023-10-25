import torch
from embedding.cxtebd import CXTEBD
from classifier.protoAttn import ProtoAttn

from dataset.utils import tprint

def get_classifier(vocab, args):
    tprint("Building classifier: {}".format(args.classifier))
    
    ebd = CXTEBD(args, return_seq=True)

    model = ProtoAttn(ebd, args)

    if args.snapshot != '':
        # load pretrained models
        tprint("Loading pretrained classifier from {}".format(
            args.snapshot + '.clf'
            ))
        model.load_state_dict(torch.load(args.snapshot + '.clf'))

    if args.cuda != -1:
        return model.cuda(args.cuda)
    else:
        return model
