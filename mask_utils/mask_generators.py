import torch
from mask_utils.utils import *


# from ner_model import NerModel
from torch import nn
mask_samp = 100
mask_rate = 0
mask_num = 1
model_paths = {
    "ner": "./model/ner.pth"
}

class Ner(nn.Module):
    def __init__(self):
        self.model = torch.load(model_paths["ner"])
        self.model.eval()

    def forword(self, tokens):

        mask_indices = []
        return mask_indices
