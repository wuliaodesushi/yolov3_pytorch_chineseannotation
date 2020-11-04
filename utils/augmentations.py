import torch
import torch.nn.functional as F
import numpy as np
import pdb


def horisontal_flip(images, targets):
#     pdb.set_trace()
    images = torch.flip(images, [-1])    
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets
