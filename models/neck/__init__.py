'''
Author: Niki
Date: 2021-01-12 15:24:56
Description: 
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn.functional as F

from models.neck.neck import AdjustLayer, AdjustAllLayer

NECKS = {
         'AdjustLayer': AdjustLayer,
         'AdjustAllLayer': AdjustAllLayer
        }

def get_neck(name, **kwargs):
    return NECKS[name](**kwargs)
