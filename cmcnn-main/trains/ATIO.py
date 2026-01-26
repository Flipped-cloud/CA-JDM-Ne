import sys
import os

# 获取项目根目录（noisyFER-main的路径）
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_path)  # 把根目录加入搜索路径

"""
ATIO -- All Train In One
"""
from trains.FER_DCNN import *
from trains.LM_DCNN import *
from trains.CMCNN import *

__all__ = ['ATIO']

class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            'FER_DCNN': FER_DCNN,
            'LM_DCNN': LM_DCNN,
            'CMCNN': CMCNN
        }
    
    def getTrain(self, args):
        return self.TRAIN_MAP[args.modelName](args)