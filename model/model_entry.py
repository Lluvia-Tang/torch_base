
from model.best.fcn import DeepLabv3Fcn
from model.better.fcn import Resnet101Fcn
from model.sota.fcn import LightFcn
from model.BiLSTM.BiLSTM import BiLSTM
from model.kimCNN.KimCNN import KimCNN

import torch.nn as nn


def select_model(args):
    type2model = {
        'resnet50_fcn': CustomFcn(args),
        'resnet101_fcn': Resnet101Fcn(args),
        'deeplabv3_fcn': DeepLabv3Fcn(args),
        'mobilnetv3_fcn': LightFcn(args),
        'BiLSTM': BiLSTM,
        'Kim_CNN': KimCNN
    }
    model = type2model[args.model_type]
    return model

'''
实现单机多卡
'''
def equip_multi_gpu(model, args):
    model = nn.DataParallel(model, device_ids=args.gpus)
    return model
