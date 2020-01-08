from learner import face_learner
from opt_arcface import get_configs
from data.data_pipe import get_train_loader, get_val_loader
import torch.optim as optim
import torch.nn as nn
from model import Backbone, ArcFace
from utils import separate_bn_paras
from collections import namedtuple, OrderedDict

def main():
    conf = get_configs()

    # -------------------------------Training initialization----------------------------
    train_loader, train_class_num = get_train_loader(conf)

    # -------------------------------Model config----------------------------
    network = namedtuple('network', ['head', 'model'])
    network.model = Backbone(conf.net_depth, conf.drop_ratio, conf.net_mode)
    network.head = ArcFace(embedding_size=conf.embedding_size, class_num=train_class_num)

    paras_only_bn, paras_wo_bn = separate_bn_paras(network.model)
    optimizer = optim.SGD([
        {'params': paras_wo_bn + [network.head.kernel], 'weight_decay': 5e-4},
        {'params': paras_only_bn}
    ], lr=conf.lr, momentum=conf.momentum)
    criterion = conf.ce_loss

    # -------------------------------valid initialization----------------------------
    valid_loader = OrderedDict([('agedb_30', get_val_loader(conf, 'agedb_30')),
                               ('cfp_fp', get_val_loader(conf, 'cfp_fp')),
                               ('lfw', get_val_loader(conf, 'lfw'))])

    network.model = nn.DataParallel(network.model.cuda(), device_ids=[0, 1])
    network.head = nn.DataParallel(network.head.cuda(), device_ids=[0, 1])
    learner = face_learner(conf, network, criterion, optimizer)
    learner.fit(train_loader, valid_loader, start_epoch=0, max_epochs=200)

if __name__ == '__main__':
    main()