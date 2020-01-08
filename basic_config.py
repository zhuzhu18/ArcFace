import torch
import torch.nn as nn
from easydict import EasyDict as edict
from pathlib import Path
import torchvision.transforms as trans

def get_basic_config():
    conf = edict()
    conf.data_path = Path('/home/zzh/dataset/faces_emore/imgs')
    conf.valid_path = Path('/home/zzh/faces_emore')
    conf.work_space = Path('work')
    conf.model_path = conf.work_space / 'models'

    conf.input_size = [112, 112]
    conf.embedding_size = 512

    conf.net_mode = 'ir_se'    # 'ir' or 'ir_se'
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # conf.device = torch.device('cpu')
    conf.test_transform = trans.Compose([
        trans.ToTensor(),
        trans.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
        ])
    conf.data_mode = 'faces_emore'
    conf.batch_size = 100  # irse net depth 50

    # --------------------Training Config ------------------------
    conf.log_path = conf.work_space / 'log'
    conf.save_path = conf.work_space / 'save'

    conf.lr = 1e-3
    conf.milestones = [12, 15, 18]
    conf.momentum = 0.9
    conf.pin_memory = True
    conf.num_workers = 3
    conf.ce_loss = nn.CrossEntropyLoss()

    # --------------------Inference Config ------------------------
    conf.facebank_path = conf.data_path / 'facebank'
    conf.threshold = 1.5      # margin m
    conf.face_limit = 10
    conf.min_face_size = 30

    return conf

if __name__ == '__main__':
    args = get_config()
    print(args.epochs)