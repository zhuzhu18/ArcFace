import cv2
import pickle
import bcolz
import numpy as np
import mxnet as mx
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from utils import hflip_batch
import torch

def get_train_dataset(imgs_folder):
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = ImageFolder(imgs_folder, transform=train_transforms)
    num_cls = dataset[-1][1] + 1

    return dataset, num_cls

def get_train_loader(config):
    dataset, class_num = get_train_dataset(str(config.data_path))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, pin_memory=config.pin_memory,
                        num_workers=config.num_workers)
    return loader, class_num

def get_val_loader(config, valid_name):
    dataset, is_same = get_val_pair(config.valid_path, valid_name)
    val_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)

    return val_loader, is_same

def get_val_pair(path, name):
    carray = bcolz.carray(rootdir=str(path/name), mode='r')     # 每个carray对象是一个-1到1的浮点型ndarray, shape为3*112*112, BGR
    issame = np.load(path/'{}_list.npy'.format(name))      # np.bool
    return carray, issame

def load_bin(path, rootdir: Path, transform, image_size=[112, 112]):
    if not rootdir.exists():
        rootdir.mkdir()
    bins, issame_list = pickle.load(open(path, 'rb'), encoding='bytes')
    data = bcolz.fill(shape=[len(bins), 3, image_size[0], image_size[1]], dtype=np.float32, rootdir=rootdir, mode='w')

    for i in range(len(bins)):
        img = mx.image.imdecode(bins[i]).asnumpy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img.astype(np.uint8))
        data[i, ...] = transform(img)
    np.save(str(rootdir)+'_list', np.array(issame_list))

def load_mx_rec(rec_path):
    save_path = rec_path / 'imgs'
    if not save_path.exists():
        save_path.mkdir()
    imgrec = mx.recordio.MXIndexedRecordIO(str(rec_path / 'train.idx'), str(rec_path / 'train.rec'), 'r')
    img_info = imgrec.read_idx(0)
    header, _ = mx.recordio.unpack(img_info)
    max_idx = int(header.label[0])

    for idx in tqdm(range(1, max_idx)):
        img_info = imgrec.read_idx(idx)
        header, img = mx.recordio.unpack_img(img_info)
        label = int(header.label)
        label_path = save_path / str(label)
        if not label_path.exists():
            label_path.mkdir()
        cv2.imwrite(str(label_path / '{}.jpg'.format(idx)), img)