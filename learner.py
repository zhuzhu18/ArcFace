from tensorboardX import SummaryWriter
from PIL import Image
from torchvision import transforms as trans
from model import l2_norm
from utils import hflip_batch, gen_plot, save_checkpoint, AverageMeter, schedule_lr
from verification import calculate_roc
import torch
from collections import OrderedDict
from progress.bar import IncrementalBar
import json
import numpy as np

class face_learner(object):
    def __init__(self, conf, model, criterion, optimizer):

        # -------------------------------Training initialization----------------------------
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.milestones = conf.milestones      # [12, 15, 18]
        self.writer = SummaryWriter(logdir=conf.log_path)

        self.conf = conf
        self.board_loss_every = 2
        self.train_log_path = conf.log_path / 'train_log'
        self.thresholds = np.arange(0, 4, 0.01)

    def fit(self, train_loader, valid_loader, start_epoch=0, max_epochs=200):

        best_acc = 0.
        bar = IncrementalBar(max=max_epochs - start_epoch)
        for e in range(start_epoch, max_epochs):
            bar.message = '{:>5.2f}%%'.format(bar.percent)
            bar.suffix = '{}/{} [{}<{}\t{:.2f}it/s]'.format(bar.index, bar.max, bar.elapsed_td, bar.eta_td, bar.avg)
            bar.next()
            if e == self.milestones[0]:
                schedule_lr(self.optimizer)       # update learning rate once
            if e == self.milestones[1]:
                schedule_lr(self.optimizer)
            if e == self.milestones[2]:
                schedule_lr(self.optimizer)
            self.train(train_loader, self.model, self.criterion, self.optimizer, e)

            accuracy, best_threshold, roc_curve_tensor = self.evaluate(self.conf, *valid_loader['agedb_30'])
            self.board_val('agedb_30', accuracy, best_threshold, roc_curve_tensor, e)
            if accuracy > best_acc:
                best_acc = accuracy
                save_checkpoint(self.model, self.optimizer, self.conf, best_acc, e)
        bar.finish()
            # accuracy, best_threshold, roc_curve_tensor = self.evaluate(self.conf, *valid_loader['lfw'])
            # self.board_val('lfw', accuracy, best_threshold, roc_curve_tensor, e)
            # accuracy, best_threshold, roc_curve_tensor = self.evaluate(self.conf, *valid_loader['cfp_fp'])
            # self.board_val('cfp_fp', accuracy, best_threshold, roc_curve_tensor, e)

    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor, step):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, step)

    def train(self, train_loader, network, criterion, optimizer, epoch):
        network.model.train()
        network.head.train()
        losses = AverageMeter()
        train_log = open(str(self.train_log_path), 'w')
        for imgs, labels in train_loader:
            imgs = imgs.to(self.conf.device)
            labels = labels.to(self.conf.device)

            optimizer.zero_grad()
            embeddings = network.model(imgs)
            thetas = network.head(embeddings, labels)
            loss = criterion(thetas, labels)
            loss.backward()
            losses.update(loss.item(), labels.size(0))
            optimizer.step()

        record = OrderedDict([('train loss', losses.val), ('epoch', epoch)])
        train_log.write(json.dumps(record) + '\n')
        train_log.close()

    def evaluate(self, conf, val_loader, issame, nrof_folds=5, tta=False):
        self.model.model.eval()
        embeddings = torch.zeros([len(val_loader.dataset), conf.embedding_size])
        with torch.no_grad():
            for idx, data in enumerate(val_loader):       # data: batch_size * 3 * 112 * 112
                batch_size = data.size(0)
                if tta:
                    fliped = hflip_batch(data)
                    emb_batch = self.model.model(data.to(conf.device)) + self.model.model(fliped.to(conf.device))
                    embeddings[idx: idx+batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx: idx+batch_size] = self.model.model(data.to(conf.device))     # embeddings: batch_size * 512

        tpr, fpr, accuracy, best_thresholds = calculate_roc(self.thresholds, embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)

        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor