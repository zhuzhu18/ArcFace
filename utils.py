import torch
import io
import matplotlib.pyplot as plt
from torchvision import transforms
from datetime import datetime

def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])    # [parameter object of batchnorm module]
            else:
                paras_wo_bn.extend([*layer.parameters()])    # [parameter object of not batchnorm module]
    return paras_only_bn, paras_wo_bn

hflip = transforms.Compose([
    transforms.Lambda(lambda x: x*0.5+0.5),
    transforms.ToPILImage(),      # 一个具有__call__方法的对象
    transforms.functional.hflip,  # 函数名
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def hflip_batch(batch_tensor):
    hfliped_imgs = torch.empty_like(batch_tensor)
    for i, img_tensor in enumerate(batch_tensor):
        hfliped_imgs[i] = hflip(img_tensor)
    return hfliped_imgs

def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()

    return buf

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

def save_checkpoint(model, optimizer, conf, accuracy, epoch, extra=None, model_only=False):

    save_path = conf.save_path
    os.makedirs(save_path, exist_ok=True)
    torch.save(
        model.model.state_dict(), save_path /
        ('model_{}_accuracy:{}_epoch:{}_{}.pth'.format(get_time(), accuracy, epoch, extra)))
    torch.save(
        optimizer.state_dict(), save_path /
        ('optimizer_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, epoch, extra)))
    if not model_only:
        torch.save(
            model.head.state_dict(), save_path /
            ('head_{}_accuracy:{}_step:{}_{}.pth'.format(get_time(), accuracy, epoch, extra)))

def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')

def schedule_lr(optimizer):
    for param_group in optimizer.param_groups():
        param_group['lr'] /= 10