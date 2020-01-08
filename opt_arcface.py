import argparse
from basic_config import get_basic_config

def get_configs():
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-e", "--epochs", help="training epochs", default=1, type=int)
    parser.add_argument("-net", '--net_mode', choices=['ir', 'ir_se', 'mobilefacenet'], default='ir_se')
    parser.add_argument("-depth", "--net_depth", choices=[50, 100, 152], default=50, type=int)
    parser.add_argument('-lr', '--lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument("-b", "--batch_size", help="batch_size", default=96, type=int)
    parser.add_argument("-w", "--num_workers", help="workers number", default=3, type=int)
    parser.add_argument("-d", '--data_mode', choices=['vgg', 'ms1m', 'emore', 'concat'], default='faces_emore', type=str)

    args = parser.parse_args()
    config = get_basic_config()

    for k, v in args.__dict__.items():
        if not k.startswith('__'):
            config[k] = v

    return config