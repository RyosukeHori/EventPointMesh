from distutils.log import INFO
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import sys
import time
import random
import torch.nn as nn
import argparse
import collections
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import yaml
from dataloader import DavisDataloader
from network import EPCPoseModel
import logging
from utils import *
from addict import Dict


def test(cfg):
    # make dirs
    start_time = cfg.checkpoint
    save_dir = '../outputs/%s/test/%s/log' % (cfg.exp_name, start_time)
    model_dir = '../outputs/%s/%s/model' % (cfg.exp_name, start_time)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    model_path = '%s/model_events_pose.pkl' % (model_dir)

    # set logger
    logger = logging.getLogger('TestLog')
    logger.setLevel(INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(INFO)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler('%s/test.py.log' % save_dir)
    file_handler.setLevel(INFO)
    logger.addHandler(file_handler)

    # log config
    logger.info('[config file] %s' % cfg.config_file)
    max_length = max([len(k) for k in cfg.keys()])
    for k, v in cfg.items():
        logger.info(' ' * (max_length - len(k)) + k + ': ' + str(v))

    # set tensorboard
    writer = SummaryWriter(save_dir)
    logger.info('[tensorboard] %s' % (save_dir))

    # set seed
    torch_fix_seed()
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:%s" % cfg.gpu_id if use_cuda else "cpu")
    dtype = torch.float32
    B = cfg.batch_size
    L = cfg.batch_length

    # set dataset
    dataset_test = DavisDataloader(cfg, 'test', start_time)

    test_generator = DataLoader(
        dataset_test,
        batch_size=B,
        shuffle=False,
        num_workers=cfg.num_worker,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    test_total_iters = len(dataset_test) // B + 1

    # set model
    model = EPCPoseModel(device, bidirectional=cfg.model.bidirectional)

    model = model.to(device=device)  # move the model parameters to CPU/GPU
    logger.info('[model dir] model loaded from %s' % model_path)
    checkpoint = torch.load(model_path, map_location=device)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        new_state_dict[k.replace('module.', '')] = v
    model.load_state_dict(new_state_dict)


    logger.info('-------------------------------------- test ----------------------------------------')
    model.eval()  # dropout layers will not work in eval mode
    results = collections.defaultdict(list)
    start_time = time.time()
    with torch.set_grad_enabled(False):  # deactivate autograd to reduce memory usage
        for iter, data in enumerate(test_generator):
            for k in data.keys():
                if k != 'info' and k != 'kp_dist':
                    data[k] = data[k].to(device=device, dtype=dtype)
            B, T = data['pc'].size()[0], data['pc'].size()[1]
            kpt_pred = data['kpt_pred'] if cfg.kpt_pred else None
            normalized_pc = normalize_events(data['pc'], cfg, device)
            output = model(normalized_pc, kpt_pred)

            mpjpe = compute_mpjpe(output['s3'].detach(), data['s3'])  # [B, T, 24]
            pa_mpjpe = compute_pa_mpjpe(output['s3'].detach(), data['s3'])  # [B, T, 24]
            pel_mpjpe = compute_pelvis_mpjpe(output['s3'].detach(), data['s3'])  # [B, T, 24]
            pck_head = compute_pck_head(output['s3'], data['s3'])  # [B, T, 24]
            _, s, R, t = batch_compute_similarity_transform_torch(
                output['s3'].view(-1, 24, 3), data['s3'].view(-1, 24, 3), True)
            # print(s.size(), R.size(), t.size())
            s = s.unsqueeze(-1).unsqueeze(-1)
            pa_verts = s * R.bmm(output['v'].reshape(B * T, 6890, 3).permute(0, 2, 1)) + t
            pa_verts = pa_verts.permute(0, 2, 1).view(B, T, 6890, 3)
            target_verts = data['v']
            pve = torch.mean(torch.sqrt(torch.sum((target_verts - pa_verts) ** 2, dim=-1)), dim=-1)  # [B, T]
            results['scalar/mpjpe'].append(torch.mean(mpjpe.detach()))
            results['scalar/pa_mpjpe'].append(torch.mean(pa_mpjpe.detach()))
            results['scalar/pel_mpjpe'].append(torch.mean(pel_mpjpe.detach()))
            results['scalar/pck_head'].append(torch.mean(pck_head.detach().float()))
            results['scalar/pve'].append(torch.mean(pve.detach()))

            action = data['info'][0][0]
            logger.info('action: {}, mpjpe {:.4f}, pa-mpjpe {:.4f}, pel-mpjpe {:.4f}, pck_head {:.4f}, pve {:.4f}'
            .format(action, results['scalar/mpjpe'][-1], results['scalar/pa_mpjpe'][-1], results['scalar/pel_mpjpe'][-1], results['scalar/pck_head'][-1], results['scalar/pve'][-1]))

            # dataset_test.export_test_results(data, output)
            
            # dataset_test.visualize_test_batch(data, output, cfg)
            # visualize_joints(output['s3'][0, 0].cpu().numpy(), data['s3'][0, 0].cpu().numpy())


        results['mpjpe'] = torch.mean(torch.stack(results['scalar/mpjpe'], dim=0))
        results['pa_mpjpe'] = torch.mean(torch.stack(results['scalar/pa_mpjpe'], dim=0))
        results['pel_mpjpe'] = torch.mean(torch.stack(results['scalar/pel_mpjpe'], dim=0))
        results['pck_head'] = torch.mean(torch.stack(results['scalar/pck_head'], dim=0))
        results['pve'] = torch.mean(torch.stack(results['scalar/pve'], dim=0))

        logger.info('Total: mpjpe {:.4f}, pa-mpjpe {:.4f}, pel-mpjpe {:.4f}, pck_head {:.4f}, pve {:.4f}'
            .format(results['mpjpe'], results['pa_mpjpe'], results['pel_mpjpe'], results['pck_head'], results['pve']))

    writer.close()


def visualize_joints(out_joints, data_joints):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(out_joints[:, 0], out_joints[:, 1], out_joints[:, 2], c='r', marker='o')
    ax.scatter(data_joints[:, 0], data_joints[:, 1], data_joints[:, 2], c='b', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # aligne range of x, y, z
    max_range = np.array([out_joints[:, 0].max() - out_joints[:, 0].min(),
                            out_joints[:, 1].max() - out_joints[:, 1].min(),
                            out_joints[:, 2].max() - out_joints[:, 2].min()]).max() / 2.0
    mid_x = (out_joints[:, 0].max() + out_joints[:, 0].min()) * 0.5
    mid_y = (out_joints[:, 1].max() + out_joints[:, 1].min()) * 0.5
    mid_z = (out_joints[:, 2].max() + out_joints[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()
    plt.close()

def normalize_events(events, cfg, device):
    events = np.array(events.cpu())
    mean = cfg.normalize.mean
    std = cfg.normalize.std
    events[:, :, :, 0] = (events[:, :, :, 0] - mean[0]) / std[0]
    events[:, :, :, 1] = (events[:, :, :, 1] - mean[1]) / std[1]
    events[:, :, :, 2] = (events[:, :, :, 2] - mean[2]) / std[2]
    events = torch.from_numpy(events).to(device=device, dtype=torch.float32)
    return events

def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='config file(.yaml)', default='config/DAVIS_all_15f_imgFPS_GL.yaml')
    args = parser.parse_args()
    return args

def main():
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.set_num_threads(1)
    args = get_args()
    cfg = Dict(yaml.safe_load(open(args.config_path)))
    cfg.config_file = args.config_path
    cfg.exp_name = args.config_path.split('/')[-1][:-5]
    test(cfg)

if __name__ == '__main__':
    main()
