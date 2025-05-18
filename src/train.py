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
import utils
from addict import Dict
from collections import OrderedDict

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from torch.cuda.amp import autocast, GradScaler

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '8888'
# os.environ['NCCL_DEBUG'] = 'INFO'


def train(rank, world_size, cfg, start_time):
    # print('pos4')
    # rank = gpu * node_rank + gpu_rank
    # print('gpu_rank:{}, world_size:{}, node_rank:{}, gpu:{}, rank:{}'.format(gpu_rank, world_size, node_rank, gpu, rank))

    torch.cuda.set_device(rank)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    # if rank == 0:
    #     print("num_gpu:{}".format(gpu))
    # print("global rank:{}".format(rank))
    # print("intra rank:{}".format(gpu_rank))

    # current_dir = os.getcwd()
    # with open(current_dir + "/hostfile") as f:
    #     host = f.readlines()
    # host[0] = host[0].rstrip("\n")
    # dist_url = "tcp://" + host[0] + ":" + str(port_num)
    # print(dist_url)
    # # initialize the process group
    # dist.init_process_group("nccl", init_method=dist_url, rank=rank, world_size=world_size)
    # print("tcp connected")



    # make dirs
    if rank == 0:
        save_dir = '../outputs/%s/%s/log' % (cfg.exp_name, start_time)
        model_dir = '../outputs/%s/%s/model' % (cfg.exp_name, start_time)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        model_path = '%s/model_events_pose.pkl' % (model_dir)

        # set logger
        logger = logging.getLogger('TrainLog')
        logger.setLevel(INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(INFO)
        logger.addHandler(stream_handler)
        file_handler = logging.FileHandler('%s/train.py.log' % save_dir)
        file_handler.setLevel(INFO)
        logger.addHandler(file_handler)

        # log config
        logger.info('[config file] %s' % cfg.config_file)
        max_length = max([len(k) for k in cfg.keys()])
        for k, v in cfg.items():
            logger.info(' ' * (max_length - len(k)) + k + ': ' + str(v))

        # set tensorboard
        writer = SummaryWriter('../outputs/%s/%s/log' % (cfg.exp_name, start_time))
        logger.info('[tensorboard] ../outputs/%s/%s/log' % (cfg.exp_name, start_time))
        logger.info('[num GPUs] %d' % world_size)
    else:
        logger = None
        writer = None

    # set seed
    torch_fix_seed()
    # GPU or CPU configuration
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    dtype = torch.float32
    B = cfg.batch_size
    L = cfg.batch_length

    # set dataset
    # calculate time to build dataset
    data_start_time = time.time()
    dataset_train = DavisDataloader(cfg, 'train', start_time)

    train_sampler = DistributedSampler(dataset_train)
    train_generator = DataLoader(
        dataset_train,
        batch_size=B,
        shuffle=False,
        sampler=train_sampler,
        num_workers=cfg.num_worker,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )

    dataset_test = DavisDataloader(cfg, 'test', start_time)

    test_sampler = DistributedSampler(dataset_test)
    test_generator = DataLoader(
        dataset_test,
        batch_size=B,
        shuffle=False,
        sampler=test_sampler,
        num_workers=cfg.num_worker,
        pin_memory=cfg.pin_memory,
        drop_last=True,
    )
    data_end_time = time.time()
    total_iters = len(dataset_train) // B // world_size + 1
    test_total_iters = len(dataset_test) // B // world_size + 1
    if rank == 0:
        logger.info('Complete data loading in {:.2f} mins.'.format((data_end_time - data_start_time) / 60))
        logger.info('dataset_train size: %d, dataset_test size: %d' % (len(dataset_train), len(dataset_test)))

    # set model
    model = EPCPoseModel(device, bidirectional=cfg.model.bidirectional, feat=cfg.model.feat)

    # set optimizer
    def make_optimizer(params, name, **kwargs):
        return optim.__dict__[name](params, **kwargs)
    def set_scheduler(optimizer, name, **kwargs):
        return optim.lr_scheduler.__dict__[name](optimizer, **kwargs)

    if cfg.model.name == 'pose' and cfg.diff_lr:
        params_to_update_1 = []
        modules_to_update_1 = [model.module0, model.module1]
        for module in modules_to_update_1:
            params_to_update_1 += list(module.parameters())
        params_to_update_2 = []
        modules_to_update_2 = [model.module2, model.module3, model.module4]
        for module in modules_to_update_2:
            params_to_update_2 += list(module.parameters())
        optimizer = optim.__dict__[cfg.optimizer.name]([
            {'params': params_to_update_1, 'lr': cfg.optimizer.lr * 0.1},
            {'params': params_to_update_2, 'lr': cfg.optimizer.lr},
        ])
    else:
        optimizer = make_optimizer(model.parameters(), **cfg.optimizer)
    if 'scheduler' in cfg and cfg.scheduler:
        scheduler = set_scheduler(optimizer, **cfg.scheduler)

    model = model.to(rank)  # move the model parameters to CPU/GPU
    # model = DistributedDataParallel(model, device_ids=[gpu_rank])

    best_loss = 1e4
    last_epoch = 0

    if cfg.checkpoint:
        model_path = '../outputs/%s/%s/model/model_events_pose.pkl' % (cfg.exp_name, cfg.checkpoint)
        checkpoint = torch.load(model_path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            new_state_dict[k.replace('module.', '')] = v
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        last_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        if rank == 0:
            logger.info('[model dir] model loaded from %s' % model_path)
            logger.info("Now continue training.")

    model = DistributedDataParallel(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    scaler = GradScaler(enabled=cfg.use_amp)

    for epoch in range(cfg.epochs):
        if cfg.checkpoint:
            epoch = epoch + last_epoch
        if rank == 0:
            logger.info('====================================== Epoch %i ========================================' % (epoch + 1))
            # '''
            logger.info('-------------------------------------- Training ----------------------------------------')
        model.train()
        results = collections.defaultdict(list)
        start_time = time.time()
        for iter, data in enumerate(train_generator):
            for k in data.keys():
                if k not in  ['info', 'kp_dist']:
                    data[k] = data[k].to(device=device)

            optimizer.zero_grad()
            with autocast(enabled=cfg.use_amp):
                kpt_pred = data['kpt_pred'] if cfg.kpt_pred else None
                normalized_pc = normalize_events(data['pc'], cfg, device)
                output = model(normalized_pc, kpt_pred)
                loss_dict = compute_losses(data, output, cfg)
                error_dict = compute_error(data, output, cfg, device)
                loss = sum(loss_dict.values())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()


            results = append_results(results, loss, loss_dict, error_dict)
            if iter % (total_iters // cfg.display) == 0 and rank == 0:
                results['info'] = (data['info'][0][0], data['info'][1][0].item())
                results = calc_results(results)

                progress = 100 * iter // total_iters
                step = 100 * epoch + progress

                write_tensorboard(writer, results, step, 'train', scheduler.get_last_lr()[0] if cfg.scheduler else None)
                dataset_train.visualize_batch(results['info'], data, output, epoch, iter, cfg)

                end_time = time.time()
                time_used = (end_time - start_time) / 60.
                logger.info('>>> [epoch {:4d} / iter {:4d}] {:3d}%\n'
                    '     loss_q {:.4f}, loss_t {:.4f}, loss_v {:.4f}, loss_s3 {:.4f}, loss_s2 {:.4f}, loss_d {:.4f}, loss_l {:.4f}, loss_b {:.4f} / loss {:.4f}\n'
                    '     error_a {:.4f}, error_t {:.4f}, error_v {:.4f}, error_s3 {:.4f}, error_s2 {:.4f}, error_l {:.4f}, acc_b {:.4f} \n'
                    '     lr: {:.6f}, time used: {:.2f} mins.'
                    .format(epoch + 1, iter * world_size, progress,
                            results['loss_q'], results['loss_t'], results['loss_v'], results['loss_s3'], results['loss_s2'], results['loss_d'], results['loss_l'], results['loss_b'], results['loss'],
                            results['error_a'], results['error_t'], results['error_v'], results['error_s3'], results['error_s2'], results['error_l'], results['acc_b'],
                            scheduler.get_last_lr()[0] if cfg.scheduler else cfg.optimizer.lr, time_used))

        if cfg.exec_test:
            if rank == 0:
                logger.info('-------------------------------------- test ----------------------------------------')
            start_time = time.time()
            model.eval()  # dropout layers will not work in eval mode
            results = collections.defaultdict(list)
            start_time = time.time()
            with torch.set_grad_enabled(False):  # deactivate autograd to reduce memory usage
                for iter, data in enumerate(test_generator):
                    for k in data.keys():
                        if k not in  ['info', 'kp_dist']:
                            data[k] = data[k].to(device=device, dtype=dtype)

                    kpt_pred = data['kpt_pred'] if cfg.kpt_pred else None
                    normalized_pc = normalize_events(data['pc'], cfg, device)
                    output = model(normalized_pc, kpt_pred)
                    loss_dict = compute_losses(data, output, cfg)
                    error_dict = compute_error(data, output, cfg, device)
                    loss = sum(loss_dict.values())
                    results = append_results(results, loss, loss_dict, error_dict)

                    if iter % (test_total_iters // cfg.display) == 0 and iter != 0:
                        info = (data['info'][0][0], data['info'][1][0].item())
                        dataset_test.visualize_batch(info, data, output, epoch, iter, cfg)

            # Gather results from all processes
            gathered_results = [{} for _ in range(world_size)]
            dist.all_gather_object(gathered_results, results)

            # Only process 0 will compute the final results and write them to tensorboard
            if rank == 0:
                results = {}
                for key in gathered_results[0].keys():
                    results[key] = np.mean([res[key] for res in gathered_results])

                results = calc_results(results)

                write_tensorboard(writer, results, epoch + 1, 'test')

                end_time = time.time()
                time_used = (end_time - start_time) / 60.
                logger.info('>>> [epoch {:4d}]\n'
                    '     loss_q {:.4f}, loss_t {:.4f}, loss_v {:.4f}, loss_s3 {:.4f}, loss_s2 {:.4f}, loss_d {:.4f}, loss_l {:.4f}, loss_b {:.4f} / loss {:.4f}\n'
                    '     error_a {:.4f}, error_t {:.4f}, error_v {:.4f}, error_s3 {:.4f}, error_s2 {:.4f}, error_l {:.4f}, acc_b {:.4f} \n'
                    '     lr: {:.6f}, time used: {:.2f} mins.'
                    .format(epoch + 1,
                            results['loss_q'], results['loss_t'], results['loss_v'], results['loss_s3'], results['loss_s2'], results['loss_d'], results['loss_l'], results['loss_b'], results['loss'],
                            results['error_a'], results['error_t'], results['error_v'], results['error_s3'], results['error_s2'], results['error_l'], results['acc_b'],
                            scheduler.get_last_lr()[0] if cfg.scheduler else cfg.optimizer.lr, time_used))


        if rank == 0:
            torch.save({
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': results['loss'],
            }, '%s/model.pkl' % model_dir)

            if best_loss > results['loss']:
                torch.save({
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': results['loss'],
                }, '%s/model_events_pose.pkl' % model_dir)
                best_loss = results['loss']
                writer.add_scalar('test/best_loss', best_loss, epoch + 1)
                logger.info('>>> Model saved as {}... best loss {:.4f}'.format(model_path, best_loss))
            if cfg.scheduler:
                scheduler.step()
    writer.close()

def normalize_events(events, cfg, device):
    events = np.array(events.cpu())
    mean = cfg.normalize.mean
    std = cfg.normalize.std
    events[:, :, :, 0] = (events[:, :, :, 0] - mean[0]) / std[0]
    events[:, :, :, 1] = (events[:, :, :, 1] - mean[1]) / std[1]
    events[:, :, :, 2] = (events[:, :, :, 2] - mean[2]) / std[2]
    events = torch.from_numpy(events).to(device=device, dtype=torch.float32)
    return events

def compute_losses(data, output, cfg):
    criterion_l1 = torch.nn.L1Loss(reduction='mean')
    criterion_mse = torch.nn.MSELoss()
    criterion_keypoint = utils.KLDiscretLoss()

    if cfg.loss_func ==  'L1':
        criterion = criterion_l1
    elif cfg.loss_func == 'L2':
        criterion = criterion_mse
    else:
        print('Error! Unknown loss function.')
        sys.exit(1)

    loss_q = criterion(output['q'], data['q'])
    loss_t = criterion(output['t'], data['t'])
    loss_v = criterion(output['v'], data['v'])
    loss_l = criterion(output['l'], data['l']) if output['l'] != None else torch.tensor(0)
    loss_b = criterion(output['b'], data['b'])
    loss_s3 = criterion(output['s3'], data['s3'])
    loss_s2 = torch.tensor(0)
    loss_d = torch.tensor(0)

    B, L, _, _, _ = data['q'].shape
    if cfg.model.name == 'pose':
        _, _, S, H = output['pred_x'].shape
        # dist_min_gt, dist_max_gt = data['kp_dist'][0][0].item(), data['kp_dist'][1][0].item()
        pred_keypoint = utils.decode_batch_sa_simdr(output['pred_x'].view(B * L, S, H), output['pred_y'].view(B * L, S, H))
        pred_keypoint = pred_keypoint.reshape(B, L, S, 2)
        pelvis = np.repeat(pred_keypoint[:, :, 0, :].reshape(B, L, 1, 2), S - 1, axis=2)
        dist = np.mean(np.linalg.norm(pred_keypoint[:, :, 1:] - pelvis, axis=3), axis=2)
        dist_min_pred, dist_max_pred = np.min(dist), np.max(dist)
        # if dist_min_pred < dist_min_gt:
        #     loss_d = dist_min_gt - dist_min_pred
        # elif dist_max_pred > dist_max_gt:
        #     loss_d = dist_max_pred - dist_max_gt
        # else:
        #     loss_d = torch.tensor(0)
        loss_d = torch.tensor(0)
        loss_s2 = criterion_keypoint(output['pred_x'].view(B * L, S, H), output['pred_y'].view(B * L, S, H), data['gt_x'].view(B * L, S, H), data['gt_y'].view(B * L, S, H))
        loss_l = torch.tensor(0)

    loss_dict = {
        'q':  loss_q  * cfg.loss_weight.q,
        't':  loss_t  * cfg.loss_weight.t,
        'v':  loss_v  * cfg.loss_weight.v,
        'l':  loss_l  * cfg.loss_weight.l,
        'b':  loss_b  * cfg.loss_weight.b,
        's3': loss_s3 * cfg.loss_weight.s3,
        's2': loss_s2 * cfg.loss_weight.s2,
        'd':  loss_d  * cfg.loss_weight.d,
    }

    return loss_dict

def compute_error(data, output, cfg, device):
    cos = torch.nn.CosineSimilarity(-1)
    root_kp = np.asarray(cfg.root_kp, dtype=np.int64)
    leaf_kp = np.asarray(cfg.leaf_kp, dtype=np.int64)
    root_kp = torch.tensor(root_kp, dtype=torch.long, device=device)
    leaf_kp = torch.tensor(leaf_kp, dtype=torch.long, device=device)
    B, L, _ = data['t'].shape
    if cfg.model.name == 'pose':
        gt_keypoint = data['s2']
        _, _, S, H = output['pred_x'].shape
        pred_keypoint = utils.decode_batch_sa_simdr(output['pred_x'].view(B * L, S, H), output['pred_y'].view(B * L, S, H))
        pred_keypoint = pred_keypoint.reshape(B, L, S, 2)
    error_dict = {
        'a': angle_loss(output['s3'], data['s3'], root_kp, leaf_kp).item() / B / L / 8,
        't': torch.sum(torch.sqrt(torch.sum(torch.square(output['t'] - data['t']), dim=-1))).item() / B / L,
        'v': torch.sum(torch.sqrt(torch.sum(torch.square(output['v'] - data['v']), dim=-1))).item() / B / L / 6890,
        's3': torch.sum(torch.sqrt(torch.sum(torch.square(output['s3'] - data['s3']), dim=-1))).item() / B / L / 24,
        's2': utils.cal_2D_mpjpe(gt_keypoint, pred_keypoint) if cfg.model.name == 'pose' else 0,
        'l': torch.sum(torch.sqrt(torch.sum(torch.square(output['l'] - data['l']), dim=-1))).item() / B / L if not cfg.model.name == 'pose' else 0,
        'b': torch.sum(cos(output['b'], data['b'])).item()
    }
    return error_dict


def angle_loss(pred_ske, true_ske, root_kp, leaf_kp):
    pred_vec = pred_ske[:, :, leaf_kp, :] - pred_ske[:, :, root_kp, :]
    true_vec = true_ske[:, :, leaf_kp, :] - true_ske[:, :, root_kp, :]
    cos_sim = torch.nn.functional.cosine_similarity(pred_vec, true_vec, dim=-1)
    angle = torch.sum(torch.abs(torch.acos(torch.clamp(cos_sim, min=-1.0, max=1.0)) / 3.14159265358 * 180.0))
    return angle


def append_results(results, loss, loss_dict, error_dict):
    results['scalar/loss'].append(loss.item())
    results['scalar/loss_q'].append(loss_dict['q'].item())
    results['scalar/loss_t'].append(loss_dict['t'].item())
    results['scalar/loss_v'].append(loss_dict['v'].item())
    results['scalar/loss_s3'].append(loss_dict['s3'].item())
    results['scalar/loss_s2'].append(loss_dict['s2'].item())
    results['scalar/loss_d'].append(loss_dict['d'].item())
    results['scalar/loss_l'].append(loss_dict['l'].item())
    results['scalar/loss_b'].append(loss_dict['b'].item())
    results['scalar/error_a'].append(error_dict['a'])
    results['scalar/error_t'].append(error_dict['t'])
    results['scalar/error_v'].append(error_dict['v'])
    results['scalar/error_s3'].append(error_dict['s3'])
    results['scalar/error_s2'].append(error_dict['s2'])
    results['scalar/error_l'].append(error_dict['l'])
    results['scalar/acc_b'].append(error_dict['b'])
    return results

def calc_results(results):
    results['loss'] = np.mean(results['scalar/loss'])
    results['loss_q'] = np.mean(results['scalar/loss_q'])
    results['loss_t'] = np.mean(results['scalar/loss_t'])
    results['loss_v'] = np.mean(results['scalar/loss_v'])
    results['loss_s3'] = np.mean(results['scalar/loss_s3'])
    results['loss_s2'] = np.mean(results['scalar/loss_s2'])
    results['loss_d'] = np.mean(results['scalar/loss_d'])
    results['loss_l'] = np.mean(results['scalar/loss_l'])
    results['loss_b'] = np.mean(results['scalar/loss_b'])
    results['error_a'] = np.mean(results['scalar/error_a'])
    results['error_t'] = np.mean(results['scalar/error_t'])
    results['error_v'] = np.mean(results['scalar/error_v'])
    results['error_s3'] = np.mean(results['scalar/error_s3'])
    results['error_s2'] = np.mean(results['scalar/error_s2'])
    results['error_l'] = np.mean(results['scalar/error_l'])
    results['acc_b'] = np.mean(results['scalar/acc_b'])
    return results

def write_tensorboard(writer, results, step, mode, lr=None):
    if writer is not None:
        writer.add_scalar('%s/loss_q' % (mode), results['loss_q'], step)
        writer.add_scalar('%s/loss_t' % (mode), results['loss_t'], step)
        writer.add_scalar('%s/loss_v' % (mode), results['loss_v'], step)
        writer.add_scalar('%s/loss_s3' % (mode), results['loss_s3'], step)
        writer.add_scalar('%s/loss_s2' % (mode), results['loss_s2'], step)
        writer.add_scalar('%s/loss_d' % (mode), results['loss_d'], step)
        writer.add_scalar('%s/loss_l' % (mode), results['loss_l'], step)
        writer.add_scalar('%s/loss_b' % (mode), results['loss_b'], step)
        writer.add_scalar('%s/loss' % (mode), results['loss'], step)
        writer.add_scalar('%s/error_a' % (mode), results['error_a'], step)
        writer.add_scalar('%s/error_t' % (mode), results['error_t'], step)
        writer.add_scalar('%s/error_v' % (mode), results['error_v'], step)
        writer.add_scalar('%s/error_s3' % (mode), results['error_s3'], step)
        writer.add_scalar('%s/error_s2' % (mode), results['error_s2'], step)
        writer.add_scalar('%s/error_l' % (mode), results['error_l'] , step)
        writer.add_scalar('%s/acc_b' % (mode), results['acc_b'] , step)
        if lr: writer.add_scalar('LerningRate', lr, step)

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
    start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    # os.environ['OMP_NUM_THREADS'] = '1'
    # torch.set_num_threads(1)
    args = get_args()
    cfg = Dict(yaml.safe_load(open(args.config_path)))
    cfg.config_file = args.config_path
    cfg.exp_name = args.config_path.split('/')[-1][:-5]

    # local_rank = int(os.environ['LOCAL_RANK'])
    # print('pos1')
    # dist.init_process_group(backend='nccl')
    # print('pos2')
    # world_size = dist.get_world_size()
    # node_rank = dist.get_rank() // world_size
    # print('pos3')
    # train(local_rank, world_size, node_rank, local_rank, cfg, start_time)

    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size, cfg, start_time), nprocs=world_size, join=True)

    # node_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])  # Process number in MPI
    # size = int(os.environ["OMPI_COMM_WORLD_SIZE"])  # The all size of process
    # print("node rank:{}".format(node_rank))
    # print("size of process:{}".format(size))
    # gpu = torch.cuda.device_count()  # gpu num per node
    # world_size = gpu * size  # total gpu num
    # print('world size:', world_size)
    # port_num = 50000
    # torch.multiprocessing.spawn(train, args=(world_size, node_rank, gpu, port_num, cfg, start_time), nprocs=gpu, join=True)

if __name__ == '__main__':
    main()
