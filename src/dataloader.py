import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import joblib
import torch
import matplotlib.pyplot as plt
import utils as utils
from smpl.SMPL import SMPL, batch_rodrigues, mat2axisangle
import pickle
import glob

class DavisDataloader(Dataset):
    def __init__(self, cfg, mode, start_time):
        self.batch_length = cfg.batch_length
        self.stride = cfg.stride
        self.exp_name = cfg.exp_name
        self.root_dir = cfg.root_dir
        self.event_data = cfg.event_data
        self.dataset = cfg.dataset
        self.dataset_div = cfg.dataset_div
        self.mode = mode
        self.start_time = start_time
        self.img_size = cfg.img_size
        self.kpt_pred = cfg.kpt_pred
        self.large_memory = cfg.large_memory

        self.data_dir = '%s/%s' % (self.root_dir, cfg.data_dir)
        self.smpl_dir = '%s/%s' % (self.root_dir, cfg.smpl_dir)
        self.event_dir = '%s/%s/events' % (self.data_dir, self.event_data)
        self.hist_dir = '%s/%s/hist' %  (self.data_dir, self.event_data)
        self.full_pic_dir = '%s/intensity' % self.data_dir
        self.pose_dir = '%s/pose' % self.data_dir
        self.kpt_pred_dir = '%s/outputs/%s_Keypoint/%s/infer_kpt' % (self.root_dir, self.exp_name, self.kpt_pred)


        self.cam_params = {
            # camera parameters for davis without IR-cut filter
            'normal': {
                'front': {
                    'intr': np.array([284.3, 284.8, 181.6, 137.9]),
                    't': np.array([-658.63292169, 1297.71397837, 3045.2672179 ]),
                    'rt': np.array([3.10456165, 0.04137824, 0.34360406]),
                    'dist': np.array([-4.1342606952675842e-01, 2.8509001608095730e-01, 8.7219928574097118e-04, -1.7010450394202813e-04, -1.2978321719269459e-01]),
                },
                'side': {
                    'intr': np.array([350.2, 350.7, 167.8, 125.9]),
                    't': np.array([-204.8878843, 1552.3306826, 2895.70436146]),
                    'rt': np.array([2.97453669, 0.12318467, 0.62776467]),
                    'dist': np.array([-3.6373981424108720e-01, 2.3787287446672745e-01, 6.4882338729224174e-05, 2.6368188020238389e-04, -1.3071116806919608e-01]),
                }
            },
            # camera parameters for davis with IR-cut filter
            'ir_cut': {
                'front': {
                    'intr': np.array([259.8, 259.2, 170.3, 131.8]),
                    't': np.array([-485.52366851, 1611.03889448, 2154.62412025]),
                    'rt': np.array([3.01950558e+00, 2.10473492e-03, 1.57264630e-01]),
                    'dist': np.array([-4.0551910083966131e-01, 2.7291405952374148e-01, -1.0306786122204741e-03, -3.0386906225556237e-04, -1.6172633923512664e-01]),
                },
                'side': {
                    'intr': np.array([266.2, 266.2, 180.8, 145.6]),
                    't': np.array([-278.95317967, 1189.84493567, 2276.50255893]),
                    'rt': np.array([-2.90034076, -0.02790158, -1.05976446]),
                    'dist': np.array([-4.3313575731991150e-01, 3.4860344388675035e-01, -4.3669553558982506e-03, 1.3855216670318003e-04, -2.5597607859325744e-01]),
                }
            }
        }
        self.img_w = 260
        self.img_h = 346

        self.smpl = SMPL(self.smpl_dir, self.batch_length)
        self.prim_kp = [0, 4, 5, 7, 8, 9, 15, 16, 17, 18, 19, 20, 21]
        self.kp_dist = [5.4698, 65.6850]

        self.all_clips = self.obtain_clips(mode)
        if cfg.data_size and mode == 'test':
            # randomly extract clips from all_clips with length of data_size (use fixed random seed)
            np.random.seed(0)
            self.all_clips = np.random.choice(self.all_clips, cfg.data_size, replace=False)

        if self.large_memory:
            self.data_cache = {}



    def obtain_clips(self, mode):

        all_clips = []
        actions = sorted(os.listdir(self.pose_dir))
        action_names = []
        if self.dataset_div == 'all':
            action_names = actions
        else:
            test_list = [
                # '20230623_subject01_suit_light_fast_take03',
                # '20230623_subject02_suit_light_medium_take03',
                # '20230623_subject03_suit_light_medium_take03',
                # '20230623_subject01_suit_light_kickboxing_take02',
                # '20230623_subject02_suit_light_baseball_take02',
                '20230626_subject01_suit_dark_medium_take03',
                '20230626_subject02_suit_dark_fast_take03',
                '20230626_subject03_suit_dark_fast_take03',
                '20230626_subject03_suit_dark_dance_take02'
            ]
            # test_list = ['take20230521_subject01_dance1']
            for action in actions:
                if mode == 'train':
                    if action in test_list:
                        continue
                    action_names.append(action)
                elif mode == 'test':
                    if action not in test_list:
                        continue
                    action_names.append(action)
                else:
                    raise ValueError('mode should be either train or test')


        for action in action_names:
            # if not ('dark' in action) and ('202306' in action) or 'inference' in action:
            #     continue
            # if '20230626_subject02_suit_dark_baseball_take01' in action:
            #     continue
            with open('%s/info/info_%s_front.pkl' % (self.data_dir, action), "rb") as f:
                info_front = pickle.load(f)
            with open('%s/info/info_%s_side.pkl' % (self.data_dir, action), "rb") as f:
                info_side = pickle.load(f)
            pose_idx_front = info_front['pose_idx']
            pose_idx_side = info_side['pose_idx']
            pose_start_idx = info_front['pose_start_idx']
            event_start_idx_front = info_front['event_start_idx']
            event_start_idx_side = info_side['event_start_idx']
            seq_len = info_front['seq_len']

            # add pose_start_idx to each element of pose_idx_side and pose_idx_front
            pose_idx_front = np.array(pose_idx_front) + pose_start_idx
            pose_idx_side = np.array(pose_idx_side) + pose_start_idx

            dataset = 'normal' if '20230521' in action or '20230523' in action else 'ir_cut'

            # load front-view data
            for idx in range(seq_len - (pose_idx_front[self.batch_length] - pose_idx_front[0]) - (self.batch_length * 2)):
                if idx % self.stride != 0:
                    continue
                clip = {
                    'action': action,
                    'camera': 'front',
                    'pose_idx': pose_idx_front[idx:idx + self.batch_length],
                    'event_idx': event_start_idx_front + idx,
                    'cam_intr': self.cam_params[dataset]['front']['intr'],
                    'dist_coeffs': self.cam_params[dataset]['front']['dist'],
                }
                all_clips.append(clip)

            # continue
            # load side-view data
            for idx in range(seq_len - (pose_idx_side[self.batch_length] - pose_idx_side[0]) - (self.batch_length * 2)):
                if idx % self.stride != 0:
                    continue

                clip = {
                    'action': action,
                    'camera': 'side',
                    'pose_idx': pose_idx_side[idx:idx + self.batch_length],
                    'event_idx': event_start_idx_side + idx,
                    'cam_intr': self.cam_params[dataset]['side']['intr'],
                    'dist_coeffs': self.cam_params[dataset]['side']['dist'],
                }
                all_clips.append(clip)

        return all_clips

    def load_item(self, idx):
        clip = self.all_clips[idx]
        action = clip['action']
        pose_idx = clip['pose_idx']
        event_idx = clip['event_idx']
        camera = clip['camera']
        fx, fy, cx, cy = clip['cam_intr']
        k1, k2, p1, p2, k3 = clip['dist_coeffs']

        events, hist_path, full_pic_path, kpt_pred = [], [], [], []
        betas, thetas, trans, vertices, skeleton3D, skeleton2D = [], [], [], [], [], []
        for i in range(self.batch_length):
            pose_data = np.load('%s/%s/smpl%06i.npz' % (self.pose_dir, action, pose_idx[i]))
            betas.append(pose_data['beta'])
            thetas.append(pose_data['theta'])
            trans.append(pose_data['trans'])
            vertices.append(pose_data['vertice'])
            skeleton3D.append(pose_data['skeleton3D'])
            joints_2D = pose_data['skeleton2D_%s' % camera]
            for idx, joint in enumerate(joints_2D):
                joints_2D[idx] = self.distort_point(joint[0], joint[1], fx, fy, cx, cy, k1, k2, p1, p2, k3)
            skeleton2D.append(joints_2D)
            events.append(joblib.load('%s/%s_%s/%06i.pkl' % (self.event_dir, action, camera, event_idx + i)))
            hist_path.append('%s/%s_%s/%06i.png' % (self.hist_dir, action, camera, event_idx + i))
            full_pic_path.append('%s/%s_%s/%06i.png' % (self.full_pic_dir, action, camera, event_idx + i))
            if self.kpt_pred:
                kpt_pred.append(joblib.load('%s/%s_%s/pred/%06i.pkl' % (self.kpt_pred_dir, action, camera, event_idx + i)))

        betas_tensor = torch.tensor(np.asarray(betas), dtype=torch.float32).squeeze(1)
        thetas_tensor = torch.tensor(np.asarray(thetas), dtype=torch.float32).squeeze(1)
        pquat = batch_rodrigues(thetas_tensor.view(-1, 3)).view(self.batch_length, 24, 3, 3)
        trans_tensor =  torch.tensor(np.asarray(trans), dtype=torch.float32).squeeze(1)
        vertices_tensor = torch.tensor(np.asarray(vertices), dtype=torch.float32)
        skeleton3D_tensor = torch.tensor(np.asarray(skeleton3D), dtype=torch.float32)
        skeleton2D_tensor = torch.tensor(np.asarray(skeleton2D), dtype=torch.float32)
        events_tensor = torch.tensor(np.asarray(events), dtype=torch.float32)
        kpt_pred_tensor = torch.tensor(np.asarray(kpt_pred), dtype=torch.float32) if self.kpt_pred else torch.tensor(0)
        # print('action: %s, vert_mean: %f' % (action, torch.mean(vertices_tensor)))

        skeleton2D_tensor = skeleton2D_tensor[:, self.prim_kp, :2]
        target_x, target_y = [], []
        for j in range(self.batch_length):
            x, y, _ = utils.generate_sa_simdr(skeleton2D_tensor[j], num_joints=len(self.prim_kp))
            target_x.append(x)
            target_y.append(y)
        target_x_tensor = torch.tensor(target_x)
        target_y_tensor = torch.tensor(target_y)

        items = {
            'q': pquat,
            't': trans_tensor,
            'v': vertices_tensor,
            's3': skeleton3D_tensor,
            's2': skeleton2D_tensor,
            'l': trans_tensor,
            'b': betas_tensor,
            'pc': events_tensor,
            'gt_x': target_x_tensor,
            'gt_y': target_y_tensor,
            'kpt_pred': kpt_pred_tensor,
            'info': (action, idx, camera, hist_path, full_pic_path, pose_idx, event_idx),
            # 'kp_dist': self.kp_dist,
            # 'cx': center_x_tensor,
            # 'cy': center_y_tensor,
        }
        return items

    def __len__(self):
        return len(self.all_clips)

    def __getitem__(self, idx):
        if self.large_memory:
            if idx not in self.data_cache:
                self.data_cache[idx] = self.load_item(idx)
            return self.data_cache[idx]
        else:
            return self.load_item(idx)

    def distort_point(self, x, y, fx, fy, cx, cy, k1, k2, p1, p2, k3):
        x_normalized = (x - cx) / fx
        y_normalized = (y - cy) / fy

        r2 = x_normalized**2 + y_normalized**2
        radial_distortion = 1 + k1*r2 + k2*r2**2 + k3*r2**3

        tangential_distortion_x = 2*p1*x_normalized*y_normalized + p2*(r2 + 2*x_normalized**2)
        tangential_distortion_y = p1*(r2 + 2*y_normalized**2) + 2*p2*x_normalized*y_normalized

        x_distorted = (x_normalized * radial_distortion + tangential_distortion_x) * fx + cx
        y_distorted = (y_normalized * radial_distortion + tangential_distortion_y) * fy + cy

        return x_distorted, y_distorted

    def add_distortion(self, image, cam_intr, dist_coeffs):
        k1, k2, p1, p2, k3 = dist_coeffs
        fx, fy, cx, cy = cam_intr
        distorted_image = np.zeros_like(image)

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                x_distorted, y_distorted = self.distort_point(x, y, fx, fy, cx, cy, k1, k2, p1, p2, k3)
                if 0 <= x_distorted < image.shape[1] and 0 <= y_distorted < image.shape[0]:
                    distorted_image[int(y_distorted), int(x_distorted)] = image[y, x]

        return distorted_image


    def visualize_batch(self, info, data, output, epoch, iter, cfg):
        action, frame_idx = info
        _, _, camera, hist_path, full_pic_path, pose_idx, event_idx = data['info']
        pc_array = data['pc'][0].cpu().detach().numpy()
        gt_v = data['v'][0].cpu().detach().numpy()
        pred_v = output['v'][0].cpu().detach().numpy()
        s2 = data['s2'][0].cpu().detach().numpy()
        pred_x = output['pred_x'][0] if cfg.model.name == 'pose' else None
        pred_y = output['pred_y'][0] if cfg.model.name == 'pose' else None
        kpt_pred = data['kpt_pred'][0].cpu().detach().numpy().astype(np.int32) if cfg.kpt_pred else None
        hist_path = np.asarray(hist_path).transpose(1, 0)[0]
        full_pic_path = np.asarray(full_pic_path).transpose(1, 0)[0]

        if '20230521' in action[0] or '20230523' in action[0]:
            if camera[0] == 'front':
                cam_intr = self.cam_params['normal']['front']['intr']
                cam_t = self.cam_params['normal']['front']['t']
                cam_rt = self.cam_params['normal']['front']['rt']
                cam_dist = self.cam_params['normal']['front']['dist']
            else:
                cam_intr = self.cam_params['normal']['side']['intr']
                cam_t = self.cam_params['normal']['side']['t']
                cam_rt = self.cam_params['normal']['side']['rt']
                cam_dist = self.cam_params['normal']['side']['dist']
        else:
            if camera[0] == 'front':
                cam_intr = self.cam_params['ir_cut']['front']['intr']
                cam_t = self.cam_params['ir_cut']['front']['t']
                cam_rt = self.cam_params['ir_cut']['front']['rt']
                cam_dist = self.cam_params['ir_cut']['front']['dist']
            else:
                cam_intr = self.cam_params['ir_cut']['side']['intr']
                cam_t = self.cam_params['ir_cut']['side']['t']
                cam_rt = self.cam_params['ir_cut']['side']['rt']
                cam_dist = self.cam_params['ir_cut']['side']['dist']

        imgs = []
        for idx in range(self.batch_length):
            # if idx >= 5:
            #     break
            img = cv2.imread(full_pic_path[idx])
            # hist = cv2.imread(hist_path[idx])
            event_num = dict()
            for pc in pc_array[idx]:
                x, y, t, p = pc
                event_num[(x, y)] = event_num.get((x, y), 0) + 1
            hist = np.zeros((256, 256))
            for (x_temp,y_temp), s in event_num.items():
                hist[round(y_temp)][round(x_temp)] = s
            hist = (hist / hist.max() * 255).astype(np.uint8)
            # hist = cv2.cvtColor(cv2.applyColorMap(hist, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            hist = cv2.cvtColor(hist, cv2.COLOR_GRAY2BGR)
            hist_pred = hist.copy()
            for point in s2[idx]:
                cv2.circle(hist, (int(point[0]), int(point[1])), 3, (255, 255, 0), thickness=2)
            if cfg.model.name == 'pose':
                if not cfg.kpt_pred:
                    pred_point = utils.decode_batch_sa_simdr(pred_x, pred_y)
                else:
                    pred_point = kpt_pred
                for pred_p in pred_point[idx]:
                    cv2.circle(hist_pred, (int(pred_p[0]), int(pred_p[1])), 3, (255, 255, 0), thickness=2)

            faces = self.smpl.faces
            vertice_render = pred_v[idx].copy()
            vertice_render = vertice_render[:, [0, 2, 1]]
            vertice_render[:, 2] *= -1
            vertice_render = vertice_render * 1000
            dist_pred = np.abs(np.mean(vertice_render, axis=0)[2])
            render_img_pred = (utils.render_model(vertice_render, faces, self.img_w, self.img_h, cam_intr, cam_t,
                        cam_rt, near=0.1, far=10000 + dist_pred, img=np.zeros((self.img_h, self.img_w, 3))) * 255).astype(np.uint8)
            # Replace the non-zero luminance value of render_img_pred overlaid on img with the corresponding luminance value of img
            padding_size = int((self.img_h - self.img_w) / 2)
            padding_img = np.zeros((self.img_h, self.img_h, 3))
            padding_img[:, padding_size:padding_size + self.img_w, :] = render_img_pred
            render_img_pred = cv2.resize(padding_img.astype(np.uint8), (256, 256))
            render_img_pred = self.add_distortion(render_img_pred, cam_intr, cam_dist)
            render_img_pred = np.where(render_img_pred > 0, render_img_pred, img)

            vertice_render = gt_v[idx].copy()
            vertice_render = vertice_render[:, [0, 2, 1]]
            vertice_render[:, 2] *= -1
            vertice_render = vertice_render * 1000
            dist_gt = np.abs(np.mean(vertice_render, axis=0)[2])
            render_img_gt = (utils.render_model(vertice_render, faces, self.img_w, self.img_h, cam_intr, cam_t,
                        cam_rt, near=0.1, far=10000 + dist_gt, img=np.zeros((self.img_h, self.img_w, 3))) * 255).astype(np.uint8)
            # cv2.imshow('render_img_gt', render_img_gt)
            # cv2.waitKey(0)
            padding_size = int((self.img_h - self.img_w) / 2)
            padding_img = np.zeros((self.img_h, self.img_h, 3))
            padding_img[:, padding_size:padding_size + self.img_w, :] = render_img_gt
            render_img_gt = cv2.resize(padding_img.astype(np.uint8), (256, 256))
            render_img_gt = self.add_distortion(render_img_gt, cam_intr, cam_dist)
            render_img_gt = np.where(render_img_gt > 0, render_img_gt, img)

            if cfg.model.name == 'pose':
                imgs.append(np.hstack((img, hist, hist_pred, render_img_gt, render_img_pred)))
            else:
                imgs.append(np.hstack((img, hist, render_img_gt, render_img_pred)))
        imgs = np.vstack(tuple(imgs))
        img_path = '../outputs/%s/%s/imgs/%s' % (self.exp_name, self.start_time, self.mode)
        os.makedirs(img_path, exist_ok=True)
        plt.imsave(str(img_path) + '/epoch%04i_iter%05i.jpg' % (epoch, iter), imgs)


    def visualize_test_batch(self, data, output, cfg):
        action, item_idx, camera, hist_path, full_pic_path = data['info']
        pc_array = data['pc'].cpu().detach().numpy()
        gt_v = data['v'].cpu().detach().numpy()
        pred_v = output['v'].cpu().detach().numpy()
        s2 = data['s2'].cpu().detach().numpy()
        pred_x = output['pred_x'] if cfg.model.name == 'pose' else None
        pred_y = output['pred_y'] if cfg.model.name == 'pose' else None
        kpt_pred = data['kpt_pred'].cpu().detach().numpy().astype(np.int32) if cfg.kpt_pred else None
        hist_path = np.asarray(hist_path).transpose(1, 0)
        full_pic_path = np.asarray(full_pic_path).transpose(1, 0)


        imgs = []
        for batch_id in range(pc_array.shape[0]):
            if '20230521' in action[batch_id] or '20230523' in action[batch_id]:
                if camera[batch_id] == 'front':
                    cam_intr = self.cam_params['normal']['front']['intr']
                    cam_t = self.cam_params['normal']['front']['t']
                    cam_rt = self.cam_params['normal']['front']['rt']
                    cam_dist = self.cam_params['normal']['front']['dist']
                else:
                    cam_intr = self.cam_params['normal']['side']['intr']
                    cam_t = self.cam_params['normal']['side']['t']
                    cam_rt = self.cam_params['normal']['side']['rt']
                    cam_dist = self.cam_params['normal']['side']['dist']
            else:
                if camera[batch_id] == 'front':
                    cam_intr = self.cam_params['ir_cut']['front']['intr']
                    cam_t = self.cam_params['ir_cut']['front']['t']
                    cam_rt = self.cam_params['ir_cut']['front']['rt']
                    cam_dist = self.cam_params['ir_cut']['front']['dist']
                else:
                    cam_intr = self.cam_params['ir_cut']['side']['intr']
                    cam_t = self.cam_params['ir_cut']['side']['t']
                    cam_rt = self.cam_params['ir_cut']['side']['rt']
                    cam_dist = self.cam_params['ir_cut']['side']['dist']
            img_list = []
            for idx in range(self.batch_length):
                # if idx >= 5:
                #     break
                frame_idx = int(full_pic_path[batch_id, idx].split('/')[-1].split('.')[0])
                # if frame_idx > 200:
                #     return

                img = cv2.imread(full_pic_path[batch_id, idx])
                # hist = cv2.imread(hist_path[idx])
                event_num = dict()
                for pc in pc_array[batch_id, idx]:
                    x, y, t, p = pc
                    event_num[(x, y)] = event_num.get((x, y), 0) + 1
                hist = np.zeros((256, 256))
                for (x_temp,y_temp), s in event_num.items():
                    hist[round(y_temp)][round(x_temp)] = s
                hist = (hist / hist.max() * 255).astype(np.uint8)
                # hist = cv2.cvtColor(cv2.applyColorMap(hist, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                hist = cv2.cvtColor(hist, cv2.COLOR_GRAY2BGR)
                hist_pred = hist.copy()
                # for point in s2[batch_id, idx]:
                #     cv2.circle(hist, (int(point[0]), int(point[1])), 3, (255, 255, 0), thickness=2)
                # if cfg.model.name == 'pose':
                #     if not cfg.kpt_pred:
                #         pred_point = utils.decode_batch_sa_simdr(pred_x[batch_id], pred_y[batch_id])
                #     else:
                #         pred_point = kpt_pred
                #     for pred_p in pred_point[idx]:
                #         cv2.circle(hist_pred, (int(pred_p[0]), int(pred_p[1])), 3, (255, 255, 0), thickness=2)

                faces = self.smpl.faces
                dist_pred = np.abs(np.mean(pred_v[batch_id, idx] * 1000, axis=0)[2])
                render_img_pred = (utils.render_model(pred_v[batch_id, idx] * 1000, faces, self.img_w, self.img_h, cam_intr, cam_t,
                            cam_rt, near=0.1, far=10000 + dist_pred, img=np.zeros((self.img_h, self.img_w, 3))) * 255).astype(np.uint8)
                # Replace the non-zero luminance value of render_img_pred overlaid on img with the corresponding luminance value of img
                padding_size = int((self.img_h - self.img_w) / 2)
                padding_img = np.zeros((self.img_h, self.img_h, 3))
                padding_img[:, padding_size:padding_size + self.img_w, :] = render_img_pred
                render_img_pred = cv2.resize(padding_img.astype(np.uint8), (256, 256))
                render_img_pred = np.where(render_img_pred > 0, render_img_pred, img)

                dist_gt = np.abs(np.mean(gt_v[batch_id, idx] * 1000, axis=0)[2])
                render_img_gt = (utils.render_model(gt_v[batch_id, idx] * 1000, faces, self.img_w, self.img_h, cam_intr, cam_t,
                            cam_rt, near=0.1, far=10000 + dist_gt, img=np.zeros((self.img_h, self.img_w, 3))) * 255).astype(np.uint8)
                # cv2.imshow('render_img_gt', render_img_gt)
                # cv2.waitKey(0)
                padding_size = int((self.img_h - self.img_w) / 2)
                padding_img = np.zeros((self.img_h, self.img_h, 3))
                padding_img[:, padding_size:padding_size + self.img_w, :] = render_img_gt
                render_img_gt = cv2.resize(padding_img.astype(np.uint8), (256, 256))
                render_img_gt = np.where(render_img_gt > 0, render_img_gt, img)

                if cfg.model.name == 'pose':
                    concat_img = np.hstack((img, hist_pred, render_img_pred))
                else:
                    concat_img = np.hstack((img, hist, render_img_gt, render_img_pred))
                # cv2.imshow('render_img_gt', concat_img)
                # cv2.waitKey(0)

                img_path = '../outputs/%s/test/%s/imgs/%s_%s' % (self.exp_name, self.start_time, action[batch_id], camera[batch_id])
                os.makedirs(img_path, exist_ok=True)
                plt.imsave(str(img_path) + '/%05i.jpg' % (int(frame_idx)), concat_img)

    def export_test_results(self, data, output):
        action, item_idx, camera, hist_path, full_pic_path, pose_idx, event_idx = data['info']
        thetas_pred = output['q']
        B, T, J, _, _ = thetas_pred.shape
        # apply rotation matrix to root joint of theta to swap x and z axis
        # thetas_pred[:, :, 0] = torch.matmul(thetas_pred[:, :, 0], torch.tensor([[-1, 0, 1], [0, 1, 0], [0, 0, -1]], dtype=torch.float32).to(thetas_pred.device))
        thetas_pred = thetas_pred.view(B * T * J, 3, 3)
        thetas_pred = mat2axisangle(thetas_pred).view(B, T, J, 3).cpu().detach().numpy()
        trans_pred = output['t'].cpu().detach().numpy()
        betas_pred = output['b'].cpu().detach().numpy()
        thetas_gt = data['q']
        thetas_gt = thetas_gt.view(B * T * J, 3, 3)
        thetas_gt = mat2axisangle(thetas_gt).view(B, T, J, 3).cpu().detach().numpy()
        trans_gt = data['t'].cpu().detach().numpy()
        betas_gt = data['b'].cpu().detach().numpy()
        hist_path = np.asarray(hist_path).transpose(1, 0)
        full_pic_path = np.asarray(full_pic_path).transpose(1, 0)
        pose_idx = np.asarray(pose_idx)
        event_idx = np.asarray(event_idx)

        for batch_id in range(thetas_pred.shape[0]):
            if '20230521' in action[batch_id] or '20230523' in action[batch_id]:
                if camera[batch_id] == 'front':
                    cam_intr = self.cam_params['normal']['front']['intr']
                    cam_t = self.cam_params['normal']['front']['t']
                    cam_rt = self.cam_params['normal']['front']['rt']
                    cam_dist = self.cam_params['normal']['front']['dist']
                else:
                    cam_intr = self.cam_params['normal']['side']['intr']
                    cam_t = self.cam_params['normal']['side']['t']
                    cam_rt = self.cam_params['normal']['side']['rt']
                    cam_dist = self.cam_params['normal']['side']['dist']
            else:
                if camera[batch_id] == 'front':
                    cam_intr = self.cam_params['ir_cut']['front']['intr']
                    cam_t = self.cam_params['ir_cut']['front']['t']
                    cam_rt = self.cam_params['ir_cut']['front']['rt']
                    cam_dist = self.cam_params['ir_cut']['front']['dist']
                else:
                    cam_intr = self.cam_params['ir_cut']['side']['intr']
                    cam_t = self.cam_params['ir_cut']['side']['t']
                    cam_rt = self.cam_params['ir_cut']['side']['rt']
                    cam_dist = self.cam_params['ir_cut']['side']['dist']
            # print('Exporting %s_%s' % (action[batch_id], camera[batch_id]))
            for idx in range(self.batch_length):
                frame_idx = int(full_pic_path[batch_id, idx].split('/')[-1].split('.')[0])
                theta_pred = thetas_pred[batch_id, idx]
                tran_pred = trans_pred[batch_id, idx]
                beta_pred = betas_pred[batch_id, idx]
                theta_gt = thetas_gt[batch_id, idx]
                tran_gt = trans_gt[batch_id, idx]
                beta_gt = betas_gt[batch_id, idx]
                save_data = {
                    'theta_pred': theta_pred,
                    'tran_pred': tran_pred,
                    'beta_pred': beta_pred,
                    'theta_gt': theta_gt,
                    'tran_gt': tran_gt,
                    'beta_gt': beta_gt,
                    'cam_intr': cam_intr,
                    'cam_dist': cam_dist,
                    'cam_t': cam_t,
                    'cam_rt': cam_rt,
                    'action': action[batch_id],
                    'fullpic_path': full_pic_path[batch_id, idx],
                    'hist_path': hist_path[batch_id, idx],
                    'pose_idx': pose_idx[batch_id, idx],
                    'event_idx': event_idx[batch_id] + idx,
                }

                save_path = '../outputs/%s/test/%s/npz/%s_%s' % (self.exp_name, self.start_time, action[batch_id], camera[batch_id])
                os.makedirs(save_path, exist_ok=True)
                np.savez(str(save_path) + '/%06i.npz' % (int(frame_idx)), **save_data)


    # def visualize_keypoint(self, info, data, output, epoch, iter, mode, pose_data_dir, pose_module):
    #     action, frame_idx = info
    #     pc_array = data['pc'][0].cpu().detach().numpy()
    #     s2 = data['s2'][0].cpu().detach().numpy()
    #     pred_x = output['pred_x'][0] if pose_module else None
    #     pred_y = output['pred_y'][0] if pose_module else None
    #     imgs = []
    #     for idx in range(len(pc_array)):
    #         img_path = '%s/full_pic_256/%s/fullpic%04i.jpg' % (pose_data_dir, action, frame_idx + idx)
    #         img = cv2.imread(img_path)

    #         event_num = dict()
    #         for pc in pc_array[idx]:
    #             x, y, time = pc
    #             event_num[(x, y)] = event_num.get((x, y), 0) + 1
    #         hist = np.zeros((256, 256))
    #         for (x_temp,y_temp), s in event_num.items():
    #             hist[int(y_temp)][int(x_temp)] = s
    #         hist = (hist / hist.max() * 255).astype(np.uint8)
    #         # hist = cv2.cvtColor(cv2.applyColorMap(hist, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    #         hist = cv2.cvtColor(hist, cv2.COLOR_GRAY2BGR)
    #         hist_pred = hist.copy()
    #         for point in s2[idx]:
    #             cv2.circle(hist, tuple(point), 3, (255, 255, 0), thickness=2)
    #         if pose_module:
    #             pred_point = utils.decode_batch_sa_simdr(pred_x, pred_y)
    #             for pred_p in pred_point[idx]:
    #                 cv2.circle(hist_pred, tuple(pred_p), 3, (255, 255, 0), thickness=2)


    #         imgs.append(np.vstack((img, hist, hist_pred)))
    #     imgs = np.hstack(tuple(imgs))
    #     img_path = '../outputs/%s/imgs/%s' % (self.exp_name, mode)
    #     os.makedirs(img_path, exist_ok=True)
    #     plt.imsave(str(img_path) + '/epoch%04i_iter%05i.jpg' % (epoch, iter), imgs)

    # def vis_grouped_points(self, pc, anchor):
    #     L, N = pc.shape[0], pc.shape[1]
    #     S = anchor.shape[1]
    #     dist = torch.cdist(pc[:, :, :2], anchor, p=2)
    #     sorted_points, sort_idx = torch.sort(dist, dim=1)
    #     count = np.count_nonzero(sorted_points < 20, axis=1)

    #     imgs = []
    #     vis_joint = 5

    #     for i in range(L):
    #         event_num = dict()
    #         for pc in pc[i].cpu().detach().numpy():
    #             x, y, time = pc
    #             event_num[(x, y)] = event_num.get((x, y), 0) + 1
    #         hist = np.zeros((256, 256))
    #         for (x_temp,y_temp), s in event_num.items():
    #             hist[int(y_temp)][int(x_temp)] = s
    #         hist = (hist / hist.max() * 255).astype(np.uint8)
    #         hist = cv2.cvtColor(hist, cv2.COLOR_GRAY2BGR)
    #         for point in anchor[i]:
    #             cv2.circle(hist, tuple(point), 20, (255, 0, 0), thickness=1)

    #         hist_bg = hist.copy()
    #         for p in range(count[i, vis_joint]):
    #             cv2.circle(hist_bg, tuple(pc[i, sort_idx[i, p, vis_joint], :2]), 1, (0, 0, 255), thickness=1)
    #         imgs.append(np.vstack([hist, hist_bg]))
    #     cv2.imshow('img', np.hstack(imgs))
    #     print()


    # def visualize_bbox(self, info, data, output, epoch, iter, cfg):
    #     action, frame_idx = info
    #     pc_array = data['pc'][0].cpu().detach().numpy()
    #     gt_v = data['v'][0].cpu().detach().numpy()
    #     cx_gt = data['cx'][0]
    #     cy_gt = data['cy'][0]
    #     size_gt = data['size'][0].cpu().detach().numpy()
    #     cx_pred = output['cx'][0] if cfg.pose_module else None
    #     cy_pred = output['cy'][0] if cfg.pose_module else None
    #     gt_point = utils.decode_batch_sa_simdr(cx_gt, cy_gt)
    #     pred_point = utils.decode_batch_sa_simdr(cx_pred, cy_pred)

    #     size_pred = output['size'][0] if cfg.pose_module else None
    #     imgs = []
    #     for idx in range(len(pc_array)):
    #         img_path = '%s/full_pic_256/%s/fullpic%04i.jpg' % (cfg.pose_data_dir, action, frame_idx + idx)
    #         img = cv2.imread(img_path)

    #         event_num = dict()
    #         for pc in pc_array[idx]:
    #             x, y, time = pc
    #             event_num[(x, y)] = event_num.get((x, y), 0) + 1
    #         hist = np.zeros((256, 256))
    #         for (x_temp,y_temp), s in event_num.items():
    #             hist[int(y_temp)][int(x_temp)] = s
    #         hist = (hist / hist.max() * 255).astype(np.uint8)
    #         # hist = cv2.cvtColor(cv2.applyColorMap(hist, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    #         hist = cv2.cvtColor(hist, cv2.COLOR_GRAY2BGR)
    #         hist_pred = hist.copy()
    #         dist_gt = np.abs(np.mean(gt_v[idx], axis=0)[2])
    #         faces = self.smpl.faces
    #         img_path = '%s/full_pic_256/%s/fullpic%04i.jpg' % (self.pose_data_dir, action, frame_idx + idx)
    #         img = cv2.imread(img_path)
    #         render_img_gt = (utils.render_model(gt_v[idx], faces, self.img_size, self.img_size, self.cam_intr, np.zeros([3]),
    #                         np.zeros([3]), near=0.1, far=20 + dist_gt, img=img) * 255).astype(np.uint8)

    #         # cv2.circle(hist, (gt_point[idx, 0, 0], gt_point[idx, 0, 1]), 3, (255, 255, 0), thickness=2)
    #         ul = (int(gt_point[idx, 0, 0] - size_gt[idx, 0] // 2), int(gt_point[idx, 0, 1] - size_gt[idx, 1] // 2))
    #         br = (int(gt_point[idx, 0, 0] + size_gt[idx, 0] // 2), int(gt_point[idx, 0, 1] + size_gt[idx, 1] // 2))
    #         cv2.rectangle(hist, ul, br, (255, 255, 0))


    #         # cv2.circle(hist_pred, (pred_point[idx, 0, 0], pred_point[idx, 0, 1]), 3, (255, 255, 0), thickness=2)
    #         ul = (int(pred_point[idx, 0, 0] - size_pred[idx, 0] // 2), int(pred_point[idx, 0, 1] - size_pred[idx, 1] // 2))
    #         br = (int(pred_point[idx, 0, 0] + size_pred[idx, 0] // 2), int(pred_point[idx, 0, 1] + size_pred[idx, 1] // 2))
    #         cv2.rectangle(hist_pred, ul, br, (255, 255, 0))

    #         imgs.append(np.vstack((img, render_img_gt, hist, hist_pred)))
    #     imgs = np.hstack(tuple(imgs))
    #     img_path = '../outputs/%s/%s/imgs/%s' % (self.exp_name, self.start_time, self.mode)
    #     os.makedirs(img_path, exist_ok=True)
    #     plt.imsave(str(img_path) + '/epoch%04i_iter%05i.jpg' % (epoch, iter), imgs)

if __name__ == '__main__':
    # get args using arg_parse
    import argparse
    import yaml
    from addict import Dict
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='config file(.yaml)')
    args = parser.parse_args()
    cfg = Dict(yaml.safe_load(open(args.config_path)))
    start_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    dataset_train = DavisDataloader(cfg, 'train', start_time)
    data = dataset_train.__getitem__(0)
    print(len(dataset_train))
