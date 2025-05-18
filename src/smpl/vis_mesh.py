import utils
import cv2
import numpy as np
from src.smpl.smpl_utils_extend import SMPL
import pickle
import os

img_size = 256
scale = img_size / 1280.
cam_intr = np.array([1679.3, 1679.3, 641, 641]) * scale

def visualize(v, g, batch_size, batch_length, V_NO, iter, mode):
    smpl_m_path = 'smpl/smpl_m.pkl'
    smpl_f_path = 'smpl/smpl_f.pkl'
    with open(smpl_m_path, 'rb') as f:
        smpl_m = pickle.load(f)
    with open(smpl_f_path, 'rb') as f:
        smpl_f = pickle.load(f)

    for batch_idx in range(batch_size):
        for frame_idx in range(batch_length):
            verts = v[batch_idx, frame_idx]

            faces = smpl_m['f'] if g[batch_idx, frame_idx] > 0.5 else smpl_f['f']
            dist = np.abs(np.mean(verts, axis=0)[2])
            render_img = (utils.render_model(verts, faces, img_size, img_size, cam_intr, np.zeros([3]),
                        np.zeros([3]), near=0.1, far=20 + dist, img=np.ones((256, 256, 3))) * 255).astype(np.uint8)

            # plt.imshow(render_img)
            # plt.axis('off')
            # plt.show()
            img_name = '%s_%05i.png' % (frame_idx)
            save_dir = './%s/iter%05i/' % (V_NO, iter)
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite('%s/%s' % (save_dir, img_name), render_img)

