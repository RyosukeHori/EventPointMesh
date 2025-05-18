import numpy as np
import cv2
from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
import torch

colors = {
    'pink': [.7, .7, .9],
    'neutral': [.9, .9, .8],
    'capsule': [.7, .75, .5],
    'yellow': [.5, .7, .75],
}


def _create_renderer(w=640,
                     h=480,
                     rt=np.zeros(3),
                     t=np.zeros(3),
                     f=None,
                     c=None,
                     k=None,
                     near=.01,
                     far=10.):

    f = np.array([w, w]) / 2. if f is None else f
    c = np.array([w, h]) / 2. if c is None else c
    k = np.zeros(5) if k is None else k

    rn = ColoredRenderer()

    rn.camera = ProjectPoints(rt=rt, t=t, f=f, c=c, k=k)
    rn.frustum = {'near': near, 'far': far, 'height': h, 'width': w}
    return rn


def _rotateY(points, angle):
    """Rotate the points by a specified angle."""
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)], [0., 1., 0.],
        [-np.sin(angle), 0., np.cos(angle)]
    ])
    return np.dot(points, ry)


def simple_renderer(rn, verts, faces, yrot=np.radians(120)):

    # Rendered model color
    color = colors['pink']

    rn.set(v=verts, f=faces, vc=color, bgcolor=np.ones(3))

    albedo = rn.vc

    # Construct Back Light (on back right corner)
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        # light_pos=_rotateY(np.array([-200, -100, -100]), yrot),
        light_pos=np.array([2000, 2000, 2000]),
        vc=albedo,
        light_color=np.array([0.8, 0.8, 0.8]))

    # Construct Left Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        # light_pos=_rotateY(np.array([800, 10, 300]), yrot),
        light_pos=np.array([-2000, 2000, 2000]),
        vc=albedo,
        light_color=np.array([0.8, 0.8, 0.8]))

    # # Construct Right Lightm
    # rn.vc += LambertianPointLight(
    #     f=rn.f,
    #     v=rn.v,
    #     num_verts=len(rn.v),
    #     # light_pos=_rotateY(np.array([-500, 500, 1000]), yrot),
    #     light_pos=np.array([-2000, 4000, 3000]),
    #     vc=albedo,
    #     light_color=np.array([.7, .7, .7]))

    return rn.r


def simple_renderer_EHPE(rn, verts, faces, yrot=np.radians(120)):

    # Rendered model color
    color = colors['pink']

    rn.set(v=verts, f=faces, vc=color, bgcolor=np.ones(3))

    albedo = rn.vc

    # Construct Back Light (on back right corner)
    rn.vc = LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-200, -100, -100]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Left Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([800, 10, 300]), yrot),
        vc=albedo,
        light_color=np.array([1, 1, 1]))

    # Construct Right Light
    rn.vc += LambertianPointLight(
        f=rn.f,
        v=rn.v,
        num_verts=len(rn.v),
        light_pos=_rotateY(np.array([-500, 500, 1000]), yrot),
        vc=albedo,
        light_color=np.array([.7, .7, .7]))

    return rn.r



def get_alpha(imtmp, bgval=1.):
    h, w = imtmp.shape[:2]
    alpha = (~np.all(imtmp == bgval, axis=2)).astype(imtmp.dtype)

    b_channel, g_channel, r_channel = cv2.split(imtmp)

    im_RGBA = cv2.merge(
        (b_channel, g_channel, r_channel, alpha.astype(imtmp.dtype)))
    return im_RGBA


def render_model(verts, faces, w, h, cam_param, cam_t, cam_rt, near=0.5, far=25, img=None, EHPE=False):
    f = cam_param[0:2]
    c = cam_param[2:4]
    rn = _create_renderer(w=w, h=h, near=near, far=far, rt=cam_rt, t=cam_t, f=f, c=c)
    # Uses img as background, otherwise white background.
    if img is not None:
        rn.background_image = img / 255. if img.max() > 1 else img

    imtmp = simple_renderer_EHPE(rn, verts, faces) if EHPE else simple_renderer(rn, verts, faces)

    # If white bg, make transparent.
    if img is None:
        imtmp = get_alpha(imtmp)

    return imtmp


def render_depth_v(verts, faces, require_visi = False,
                   t = [0.,0.,0.], img_size=[448, 448], f=[400.0,400.0], c=[224.,224.]):
    from opendr.renderer import DepthRenderer
    rn = DepthRenderer()
    rn.camera = ProjectPoints(rt = np.zeros(3),
                              t = t,
                              f = f,
                              c = c,
                              k = np.zeros(5))
    rn.frustum = {'near': .01, 'far': 10000.,
                  'width': img_size[1], 'height': img_size[0]}
    rn.v = verts
    rn.f = faces
    rn.bgcolor = np.zeros(3)
    if require_visi is True:
        return rn.r, rn.visibility_image
    else:
        return rn.r


# others
def projection(xyz, intr_param, cam_t=None, cam_rt=None, simple_mode=False):
    # xyz: [N, 3]
    # intr_param: (fx, fy, cx, cy, w, h, k1, k2, p1, p2, k3, k4, k5, k6)
    assert xyz.shape[1] == 3
    fx, fy, cx, cy = intr_param[0:4]

    if cam_t is not None and cam_rt is not None:
        R = cv2.Rodrigues(cam_rt)[0]
        xyz = np.dot(xyz, R.T) + cam_t

    if not simple_mode:
        k1, k2, p1, p2, k3, k4, k5, k6 = intr_param[6:14]

        x_p = xyz[:, 0] / xyz[:, 2]
        y_p = xyz[:, 1] / xyz[:, 2]
        r2 = x_p ** 2 + y_p ** 2

        a = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        b = 1 + k4 * r2 + k5 * r2 ** 2 + k6 * r2 ** 3
        b = b + (b == 0)
        d = a / b

        x_pp = x_p * d + 2 * p1 * x_p * y_p + p2 * (r2 + 2 * x_p ** 2)
        y_pp = y_p * d + p1 * (r2 + 2 * y_p ** 2) + 2 * p2 * x_p * y_p

        u = fx * x_pp + cx
        v = fy * y_pp + cy
        d = xyz[:, 2]

        return np.stack([u, v, d], axis=1)
    else:
        u = xyz[:, 0] / xyz[:, 2] * fx + cx
        v = xyz[:, 1] / xyz[:, 2] * fy + cy
        d = xyz[:, 2]

        return np.stack([u, v, d], axis=1)


def adjust_target_weight(joint, target_weight, tmp_size, sx=346, sy=260):
    mu_x = joint[0]
    mu_y = joint[1]
    # Check that any part of the gaussian is in-bounds
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= sx or ul[1] >= sy or br[0] < 0 or br[1] < 0:
        # If not, just return the image as is
        target_weight = 0

    return target_weight


def generate_sa_simdr(joints, sigma=8, sx=256, sy=256, num_joints=13):
    """
    joints:  [num_joints, 3]

    return => target, target_weight(1: visible, 0: invisible)
    """

    target_weight = np.ones((num_joints, 1), dtype=np.float32)

    target_x = np.zeros((num_joints, int(sx)), dtype=np.float32)
    target_y = np.zeros((num_joints, int(sy)), dtype=np.float32)

    tmp_size = sigma * 3

    frame_size = np.array([sx, sy])
    frame_resize = np.array([sx, sy])
    feat_stride = frame_size / frame_resize

    for joint_id in range(num_joints):
        target_weight[joint_id] = adjust_target_weight(joints[joint_id], target_weight[joint_id], tmp_size)
        if target_weight[joint_id] == 0:
            continue

        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)

        x = np.arange(0, int(sx), 1, np.float32)
        y = np.arange(0, int(sy), 1, np.float32)

        v = target_weight[joint_id]
        if v > 0.5:
            target_x[joint_id] = (np.exp(- ((x - mu_x) ** 2) / (2 * sigma ** 2))) / (sigma * np.sqrt(np.pi * 2))
            target_y[joint_id] = (np.exp(- ((y - mu_y) ** 2) / (2 * sigma ** 2))) / (sigma * np.sqrt(np.pi * 2))

            # norm to [0,1]
            target_x[joint_id] = (target_x[joint_id] - target_x[joint_id].min()) / (
                    target_x[joint_id].max() - target_x[joint_id].min())
            target_y[joint_id] = (target_y[joint_id] - target_y[joint_id].min()) / (
                    target_y[joint_id].max() - target_y[joint_id].min())

    return target_x, target_y, target_weight


def decode_batch_sa_simdr(output_x, output_y):

    max_val_x, preds_x = output_x.max(2, keepdim=True)
    max_val_y, preds_y = output_y.max(2, keepdim=True)

    output = torch.ones([output_x.size(0), preds_x.size(1), 2])
    output[:, :, 0] = torch.squeeze(preds_x, dim=-1)
    output[:, :, 1] = torch.squeeze(preds_y, dim=-1)

    output = output.cpu().numpy()
    preds = output.copy()

    return preds


def cal_2D_mpjpe(gt, pred):
    gt = gt.cpu().detach().numpy()
    gt_float = gt.astype(np.float)
    pred_float = pred.astype(np.float)
    dist_2d = np.linalg.norm((gt_float - pred_float), axis=-1)
    mpjpe2D = np.nanmean(dist_2d)

    return mpjpe2D


class KLDiscretLoss(torch.nn.Module):
    """
    "https://github.com/leeyegy/SimDR"
    """
    def __init__(self):
        super(KLDiscretLoss, self).__init__()
        self.LogSoftmax = torch.nn.LogSoftmax(dim=1)  # [B,LOGITS]
        self.criterion_ = torch.nn.KLDivLoss(reduction='none')

    def criterion(self, dec_outs, labels):
        scores = self.LogSoftmax(dec_outs)
        loss = torch.mean(self.criterion_(scores, labels), dim=1)
        return loss

    def forward(self, output_x, output_y, target_x, target_y):
        num_joints = output_x.size(1)
        # print(num_joints)
        loss = 0
        for idx in range(num_joints):
            coord_x_pred = output_x[:, idx]
            coord_y_pred = output_y[:, idx]
            coord_x_gt = target_x[:, idx]
            coord_y_gt = target_y[:, idx]
            loss += (self.criterion(coord_x_pred, coord_x_gt).mean())
            loss += (self.criterion(coord_y_pred, coord_y_gt).mean())

        return loss / num_joints


def batch_compute_similarity_transform_torch(S1, S2, return_transform=False):
    '''
    S1: prediction
    S2: target
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (B x 3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.permute(0, 2, 1)
        S2 = S2.permute(0, 2, 1)
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=-1, keepdims=True)
    mu2 = S2.mean(axis=-1, keepdims=True)

    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = torch.sum(X1**2, dim=1).sum(dim=1)

    # 3. The outer product of X1 and X2.
    K = X1.bmm(X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, V = torch.svd(K)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=S1.device).unsqueeze(0)
    Z = Z.repeat(U.shape[0], 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.det(U.bmm(V.permute(0, 2, 1))))

    # Construct R.
    R = V.bmm(Z.bmm(U.permute(0,2,1)))

    # 5. Recover scale.
    scale = torch.cat([torch.trace(x).unsqueeze(0) for x in R.bmm(K)]) / var1

    # 6. Recover translation.
    t = mu2 - (scale.unsqueeze(-1).unsqueeze(-1) * (R.bmm(mu1)))

    # 7. Error:
    S1_hat = scale.unsqueeze(-1).unsqueeze(-1) * R.bmm(S1) + t

    if transposed:
        S1_hat = S1_hat.permute(0, 2, 1)

    if return_transform:
        return S1_hat, scale, R, t

    return S1_hat


def compute_mpjpe(pred, target):
    # [B, T, 24, 3]
    mpjpe = torch.sqrt(torch.sum((pred - target) ** 2, dim=-1))
    return mpjpe


def compute_pa_mpjpe(pred, target):
    B, T, _, _ = pred.size()
    pred_hat = batch_compute_similarity_transform_torch(pred.view(-1, 24, 3), target.view(-1, 24, 3))
    pa_mpjpe = torch.sqrt(torch.sum((pred_hat - target.view(-1, 24, 3)) ** 2, dim=-1))
    return pa_mpjpe.view(B, T, 24)


def compute_pelvis_mpjpe(pred, target):
    # [B, T, 24, 3]
    left_heap_idx = 1
    right_heap_idx = 2
    pred_pel = (pred[:, :, left_heap_idx:left_heap_idx+1, :] + pred[:, :, right_heap_idx:right_heap_idx+1, :]) / 2
    pred = pred - pred_pel
    target_pel = (target[:, :, left_heap_idx:left_heap_idx+1, :] + target[:, :, right_heap_idx:right_heap_idx+1, :]) / 2
    target = target - target_pel
    pel_mpjpe = torch.sqrt(torch.sum((pred - target) ** 2, dim=-1))
    return pel_mpjpe


def compute_pck(pred, target):
    pel_mpjpe = compute_pelvis_mpjpe(pred, target)
    pck = pel_mpjpe < 0.1
    return pck


def compute_pck_head(pred, target):
    # 0.5 head PCKh@0.5
    pel_mpjpe = compute_pelvis_mpjpe(pred, target)  # [B, T, 24]
    neck_idx = 12
    head_idx = 15
    thre = 0.5 * 2 * torch.sqrt(torch.sum(
        (target[:, :, neck_idx:neck_idx+1, :] - target[:, :, head_idx:head_idx+1, :]) ** 2, dim=-1))
    pck = pel_mpjpe < thre
    return pck


def compute_pck_torso(pred, target):
    # 0.2 torso PCK@0.2
    pel_mpjpe = compute_pelvis_mpjpe(pred, target)
    neck_idx = 12
    pel_idx = 0
    thre = 0.2 * torch.sqrt(torch.sum(
        (target[:, :, neck_idx:neck_idx + 1, :] - target[:, :, pel_idx:pel_idx + 1, :]) ** 2, dim=-1))
    pck = pel_mpjpe < thre
    return pck
