import torch
import torch.nn as nn
from smpl.smpl_utils_extend import SMPL
from torchvision.models import resnet50
import utils


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def point_ball_set(nsample, xyz, new_xyz):
    """
    Input:
        nsample: number of points to sample
        xyz: all points, [B, N, 3] = [40, 7500, 3]
        new_xyz: anchor points [B, S, 3] = [40, 13, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])  # [40, 13, 7500]
    sqrdists = torch.cdist(new_xyz, xyz[:, :, :2])  # [40, 13, 7500]
    sorted_points, sort_idx = torch.sort(sqrdists)
    sort_idx=sort_idx[:, :, :500]
    batch_idx=torch.arange(B, dtype=torch.long).to(device).view((B,1,1)).repeat((1,S,nsample))
    centroids_idx=torch.arange(S, dtype=torch.long).to(device).view((1,S,1)).repeat((B,1,nsample))
    return group_idx[batch_idx, centroids_idx, sort_idx]  # [[40, 13, 500], [40, 13, 500], [40, 13, 500]]

def AnchorJointGrouping(anchors, nsample, xyz, points):
    """
    Input:
        anchors: [B, 13, 2], njoint=13
        nsample: number of points to sample
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D] = [40, 7500, 24]
    Return:
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C_xyz = xyz.shape
    _, S, C_xy = anchors.shape  # s: 13
    idx = point_ball_set(nsample, xyz, anchors)  # [40, 12, 500]
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C] = [40, 13, 500, 3]
    grouped_anchors=anchors.view(B, S, 1, C_xy).repeat(1,1,nsample,1)  # [40, 13, 500, 2]
    grouped_xyz_norm = grouped_xyz[:, :, :, :2] - grouped_anchors  # [40, 13, 500, 2]

    grouped_points = index_points(points, idx)  # [40, 13, 500, 24]
    new_points = torch.cat([grouped_anchors, grouped_xyz_norm, grouped_points, grouped_xyz], dim=-1) # [40, 13, 500, 31]
    return new_points

class AnchorPointNet(nn.Module):
    def __init__(self, polarity=1):
        super(AnchorPointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=24+4+3+polarity,   out_channels=32,  kernel_size=1)
        self.cb1 = nn.BatchNorm1d(32)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=32,  out_channels=48, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(48)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=48, out_channels=64, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(64)
        self.caf3 = nn.ReLU()

        self.attn=nn.Linear(64, 1)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x):
        x = x.transpose(1,2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x))) #(Batch, feature, frame_point_number)

        x = x.transpose(1,2)

        attn_weights=self.softmax(self.attn(x))
        attn_vec=torch.sum(x*attn_weights, dim=1)
        return attn_vec, attn_weights

class AnchorPoseConvNet(nn.Module):
    def __init__(self):
        super(AnchorPoseConvNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=64, out_channels=96, kernel_size=5)
        self.cb1 = nn.BatchNorm1d(96)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=96, out_channels=128, kernel_size=5)
        self.cb2 = nn.BatchNorm1d(128)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=5)
        self.cb3 = nn.BatchNorm1d(64)
        self.caf3 = nn.ReLU()

    def forward(self, x):
        B=x.size()[0]
        x=x.permute(0, 2, 1)

        x=self.caf1(self.cb1(self.conv1(x)))
        x=self.caf2(self.cb2(self.conv2(x)))
        x=self.caf3(self.cb3(self.conv3(x)))

        x=x.view(B, 64)
        return x

class AnchorPoseRNN(nn.Module):
    def __init__(self, bidirectional):
        super(AnchorPoseRNN, self).__init__()
        self.bidirectional = bidirectional
        self.rnn=nn.LSTM(input_size=64, hidden_size=64, num_layers=3, batch_first=True, dropout=0.1, bidirectional=bidirectional)
        self.fc = nn.Linear(128, 64)

    def forward(self, x, h0, c0):
        a_vec, (hn, cn)=self.rnn(x, (h0, c0))
        if self.bidirectional:
            a_vec = self.fc(a_vec)
        return a_vec, hn, cn

class AnchorPoseModule(nn.Module):
    def __init__(self, device, bidirectional, polarity=1):
        super(AnchorPoseModule, self).__init__()
        self.device = device
        self.apointnet=AnchorPointNet(polarity)
        self.aconv=AnchorPoseConvNet()
        self.arnn=AnchorPoseRNN(bidirectional)

    def forward(self, x, pred_x, pred_y, h0, c0, B, L, feature_size, kpt_pred=None):
        B, L, joint_num, img_size = pred_x.shape
        pred_x = pred_x.view(B * L, joint_num, img_size)
        pred_y = pred_y.view(B * L, joint_num, img_size)
        if kpt_pred == None:
            anchors = utils.decode_batch_sa_simdr(pred_x, pred_y)
            anchors = torch.tensor(anchors).to(self.device)
        else:
            anchors = kpt_pred.view(B * L, joint_num, 2)
        nsample = 500
        grouped_points=AnchorJointGrouping(anchors, nsample, xyz=x[..., :3], points=x[..., 3:])  # [40, 12, 500, 28]
        grouped_points=grouped_points.view(B*L*joint_num, nsample, 4+feature_size)  # [480, 500, 30]
        feat_points, attn_weights=self.apointnet(grouped_points)
        feat_points=feat_points.view(B*L, 13, 64)
        feat_vec=self.aconv(feat_points)
        feat_vec=feat_vec.view(B, L, 64)
        a_vec, hn, cn=self.arnn(feat_vec, h0, c0)
        return a_vec, attn_weights, hn, cn

class BasePointNet(nn.Module):
    def __init__(self, polarity=1):
        super(BasePointNet, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=3+polarity,   out_channels=8,  kernel_size=1)
        self.cb1 = nn.BatchNorm1d(8)
        self.caf1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=8,  out_channels=16, kernel_size=1)
        self.cb2 = nn.BatchNorm1d(16)
        self.caf2 = nn.ReLU()

        self.conv3 = nn.Conv1d(in_channels=16, out_channels=24, kernel_size=1)
        self.cb3 = nn.BatchNorm1d(24)
        self.caf3 = nn.ReLU()

    def forward(self, in_mat):
        x = in_mat.transpose(1,2)

        x = self.caf1(self.cb1(self.conv1(x)))
        x = self.caf2(self.cb2(self.conv2(x)))
        x = self.caf3(self.cb3(self.conv3(x)))

        x = x.transpose(1,2)
        x = torch.cat((in_mat, x), -1)

        return x

class PosePointNet(nn.Module):
    def __init__(self, polarity=1):
        super(PosePointNet, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv1d(24+3+polarity, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv1d(64, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(128),
                                   nn.LeakyReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(128, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(256),
                                   nn.LeakyReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(1024),
                                   nn.LeakyReLU())
        self.out_conv = nn.Linear(1024 * 2, 1024)

    def forward(self, x):  # [40, 7500, 27]
        B = x.size(0)
        x = x.transpose(1,2)  # [40, 27, 7500]

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        # x = x.transpose(1,2)

        x1 = nn.functional.adaptive_max_pool1d(x, 1).view(B, -1)
        x2 = nn.functional.adaptive_avg_pool1d(x, 1).view(B, -1)
        x = torch.cat((x1, x2), 1)

        x = self.out_conv(x)

        return x

class PosePointModule(nn.Module):
    def __init__(self, bidirectional, img_size_H=256, img_size_W=256, num_joints=13, polarity=1):
        super(PosePointModule, self).__init__()
        self.img_size_H = img_size_H
        self.img_size_W = img_size_W
        self.num_joints = num_joints
        self.ppointnet=PosePointNet(polarity)
        self.prnn=nn.LSTM(input_size=1024, hidden_size=1024, num_layers=3, batch_first=True, dropout=0.1, bidirectional=bidirectional)

        self.input_size = 2048 if bidirectional else 1024
        self.conv = nn.Sequential(nn.Linear(self.input_size, self.num_joints * 128, bias=False),
                                    nn.BatchNorm1d(self.num_joints * 128),
                                    nn.LeakyReLU(),
                                    nn.Dropout(p=0.1))
        self.mlp_head_x = nn.Linear(128, self.img_size_W)
        self.mlp_head_y = nn.Linear(128, self.img_size_H)

        self.fc1 = nn.Linear(self.input_size, 256)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 64)

    def forward(self, x, h0, c0,  B, L):
        x = self.ppointnet(x)
        x=x.view(B, L, 1024)
        g_vec, (hn, cn) = self.prnn(x, (h0, c0))

        # joint pos
        x = self.conv(g_vec.contiguous().view(-1, self.input_size))
        x = x.view(B, L, self.num_joints, -1)
        pred_x = self.mlp_head_x(x)
        pred_y = self.mlp_head_y(x)

        # global feature
        g_vec = self.faf1(self.fc1(g_vec))
        g_vec = self.fc2(g_vec)

        return g_vec, pred_x, pred_y, hn, cn



class CombinePoseModule(nn.Module):
    def __init__(self):
        super(CombinePoseModule, self).__init__()
        self.fc1 = nn.Linear(128, 128)
        self.faf1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 24*6+3+10)

    def forward(self, g_vec, a_vec, B, L):
        x=torch.cat((g_vec, a_vec), -1)
        x=self.fc1(x)
        x=self.faf1(x)
        x=self.fc2(x)

        q=x[:,:,:24*6].reshape(B*L*24, 6).contiguous()
        tmp_x=nn.functional.normalize(q[:,:3], dim=-1)
        tmp_z=nn.functional.normalize(torch.cross(tmp_x, q[:,3:], dim=-1), dim=-1)
        tmp_y=torch.cross(tmp_z, tmp_x, dim=-1)

        tmp_x=tmp_x.view(B, L, 24, 3, 1)
        tmp_y=tmp_y.view(B, L, 24, 3, 1)
        tmp_z=tmp_z.view(B, L, 24, 3, 1)
        q=torch.cat((tmp_x, tmp_y, tmp_z), -1)

        t=x[:,:,24*6  :24*6+3]
        b=x[:,:,24*6+3:24*6+3+10]
        return q, t, b

class SMPLModule(nn.Module):
    def __init__(self, device):
        super(SMPLModule, self).__init__()
        self.device=device
        self.blank_atom=torch.tensor([[1,0,0],[0,1,0],[0,0,1]], dtype=torch.float32, requires_grad=False, device=self.device)
        self.smpl_model_m=SMPL('m', device)
        # self.smpl_model_f=SMPL('f')

    def forward(self, q, t, b): #b: (10,)
        B=q.size()[0]
        L=q.size()[1]
        pose=q.view(B*L, 24, 3, 3)
        t=t.view(B*L, 3)
        b=b.view(B*L, 10)
        rotmat=pose[:,0,:,:]

        smpl_vertice=torch.zeros((B*L, 6890, 3), dtype=torch.float32, requires_grad=False, device=self.device)
        smpl_skeleton=torch.zeros((B*L, 24, 3), dtype=torch.float32, requires_grad=False, device=self.device)
        smpl_vertice, smpl_skeleton =self.smpl_model_m(b, pose, t)

        # smpl_vertice=torch.transpose(torch.bmm(rotmat, torch.transpose(smpl_vertice, 1, 2)), 1,2)+t
        # smpl_skeleton=torch.transpose(torch.bmm(rotmat, torch.transpose(smpl_skeleton, 1, 2)), 1,2)+t
        smpl_vertice=smpl_vertice.view(B, L, 6890, 3)
        smpl_skeleton=smpl_skeleton.view(B, L, 24, 3)
        return smpl_vertice, smpl_skeleton


class EPCPoseModel(nn.Module):
    def __init__(self, device, bidirectional=False, feat='all', polarity=1):
        super(EPCPoseModel, self).__init__()
        self.layer_num = 6 if bidirectional else 3
        self.module0=BasePointNet(polarity)
        self.module1=PosePointModule(bidirectional, polarity=polarity)
        self.module2=AnchorPoseModule(device, bidirectional, polarity)
        self.module3=CombinePoseModule()
        self.module4=SMPLModule(device)
        self.feat=feat
        self.polarity=polarity

    def forward(self, x, kpt_pred=None):
        dtype = x.dtype
        device = x.device
        B=x.size()[0]
        L=x.size()[1]
        pt_size=x.size()[2]
        in_feature_size=x.size()[3]
        out_feature_size=24+3+self.polarity  # 24+4

        x=x.view(B*L, pt_size, in_feature_size)
        x=self.module0(x)  # [40, 7500, 27]

        h0_g = torch.zeros((self.layer_num, B, 1024), dtype=dtype, device=device)
        c0_g = torch.zeros((self.layer_num, B, 1024), dtype=dtype, device=device)
        g_vec, pred_x, pred_y, hn_p, cn_p = self.module1(x, h0_g, c0_g, B, L)
        # g_vec:[8, 5, 1024], g_loc:[8, 5, 3], pred_x:[8, 5, 12, 256], pred_y:[8, 5, 12, 256]

        h0_a = torch.zeros((self.layer_num, B, 64), dtype=dtype, device=device)
        c0_a = torch.zeros((self.layer_num, B, 64), dtype=dtype, device=device)

        a_vec, anchor_weights, hn_a, cn_a = self.module2(x, pred_x, pred_y, h0_a, c0_a, B, L, out_feature_size, kpt_pred=kpt_pred)
        if self.feat=='all':
            q, t, b = self.module3(g_vec, a_vec, B, L)
        elif self.feat=='global':
            q, t, b = self.module3(g_vec, torch.zeros_like(a_vec), B, L)
        elif self.feat=='local':
            q, t, b = self.module3(torch.zeros_like(g_vec), a_vec, B, L)
        else:
            raise ValueError('feat should be all, global or local')

        v,s=self.module4(q,t,b)
        output = {
            'q': q,
            't': t,
            'v': v,
            's3': s,
            'pred_x': pred_x,
            'pred_y': pred_y,
            'l': None,
            'b': b,
            # 'anchor_weights': anchor_weights,
            # 'hn_g': hn_p,
            # 'cn_g': cn_p,
            # 'hn_a': hn_a,
            # 'cn_a': cn_a,
        }
        return output




# class GlobalPointModule(nn.Module):
#     def __init__(self, bidirectional, img_size_H=256, img_size_W=256, num_joints=13, ):
#         super(GlobalPointModule, self).__init__()
#         self.img_size_H = img_size_H
#         self.img_size_W = img_size_W
#         self.num_joints = num_joints
#         self.ppointnet=PosePointNet()
#         self.prnn=nn.LSTM(input_size=1024, hidden_size=1024, num_layers=3, batch_first=True, dropout=0.1, bidirectional=bidirectional)

#         self.input_size = 2048 if bidirectional else 1024
#         self.fc1 = nn.Linear(self.input_size, 256)
#         self.faf1 = nn.ReLU()
#         self.fc2 = nn.Linear(256, 64)

#     def forward(self, x, h0, c0,  B, L):
#         x = self.ppointnet(x)
#         x=x.view(B, L, 1024)
#         g_vec, (hn, cn) = self.prnn(x, (h0, c0))

#         # global feature
#         g_vec = self.faf1(self.fc1(g_vec))
#         g_vec = self.fc2(g_vec)

#         return g_vec, hn, cn


# class BoundingBoxModule(nn.Module):
#     def __init__(self, bidirectional, img_size_H=256, img_size_W=256):
#         super(BoundingBoxModule, self).__init__()
#         self.img_size_H = img_size_H
#         self.img_size_W = img_size_W
#         self.ppointnet=PosePointNet()
#         self.prnn=nn.LSTM(input_size=1024, hidden_size=1024, num_layers=3, batch_first=True, dropout=0.1, bidirectional=bidirectional)

#         self.input_size = 2048 if bidirectional else 1024
#         self.fc = nn.Sequential(nn.Linear(self.input_size, 256, bias=False),
#                                     nn.BatchNorm1d(256),
#                                     nn.LeakyReLU(),
#                                     nn.Dropout(p=0.1))
#         self.mlp_head_x = nn.Linear(256, self.img_size_W)
#         self.mlp_head_y = nn.Linear(256, self.img_size_H)

#         self.fc2 = nn.Sequential(nn.Linear(256, 64, bias=False),
#                                     nn.BatchNorm1d(64),
#                                     nn.LeakyReLU(),
#                                     nn.Dropout(p=0.1))
#         self.fc3 = nn.Linear(64, 2)

#         self.fc4 = nn.Linear(self.input_size, 256)
#         # self.fc4 = nn.Sequential(nn.Linear(self.input_size, 256, bias=False),
#         #                             nn.BatchNorm1d(256),
#         #                             nn.LeakyReLU(),
#         #                             nn.Dropout(p=0.1))
#         self.faf = nn.ReLU()
#         self.fc5 = nn.Linear(256, 64)

#     def forward(self, x, h0, c0,  B, L):
#         x = self.ppointnet(x)
#         x=x.view(B, L, 1024)
#         g_vec, (hn, cn) = self.prnn(x, (h0, c0))

#         # center pos
#         x = self.fc(g_vec.contiguous().view(-1, self.input_size))
#         pred_x = self.mlp_head_x(x)
#         pred_y = self.mlp_head_y(x)

#         # height and width
#         size = self.fc3(self.fc2(x))

#         # global feature
#         g_vec = self.faf(self.fc4(g_vec))
#         g_vec = self.fc5(g_vec)

#         return g_vec, pred_x, pred_y, size, hn, cn


# class KeypointModule(nn.Module):
#     def __init__(self, bidirectional, img_size_H=256, img_size_W=256, num_joints=13, ):
#         super(KeypointModule, self).__init__()
#         self.img_size_H = img_size_H
#         self.img_size_W = img_size_W
#         self.num_joints = num_joints
#         self.ppointnet=PosePointNet()
#         self.prnn=nn.LSTM(input_size=1024, hidden_size=1024, num_layers=3, batch_first=True, dropout=0.1, bidirectional=bidirectional)

#         self.input_size = 2048 if bidirectional else 1024
#         self.conv = nn.Sequential(nn.Linear(self.input_size, self.num_joints * 128, bias=False),
#                                     nn.BatchNorm1d(self.num_joints * 128),
#                                     nn.LeakyReLU(),
#                                     nn.Dropout(p=0.1))
#         self.mlp_head_x = nn.Linear(128, self.img_size_W)
#         self.mlp_head_y = nn.Linear(128, self.img_size_H)

#         self.fc1 = nn.Linear(self.input_size, 256)
#         self.faf1 = nn.ReLU()
#         self.fc2 = nn.Linear(256, 64)

#     def forward(self, x, h0, c0,  B, L):
#         x = self.ppointnet(x)
#         x=x.view(B, L, 1024)
#         g_vec, (hn, cn) = self.prnn(x, (h0, c0))

#         # joint pos
#         x = self.conv(g_vec.contiguous().view(-1, self.input_size))
#         x = x.view(B, L, self.num_joints, -1)
#         pred_x = self.mlp_head_x(x)
#         pred_y = self.mlp_head_y(x)

#         # global feature
#         g_vec = self.faf1(self.fc1(g_vec))
#         g_vec = self.fc2(g_vec)

#         return g_vec, pred_x, pred_y, hn, cn

# class KeypointModel(nn.Module):
#     def __init__(self, device, bidirectional=False):
#         super(KeypointModel, self).__init__()
#         self.layer_num = 6 if bidirectional else 3
#         self.module0=BasePointNet()
#         self.module1=KeypointModule(bidirectional)

#     def forward(self, x):
#         dtype = x.dtype
#         device = x.device
#         B=x.size()[0]
#         L=x.size()[1]
#         pt_size=x.size()[2]
#         in_feature_size=x.size()[3]
#         out_feature_size=24+3  # 24+4

#         x=x.view(B*L, pt_size, in_feature_size)
#         x=self.module0(x)  # [40, 7500, 27]

#         h0_g = torch.zeros((self.layer_num, B, 1024), dtype=dtype, device=device)
#         c0_g = torch.zeros((self.layer_num, B, 1024), dtype=dtype, device=device)
#         g_vec, pred_x, pred_y, hn_p, cn_p = self.module1(x, h0_g, c0_g, B, L)

#         output = {
#             'g_vec': g_vec,
#             'pred_x': pred_x,
#             'pred_y': pred_y,
#         }
#         return output


# class BoundingBoxModel(nn.Module):
#     def __init__(self, device, bidirectional=False):
#         super(BoundingBoxModel, self).__init__()
#         self.layer_num = 6 if bidirectional else 3
#         self.module0=BasePointNet()
#         self.module1=BoundingBoxModule(bidirectional)

#     def forward(self, x):
#         dtype = x.dtype
#         device = x.device
#         B=x.size()[0]
#         L=x.size()[1]
#         pt_size=x.size()[2]
#         in_feature_size=x.size()[3]
#         out_feature_size=24+3  # 24+4

#         x=x.view(B*L, pt_size, in_feature_size)
#         x=self.module0(x)  # [40, 7500, 27]

#         h0_g = torch.zeros((self.layer_num, B, 1024), dtype=dtype, device=device)
#         c0_g = torch.zeros((self.layer_num, B, 1024), dtype=dtype, device=device)
#         g_vec, cx, cy, size, hn_p, cn_p = self.module1(x, h0_g, c0_g, B, L)

#         output = {
#             'cx': cx.view(8, 5, 1, 256),
#             'cy': cy.view(8, 5, 1, 256),
#             'size': size.view(8, 5, 2)
#         }
#         return output



# if __name__=='__main__':
#     print('AnchorInit:')
#     templates=AnchorInit()
#     print('\tOutput:', templates.shape) #(9,3,3,3)
#     #print(templates)

#     print('AnchorGrouping:')
#     templates=templates.view(1, 9*3*3, 3)
#     #z=torch.zeros((1,1,3))
#     print('\tInput:', templates.shape, 'nsample:', 7, templates.shape, templates.shape)
#     points=AnchorGrouping(templates, 7, templates, templates)
#     print('\tOutput:', points.shape)
#     #print(points)

#     print('AnchorPointNet:')
#     data=torch.rand((7*13, 50, 24+4+3), dtype=torch.float32, device='cpu')
#     print('\tInput:', data.shape)
#     model=AnchorPointNet()
#     x,w=model(data)
#     print('\tOutput:', x.shape, w.shape)

#     print('AnchorVoxelNet:')
#     data=torch.rand((7*13, 9, 3, 3, 64), dtype=torch.float32, device='cpu')
#     print('\tInput:', data.shape)
#     model=AnchorVoxelNet()
#     x=model(data)
#     print('\tOutput:', x.shape)

#     print('AnchorRNN:')
#     data=torch.rand((7, 13, 64), dtype=torch.float32, device='cpu')
#     h0=torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
#     c0=torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
#     print('\tInput:', data.shape, h0.shape, c0.shape)
#     model=AnchorRNN()
#     a, hn, cn=model(data, h0, c0)
#     print('\tOutput:', a.shape,hn.shape, cn.shape)

#     print('AnchorModule:')
#     data=torch.rand((7* 13, 50, 24+4), dtype=torch.float32, device='cpu')
#     g_loc=torch.full((7, 13, 2), 100.0, dtype=torch.float32, device='cpu')
#     h0=torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
#     c0=torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
#     print('\tInput:', data.shape, g_loc.shape, h0.shape, c0.shape)
#     model=AnchorModule()
#     a,w,hn,cn=model(data, g_loc, h0, c0, 7, 13, 24+4)
#     print('\tOutput:', a.shape, w.shape, hn.shape, cn.shape)

#     print('BasePointNet:')
#     data=torch.rand((7*13, 50, 6), dtype=torch.float32, device='cpu')
#     print('\tInput:', data.shape)
#     model=BasePointNet()
#     x=model(data)
#     print('\tOutput:', x.shape)

#     print('GlobalPointNet:')
#     data=torch.rand((7*13, 50, 24+4), dtype=torch.float32, device='cpu')
#     print('\tInput:', data.shape)
#     model=GlobalPointNet()
#     x,w=model(data)
#     print('\tOutput:', x.shape, w.shape)

#     print('GlobalRNN:')
#     data=torch.rand((7, 13, 64), dtype=torch.float32, device='cpu')
#     h0=torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
#     c0=torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
#     print('\tInput:', data.shape, h0.shape, c0.shape)
#     model=GlobalRNN()
#     g, l, hn, cn=model(data, h0, c0)
#     print('\tOutput:', g.shape, l.shape, hn.shape, cn.shape)

#     print('GlobalModule:')
#     data=torch.rand((7*13, 50, 24+4), dtype=torch.float32, device='cpu')
#     h0=torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
#     c0=torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
#     print('\tInput:', data.shape, h0.shape, c0.shape)
#     model=GlobalModule()
#     x,l,w,hn,cn=model(data,h0,c0,7,13)
#     print('\tOutput:', x.shape, l.shape, w.shape, hn.shape, cn.shape)

#     print('CombineModule:')
#     g=torch.rand((7, 13, 64), dtype=torch.float32, device='cpu')
#     a=torch.rand((7, 13, 64), dtype=torch.float32, device='cpu')
#     print('\tInput:', g.shape, a.shape)
#     model=CombineModule()
#     q,t,b,g=model(g, a, 7, 13)
#     print('\tOutput:', q.shape, t.shape, b.shape, g.shape)

#     print('SMPLModule:')
#     q=torch.rand((7, 13, 9, 3, 3), dtype=torch.float32, device='cpu')
#     t=torch.rand((7, 13, 3), dtype=torch.float32, device='cpu')
#     b=torch.rand((7, 13, 10), dtype=torch.float32, device='cpu')
#     gm=torch.ones((4, 13,1), dtype=torch.float32, device='cpu')
#     gf=torch.zeros((3, 13,1), dtype=torch.float32, device='cpu')
#     g=torch.cat((gm,gf), 0)
#     print('\tInput:', q.shape, t.shape, b.shape, g.shape)
#     model=SMPLModule()
#     v,s=model(q,t,b,g)
#     print('\tOutput:', v.shape, s.shape)

#     print('WHOLE:')
#     data=torch.rand((7, 13, 50, 6), dtype=torch.float32, device='cpu')
#     h0=torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
#     c0=torch.zeros((3, 7, 64), dtype=torch.float32, device='cpu')
#     print('\tInput:', data.shape, g.shape, h0.shape, c0.shape)
#     model=mmWaveModel()
#     q, t, v, s, l, b, g, _, _, _, _, _, _=model(data, g, h0, c0, h0, c0)
#     print('\tOutput:', q.shape, t.shape, v.shape, s.shape, l.shape, b.shape, g.shape)
#     q, t, v, s, l, b, g, _, _, _, _, _, _=model(data, None, h0, c0, h0, c0)
#     print('\tOutput:', q.shape, t.shape, v.shape, s.shape, l.shape, b.shape, g.shape)
