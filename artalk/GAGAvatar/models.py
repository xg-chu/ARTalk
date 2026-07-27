#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import copy
import math
import torch
import torchvision
import torch.nn as nn
from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.renderer.implicit.harmonic_embedding import HarmonicEmbedding

from .modules import DINOBase, StyleUNet
from .utils_renderer import render_gaussian

class GAGAvatar(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.base_model = DINOBase(output_dim=256)
        for param in self.base_model.dino_model.parameters():
            param.requires_grad = False
        # dir_encoder
        n_harmonic_dir = 4
        self.direnc_dim = n_harmonic_dir * 2 * 3 + 3
        self.harmo_encoder = HarmonicEmbedding(n_harmonic_dir)
        # pre_trained
        self.head_base = nn.Parameter(torch.randn(5023, 256), requires_grad=True)
        self.gs_generator_g = LinearGSGenerator(in_dim=1024, dir_dim=self.direnc_dim)
        self.gs_generator_l0 = ConvGSGenerator(in_dim=256, dir_dim=self.direnc_dim)
        self.gs_generator_l1 = ConvGSGenerator(in_dim=256, dir_dim=self.direnc_dim)
        self.cam_params = {'focal_x': 12.0, 'focal_y': 12.0, 'size': [512, 512]}
        self.upsampler = StyleUNet(in_size=512, in_dim=32, out_dim=3, out_size=512)

        _abs_path = os.path.dirname(os.path.abspath(__file__))
        _model_path = os.path.join(_abs_path, '../../assets/GAGAvatar/GAGAvatar.pt')
        _tracked_id_path = os.path.join(_abs_path, '../../assets/GAGAvatar/tracked.pt')
        _water_mark_path = os.path.join(_abs_path, '../../assets/GAGAvatar/gagavatar_logo.png')
        assert os.path.exists(_model_path), f"Model not found: {_model_path}."
        assert os.path.exists(_tracked_id_path), f"Tracked id not found: {_tracked_id_path}."
        ckpt = torch.load(_model_path, map_location='cpu', weights_only=True)['model']
        ckpt = {k:v for k, v in ckpt.items() if 'percep_loss' not in k}
        self.load_state_dict(ckpt)
        self.eval()
        self.all_gagavatar_id = torch.load(_tracked_id_path, map_location='cpu', weights_only=False)
        _water_mark_size = (82, 256)
        _water_mark = torchvision.io.read_image(_water_mark_path, mode=torchvision.io.ImageReadMode.RGB_ALPHA).float()/255.0
        self._water_mark = torchvision.transforms.functional.resize(_water_mark, _water_mark_size, antialias=True)

    def set_avatar_id(self, avatar_id):
        tracked_id = copy.deepcopy(self.all_gagavatar_id[avatar_id])
        for key in tracked_id.keys():
            if not isinstance(tracked_id[key], torch.Tensor):
                tracked_id[key] = torch.tensor(tracked_id[key]).float()
        self._tracked_id = tracked_id
        self._tracked_name = avatar_id
        if hasattr(self, '_gs_params'):
            del self._gs_params
        if hasattr(self, 'feature_batch'):
            del self.feature_batch
        if hasattr(self, 'shapecode'):
            del self.shapecode

    @torch.no_grad()
    def forward_expression(self, batch):
        if not hasattr(self, '_gs_params'):
            batch_size = batch['f_image'].shape[0]
            f_image, f_planes = batch['f_image'], batch['f_planes']
            f_feature0, f_feature1 = self.base_model(f_image)
            # dir encoding
            plane_direnc = self.harmo_encoder(f_planes['plane_dirs'])
            # global part
            gs_params_g = self.gs_generator_g(
                torch.cat([
                        self.head_base[None].expand(batch_size, -1, -1), f_feature1[:, None].expand(-1, 5023, -1), 
                    ], dim=-1
                ), plane_direnc
            )
            gs_params_g['xyz'] = batch['f_image'].new_zeros((batch_size, 5023, 3))
            # local part
            gs_params_l0 = self.gs_generator_l0(f_feature0, plane_direnc)
            gs_params_l1 = self.gs_generator_l1(f_feature0, plane_direnc)
            gs_params_l0['xyz'] = f_planes['plane_points'] + gs_params_l0['positions'] * f_planes['plane_dirs'][:, None]
            gs_params_l1['xyz'] = f_planes['plane_points'] + -1 * gs_params_l1['positions'] * f_planes['plane_dirs'][:, None]
            gs_params = {
                k:torch.cat([gs_params_g[k], gs_params_l0[k], gs_params_l1[k]], dim=1) for k in gs_params_g.keys()
            }
            self._gs_params = gs_params
        gs_params = self._gs_params
        t_image, t_points, t_transform = batch['t_image'], batch['t_points'], batch['t_transform']
        gs_params['xyz'][:, :5023] = t_points
        gen_images = render_gaussian(
            gs_params=gs_params, cam_matrix=t_transform, cam_params=self.cam_params
        )['images']
        sr_gen_images = self.upsampler(gen_images)
        return self.add_water_mark(sr_gen_images.clamp(0, 1))

    @torch.no_grad()
    def build_forward_batch(self, motion_code, flame_model):
        if not hasattr(self, '_tracked_id'):
            self.set_avatar_id('11.jpg')
        device = motion_code.device
        if not hasattr(self, 'feature_batch'):
            feature_batch = {}
            feature_batch['f_image'] = torchvision.transforms.functional.resize(self._tracked_id['image'], (518, 518), antialias=True)[None].to(device)
            feature_batch['f_planes'] = build_points_planes(296, self._tracked_id['transform_matrix'])
            feature_batch['f_planes']['plane_points'] = feature_batch['f_planes']['plane_points'][None].to(device)
            feature_batch['f_planes']['plane_dirs'] = feature_batch['f_planes']['plane_dirs'][None].to(device)
            feature_batch['t_image'] = torchvision.transforms.functional.resize(self._tracked_id['image'], (512, 512), antialias=True)[None].to(device)
            feature_batch['t_transform'] = self._tracked_id['transform_matrix'][None].to(device)
            self.feature_batch = feature_batch
            self.shapecode = self._tracked_id['shapecode'][None].to(device)

        feature_batch = copy.deepcopy(self.feature_batch)
        exp_code = motion_code[:, :100]
        pose_code = torch.cat([motion_code.new_zeros(1, 3), motion_code[:, 103:]], dim=-1)
        t_points = flame_model(
            shape_params=self.shapecode, pose_params=pose_code, 
            expression_params=exp_code, eye_pose_params=pose_code.new_zeros(1, 6)
        ).float()
        if not hasattr(self, 'upper_points'):
            self.upper_points = t_points[:, forehead_indices]
        else:
            current_points = t_points[:, forehead_indices]
            self.upper_points = 0.98 * self.upper_points + 0.02 * current_points
            t_points[:, forehead_indices] = self.upper_points
        feature_batch['t_points'] = t_points
        feature_batch['t_transform'][:, :3, :3] = transform_emoca_to_p3d(motion_code[:, 100:103])[:, :3, :3]
        return feature_batch

    @torch.no_grad()
    def add_water_mark(self, image):
        water_mark = self._water_mark.clone().to(image.device)
        _water_mark_rgb = water_mark[None, :3]
        _water_mark_alpha = water_mark[None, 3:4].expand(-1, 3, -1, -1) * 0.8
        _mark_patch = image[..., -water_mark.shape[-2]:, -water_mark.shape[-1]:]
        _mark_patch = _mark_patch * (1-_water_mark_alpha) + _water_mark_rgb * _water_mark_alpha
        image[..., -water_mark.shape[-2]:, -water_mark.shape[-1]:] = _mark_patch
        return image


class LinearGSGenerator(nn.Module):
    def __init__(self, in_dim=1024, dir_dim=27, **kwargs):
        super().__init__()
        # params
        self.feature_layers = nn.Sequential(
            nn.Linear(in_dim, in_dim//4, bias=True),
            nn.ReLU(),
            nn.Linear(in_dim//4, in_dim//4, bias=True),
            nn.ReLU(),
            nn.Linear(in_dim//4, in_dim//4, bias=True),
            nn.ReLU(),
            nn.Linear(in_dim//4, in_dim//4, bias=True),
        )
        layer_in_dim = in_dim//4 + dir_dim
        self.color_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 32, bias=True),
        )
        self.opacity_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 1, bias=True),
        )
        self.scale_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 3, bias=True)
        )
        self.rotation_layers = nn.Sequential(
            nn.Linear(layer_in_dim, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 4, bias=True),
        )

    def forward(self, input_features, plane_direnc):
        input_features = self.feature_layers(input_features)
        plane_direnc = plane_direnc[:, None].expand(-1, input_features.shape[1], -1)
        input_features = torch.cat([input_features, plane_direnc], dim=-1)
        # color
        colors = self.color_layers(input_features)
        colors[..., :3] = torch.sigmoid(colors[..., :3])
        # opacity
        opacities = self.opacity_layers(input_features)
        opacities = torch.sigmoid(opacities)
        # scale
        scales = self.scale_layers(input_features)
        # scales = torch.exp(scales) * 0.01
        scales = torch.sigmoid(scales) * 0.05
        # rotation
        rotations = self.rotation_layers(input_features)
        rotations = nn.functional.normalize(rotations)
        return {'colors':colors, 'opacities':opacities, 'scales':scales, 'rotations':rotations}


class ConvGSGenerator(nn.Module):
    def __init__(self, in_dim=256, dir_dim=27, **kwargs):
        super().__init__()
        out_dim = 32 + 1 + 3 + 4 + 1 # color + opacity + scale + rotation + position
        self.gaussian_conv = nn.Sequential(
            nn.Conv2d(in_dim+dir_dim, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_dim//2, in_dim//2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_dim//2, out_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, input_features, plane_direnc):
        plane_direnc = plane_direnc[:, :, None, None].expand(-1, -1, input_features.shape[2], input_features.shape[3])
        input_features = torch.cat([input_features, plane_direnc], dim=1)
        gaussian_params = self.gaussian_conv(input_features)
        # color
        colors = gaussian_params[:, :32]
        colors[..., :3] = torch.sigmoid(colors[..., :3])
        # opacity
        opacities = gaussian_params[:, 32:33]
        opacities = torch.sigmoid(opacities)
        # scale
        scales = gaussian_params[:, 33:36]
        # scales = torch.exp(scales) * 0.01
        scales = torch.sigmoid(scales) * 0.05
        # rotation
        rotations = gaussian_params[:, 36:40]
        rotations = nn.functional.normalize(rotations)
        # position
        positions = gaussian_params[:, 40:41]
        positions = torch.sigmoid(positions)
        results = {'colors':colors, 'opacities':opacities, 'scales':scales, 'rotations':rotations, 'positions':positions}
        for key in results.keys():
            results[key] = results[key].permute(0, 2, 3, 1).reshape(results[key].shape[0], -1, results[key].shape[1])
        return results


def build_points_planes(plane_size, transforms):
    x, y = torch.meshgrid(
        torch.linspace(1, -1, plane_size, dtype=torch.float32), 
        torch.linspace(1, -1, plane_size, dtype=torch.float32), 
        indexing="xy",
    )
    R = transforms[:3, :3]; T = transforms[:3, 3:]
    cam_dirs = torch.tensor([[0., 0., 1.]], dtype=torch.float32)
    ray_dirs = torch.nn.functional.pad(
        torch.stack([x/12.0, y/12.0], dim=-1), (0, 1), value=1.0
    )
    cam_dirs = torch.matmul(R, cam_dirs.reshape(-1, 3)[:, :, None])[..., 0]
    ray_dirs = torch.matmul(R, ray_dirs.reshape(-1, 3)[:, :, None])[..., 0]
    origins = (-torch.matmul(R, T)[..., 0]).broadcast_to(ray_dirs.shape).squeeze()
    distance = ((origins[0] * cam_dirs[0]).sum()).abs()
    plane_points = origins + distance * ray_dirs
    return {'plane_points': plane_points, 'plane_dirs': cam_dirs[0]}


def transform_emoca_to_p3d(emoca_base_rotation):
    device = emoca_base_rotation.device
    batch_size = emoca_base_rotation.shape[0]
    initial_trans = torch.tensor([[0, 0, 5000.0/512]]).to(device)
    emoca_base_rotation[:, [0, 2]] *= -1
    emoca_base_matrix = axis_angle_to_matrix(emoca_base_rotation)
    emoca_base_matrix = torch.matmul(emoca_base_matrix, torch.tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).to(device).float())
    emoca_base_matrix = emoca_base_matrix.inverse()
    base_transform_p3d = torch.cat([emoca_base_matrix, initial_trans.reshape(1, -1, 1).repeat(batch_size, 1, 1)], dim=-1)
    return base_transform_p3d


def transform_inv(transforms):
    if transforms.dim() == 3:
        return torch.stack([transform_opencv_to_p3d(t) for t in transforms])
    if transforms.shape[-1] != transforms.shape[-2]:
        new_transform = torch.eye(4)
        new_transform[:3, :] = transforms
        transforms = new_transform
    transforms = torch.linalg.inv(transforms)
    return transforms[:3]


def batch_rodrigues(rot_vecs,):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''
    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype
    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)
    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_opencv_to_p3d(opencv_transform, verts_scale=1, type='w2c'):
    assert type in ['w2c', 'c2w']
    if opencv_transform.dim() == 3:
        return torch.stack([transform_opencv_to_p3d(t, verts_scale, type) for t in opencv_transform])
    if type == 'c2w':
        if opencv_transform.shape[-1] != opencv_transform.shape[-2]:
            new_transform = torch.eye(4).to(opencv_transform.device)
            new_transform[:3, :] = opencv_transform
            opencv_transform = new_transform
        opencv_transform = torch.linalg.inv(opencv_transform) # c2w to w2c
    rotation = opencv_transform[:3, :3]
    rotation = rotation.permute(1, 0)
    rotation[:, :2] *= -1
    if opencv_transform.shape[-1] == 4:
        translation = opencv_transform[:3, 3] * verts_scale
        translation[:2] *= -1
        rotation = torch.cat([rotation, translation.reshape(-1, 1)], dim=-1)
    return rotation


forehead_indices = [
    2168, 2165, 3068, 2199, 2196, 3720, 2091, 2088, 3524, 625, 628, 3871, 705, 708, 2030, 667, 670,
    3708, 3706, 3729, 3721, 3773, 3789, 3735, 3732, 3786, 3876, 3878, 3913, 3899, 3872, 3874, 3864, 3865,
    3158, 3157, 336, 335, 3153, 3705, 2177, 2176, 3540, 671, 672, 3863, 2134, 16, 17, 2138, 2139,
    2567, 2566, 337, 338, 3154, 3712, 2178, 2179, 3495, 674, 673, 3868, 2135, 27, 18, 1429, 1430
]
