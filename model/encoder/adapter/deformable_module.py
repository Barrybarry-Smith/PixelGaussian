# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Tuple, NamedTuple
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.models import resnet50, resnet18
from torch.cuda.amp.autocast_mode import autocast
from src.ops import DeformableAggregationFunction as DAF
from .utils import get_rotation_matrix, safe_sigmoid
from .utils import cartesian, inv_cartesian


class SparseGaussian3DKeyPointsGenerator(nn.Module):
    def __init__(
        self,
        embed_dims=128,
        num_learnable_pts=6,
        fix_scale=None,
        phi_activation='sigmoid',
        xyz_coordinate='xyz'
    ):
        super(SparseGaussian3DKeyPointsGenerator, self).__init__()
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        self.fix_scale = np.array(fix_scale)
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(self.embed_dims, num_learnable_pts * 3)

        self.phi_activation = phi_activation
        self.xyz_coordinate = xyz_coordinate

    def init_weight(self):
        if self.num_learnable_pts > 0:
            nn.init.xavier_uniform_(self.learnable_fc.weight)
            if self.learnable_fc.bias is not None:
                nn.init.constant_(self.learnable_fc.bias, 0.0)

    def update_pc_range(self, pts3d):
   
        self.pc_range = [
            torch.min(pts3d[:, 0]), torch.min(pts3d[:, 1]), torch.min(pts3d[:, 2]),
            torch.max(pts3d[:, 0]), torch.max(pts3d[:, 0]), torch.max(pts3d[:, 0])
        ]

    def forward(
        self,
        pts3d,
        anchor,
        instance_feature=None
    ):
        pts3d = pts3d.reshape(-1, 3)
        self.update_pc_range(pts3d)
        bs, num_anchor = anchor.shape[:2]

        # generate learnable scale for DAF
        fix_scale = anchor.new_tensor(self.fix_scale)
        scale = fix_scale[None, None].tile([bs, num_anchor, 1, 1]) # (B, N, K, 3)

        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = (
                safe_sigmoid(self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3))
                - 0.5
            )
            scale = torch.cat([scale, learnable_scale], dim=-2) # (B, N, K+K', 3)
        
        gs_scales = anchor[..., None, 3:6]

        key_points = scale * gs_scales # (B, N, K+K', 3)
        rots = anchor[..., 6:10]
        rotation_mat = get_rotation_matrix(rots).transpose(-1, -2)
        
        key_points = torch.matmul(
            rotation_mat[:, :, None], key_points[..., None]
        ).squeeze(-1) # (B, N, K+K', 3)

        xyz = anchor[..., :3]
        key_points = key_points + xyz.unsqueeze(2)

        return key_points


class DeformableFeatureAggregation(nn.Module):
    def __init__(
        self,
        embed_dims: int = 128,
        num_groups: int = 2,
        num_levels: int = 2,
        max_num_view: int = 12,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        use_deformable_func=True,
        residual_mode="add"
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(f"embed_dims must be divisible by num_groups, but got {embed_dims} and {num_groups}")
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.max_num_view = max_num_view
        self.use_deformable_func = use_deformable_func and DAF is not None
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)
        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = SparseGaussian3DKeyPointsGenerator(**kps_generator)
        self.num_pts = self.kps_generator.num_pts
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.weights_fc = nn.Linear(self.embed_dims, self.num_groups * self.max_num_view * self.num_levels * self.num_pts)

    def init_weight(self):
        nn.init.constant_(self.weights_fc.weight, 0.0)
        if self.weights_fc.bias is not None:
            nn.init.constant_(self.weights_fc.bias, 0.0)

        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        pts3d: torch.Tensor,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        anchor_embed=None,
        anchor_encoder=None,
    ):
        
        # generate key points
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(pts3d, anchor, instance_feature)

        view = metas["projection_mat"].shape[1]
        weights = self._get_weights(instance_feature, view, anchor_embed)

        weights = (
            weights.permute(0, 1, 4, 2, 3, 5)
            .contiguous()
            .reshape(
                bs,
                num_anchor * self.num_pts,
                view,
                self.num_levels,
                self.num_groups,
            )
        )

        points_2d = (
            self.project_points(
                key_points, 
                metas["projection_mat"],
                metas.get("image_wh")
            )
            .permute(0, 2, 3, 1, 4)
            .reshape(bs, num_anchor * self.num_pts, view, 2)
        )

        features = DAF.apply(
            *feature_maps, points_2d, weights
        ).reshape(bs, num_anchor, self.num_pts, self.embed_dims)

        features = features.sum(dim=2)
        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)

        return output

    def _get_weights(self, instance_feature, view, anchor_embed=None):
        bs, num_anchor = instance_feature.shape[:2]
        assert self.max_num_view % view == 0

        if anchor_embed is not None:
            feature = instance_feature + anchor_embed
        else:
            feature = instance_feature

        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                view,
                self.max_num_view // view,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        ).mean(dim=3)

        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, view, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):

        bs, num_anchor, num_pts = key_points.shape[:3]

        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )

        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(-1)

        points_2d = points_2d[..., :2] / torch.clamp(
            points_2d[..., 2:3], min=1e-5
        )
        
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
            points_2d = torch.clamp(points_2d, min=0.0, max=0.9999)

        return points_2d
