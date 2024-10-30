import random, torch
import torch.nn as nn
from src.ops import DeformableAggregationFunction as DAF
from .deformable_module import DeformableFeatureAggregation
from .utils import SparseGaussian3DEncoder
from .refine import SparseGaussian3DRefinementModule

class IterativeGaussianRefiner(nn.Module):

    def __init__(self,
                 stages,
                 num_groups,
                 num_levels,
                 attn_drop,
                 num_learnable_pts,
                 max_num_view,
                 fix_scale,
                 num_anchors,
                 embed_dim):

        super(IterativeGaussianRefiner, self).__init__()

        self.stages = stages
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.attn_drop = attn_drop
        self.num_learnable_pts = num_learnable_pts
        self.fix_scale = fix_scale
        self.num_anchors = num_anchors
        self.embed_dim = embed_dim
        self.gaussian_encoder = SparseGaussian3DEncoder(embed_dims=self.embed_dim)

        self.deformable = DeformableFeatureAggregation(
            embed_dims=embed_dim,
            num_groups=num_groups,
            num_levels=num_levels,
            max_num_view=max_num_view,
            attn_drop=attn_drop,
            kps_generator=dict(
                num_learnable_pts=num_learnable_pts,
                fix_scale=fix_scale
            )
        )

        self.refiner = SparseGaussian3DRefinementModule(embed_dims=embed_dim)

    def get_deformable_metas(self, 
                             extrinsics,
                             intrinsics, 
                             img_size,
                             device):

        nbr = extrinsics.shape[0]
        w, h = img_size
        project_mats = []
        for view in range(nbr):
            w2c = extrinsics[view].inverse()
            intri = intrinsics[view].clone()
            intri[0, 0] *= w
            intri[0, 2] *= w
            intri[1, 1] *= w
            intri[1, 2] *= w

            K = torch.eye(4, 4, dtype=w2c.dtype).to(device)
            K[:3, :3] = intri
            project_mats.append(torch.matmul(K, w2c))

        return dict(
            projection_mat=torch.stack(project_mats, dim=0).unsqueeze(0),
            image_wh=torch.tensor([w, h], dtype=w2c.dtype, device=device).repeat(nbr, 1).unsqueeze(0)
        )
           

    def forward(self, 
                adaptive_gaussians,
                feature_maps,
                alphas,
                extrinsics,
                intrinsics,
                image_size):

        b, v, _, _ = extrinsics.shape
        device = extrinsics.device
        assert b == len(adaptive_gaussians)
        feature_maps = DAF.feature_maps_format([feature_maps])
        refined_gaussians_list = []

        for batch in range(b):
            gaussians = adaptive_gaussians[batch]
            N = gaussians.means.shape[0]
            anchor_idxs = random.sample(range(N), self.num_anchors)

            means = gaussians.means[anchor_idxs]
            scales = gaussians.scales[anchor_idxs]
            rotations = gaussians.rotations[anchor_idxs]
            opacities = gaussians.opacities[anchor_idxs].unsqueeze(-1)
            
            anchors = torch.cat([means, scales, rotations, opacities], dim=-1).unsqueeze(0)
            anchor_embed = self.gaussian_encoder(anchors)
            instance_feature = nn.Parameter(torch.zeros_like(anchor_embed), requires_grad=True)

            metas = self.get_deformable_metas(
                extrinsics[batch],
                intrinsics[batch],
                image_size,
                device
            )

            for stage in range(self.stages):
                instance_feature = self.deformable(
                    pts3d=means,
                    instance_feature=instance_feature,
                    anchor=anchors,
                    anchor_embed=anchor_embed,
                    feature_maps=feature_maps,
                    metas=metas,
                    anchor_encoder=self.gaussian_encoder
                )

            refined_gaussians = self.refiner(
                instance_feature=instance_feature,
                anchor=anchors,
                anchor_embed=anchor_embed,
                gaussians=gaussians,
                anchor_idxs=anchor_idxs
            )
            refined_gaussians_list.append(refined_gaussians)
            
        return refined_gaussians_list
                

