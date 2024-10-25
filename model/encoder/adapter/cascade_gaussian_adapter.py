import random, copy
import torch
import torch.nn as nn
from src.geometry.projection import project
from src.model.types import Gaussian
from src.ops import DeformableAggregationFunction as DAF
from .deformable_module import DeformableFeatureAggregation
from ..common.gaussians import build_covariance

class ContextHyper(nn.Module):
    
    def __init__(self,
                 score_embed,
                 num_groups,
                 num_levels,
                 attn_drop,
                 num_learnable_pts,
                 fix_scale,
                 num_anchors,
                 low_ratio=0.05,
                 high_ratio=0.95
                ):
        
        super(ContextHyper, self).__init__()

        self.num_anchors = num_anchors
        self.low_ratio = low_ratio
        self.high_ratio = high_ratio
        self.deformable = DeformableFeatureAggregation(
            embed_dims=score_embed,
            num_groups=num_groups,
            num_levels=num_levels,
            attn_drop=attn_drop,
            kps_generator=dict(
                num_learnable_pts=num_learnable_pts,
                fix_scale=fix_scale
            )
        )

        self.score_encoder = nn.Sequential(
            nn.Linear(1, score_embed),
            nn.ReLU(inplace=True),
            nn.LayerNorm(score_embed)
        )
    

    def forward(self,
                gaussians,
                gaussian_scores,
                score_maps,
                metas):

        N = gaussians.means.shape[0]
        anchor_idxs = random.sample(range(N), self.num_anchors)

        means = gaussians.means[anchor_idxs]
        scales = gaussians.scales[anchor_idxs]
        rotations = gaussians.rotations[anchor_idxs]
        opacities = gaussians.opacities[anchor_idxs].unsqueeze(-1)
        anchors = torch.cat([means, scales, rotations, opacities], dim=-1)

        scores = gaussian_scores[anchor_idxs].reshape(self.num_anchors, 1)
        score_instances = self.score_encoder(scores)
        min_scores, max_scores = (torch.min(gaussian_scores), torch.max(gaussian_scores))

        v, h, w = score_maps.shape
        score_maps = score_maps.reshape(v * h * w, -1)
        score_embeds = self.score_encoder(score_maps).reshape(1, v, -1, h, w)
        score_embeds = DAF.feature_maps_format([score_embeds])

        score_instances = self.deformable(
            pts3d=means,
            instance_feature=score_instances.unsqueeze(0),
            anchor=anchors.unsqueeze(0),
            anchor_embed=None,
            feature_maps=score_embeds,
            metas=metas
        ).mean(dim=-1).squeeze()

        score_instances_min, score_instances_max = (torch.min(score_instances), torch.max(score_instances))
        score_instances = (score_instances - score_instances_min) / (score_instances_max - score_instances_min)
        scores = min_scores + score_instances * (max_scores - min_scores)

        tao_low = torch.quantile(scores, self.low_ratio)
        tao_high = torch.quantile(scores, self.high_ratio)
        
        return tao_low, tao_high


class CascadeGaussianAdapter(nn.Module):

    def __init__(self, 
                 stages,
                 opacity_thres,
                 split_count,
                 scaling_factor,
                 opacity_factor,
                 num_groups,
                 num_levels,
                 attn_drop,
                 num_learnable_pts,
                 fix_scale,
                 num_anchors,
                 score_embed):

        super(CascadeGaussianAdapter, self).__init__()
        
        self.stages = stages
        self.opacity_thres = opacity_thres
        self.split_count = split_count
        self.scaling_factor = scaling_factor
        self.opacity_factor = opacity_factor
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.attn_drop = attn_drop
        self.num_learnable_pts = num_learnable_pts
        self.fix_scale = fix_scale
        self.num_anchors = num_anchors
        self.score_embed = score_embed

        self.hypers = []
        for stage in range(self.stages):
            self.hypers.append(
                ContextHyper(
                    score_embed=self.score_embed,
                    num_groups=self.num_groups,
                    num_levels=self.num_levels,
                    attn_drop=self.attn_drop,
                    num_learnable_pts=self.num_learnable_pts,
                    fix_scale=self.fix_scale,
                    num_anchors=self.num_anchors
                ).cuda()
            )

    def remove_redundancy(self,
                          gaussians,
                          extrinsics,
                          intrinsics,
                          image_size,
                          distance_thres=0.2):

        h, w = image_size
        b, v, _, _ = extrinsics.shape
        gaussian_list = []
        
        for batch in range(b):
            for view in range(v):

                view_means = gaussians.means[batch, view].squeeze()
                view_covariances = gaussians.covariances[batch, view].squeeze()
                view_harmonics = gaussians.harmonics[batch, view].squeeze()
                view_opacities = gaussians.opacities[batch, view].squeeze()
                view_scales = gaussians.scales[batch, view].squeeze()
                view_rotations = gaussians.rotations[batch, view].squeeze()

                if view == 0:
                    gaussian_means = view_means
                    gaussian_covariances = view_covariances
                    gaussian_harmonics = view_harmonics
                    gaussian_opacities = view_opacities
                    gaussian_scales = view_scales
                    gaussian_rotations = view_rotations

                else:
                    points_ndc, valid_z = project(points=gaussian_means,
                                                  intrinsics=intrinsics[batch, view],
                                                  extrinsics=extrinsics[batch, view],
                                                  epsilon=1e-8)
                    valid_x = (points_ndc[:, 0] >= 0) & (points_ndc[:, 0] < 1)
                    valid_y = (points_ndc[:, 1] >= 0) & (points_ndc[:, 1] < 1)
                    mask = valid_x & valid_y & valid_z

                    points_2d = torch.zeros_like(points_ndc)
                    points_2d[:, 0] = (points_ndc[:, 0]) * w
                    points_2d[:, 1] = (points_ndc[:, 1]) * h
                    points_2d = points_2d.floor().long()
                    
                    occupied_gaussian_points = gaussian_means[mask]
                    wh_query = torch.chunk(points_2d[mask], 2, dim=-1)
                    occupied_view_points = view_means[wh_query[1] * h + wh_query[0]].squeeze()
                    distances = torch.norm(occupied_gaussian_points - occupied_view_points, dim=1)
                    
                    mask_indices = torch.where(mask)[0]
                    invalid_index = torch.where(distances > distance_thres)[0]
                    invalid_mask_indices = mask_indices[invalid_index]
                    mask[invalid_mask_indices] = False
                    
                    wh_query = torch.chunk(points_2d[mask], 2, dim=-1)
                    image_mask = torch.ones(h * w, dtype=mask.dtype, device=mask.device)
                    image_mask[wh_query[1] * h + wh_query[0]] = 0

                    gaussian_means = torch.cat([gaussian_means, view_means[image_mask]], dim=0)
                    gaussian_covariances = torch.cat([gaussian_covariances, view_covariances[image_mask]], dim=0)
                    gaussian_harmonics = torch.cat([gaussian_harmonics, view_harmonics[image_mask]], dim=0)
                    gaussian_opacities = torch.cat([gaussian_opacities, view_opacities[image_mask]], dim=0)
                    gaussian_rotations = torch.cat([gaussian_rotations, view_rotations[image_mask]], dim=0)
                    gaussian_scales = torch.cat([gaussian_scales, view_scales[image_mask]], dim=0)

            gaussian_list.append(
                Gaussian(
                    means=gaussian_means,
                    covariances=gaussian_covariances,
                    harmonics=gaussian_harmonics,
                    opacities=gaussian_opacities,
                    scales=gaussian_scales,
                    rotations=gaussian_rotations
                )
            )
        return gaussian_list

    def gaussian_scorer(self,
                        gaussian_centers,
                        score_maps,
                        extrinsics,
                        intrinsics,
                        alphas,
                        image_size):
        
        h, w = image_size
        v = extrinsics.shape[0]
        N = gaussian_centers.shape[0]
        gaussian_scores = torch.zeros((N, v), dtype=gaussian_centers.dtype, device=gaussian_centers.device)

        for view in range(v):
            points_ndc, valid_z = project(points=gaussian_centers,
                                          intrinsics=intrinsics[view],
                                          extrinsics=extrinsics[view],
                                          epsilon=1e-8)
            valid_x = (points_ndc[:, 0] >= 0) & (points_ndc[:, 0] < 1)
            valid_y = (points_ndc[:, 1] >= 0) & (points_ndc[:, 1] < 1)
            mask = valid_x & valid_y & valid_z

            points_2d = torch.zeros_like(points_ndc)
            points_2d[:, 0] = (points_ndc[:, 0]) * w
            points_2d[:, 1] = (points_ndc[:, 1]) * h
            points_2d = points_2d.floor().long()
            wh_query = torch.chunk(points_2d[mask], 2, dim=-1)
            gaussian_scores[mask, view] = score_maps[view, wh_query[1], wh_query[0]].squeeze()

        return torch.matmul(gaussian_scores, alphas)

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
            
    def gaussian_pruner(self, gaussians, prune_idx):
        
        prune_opacities = gaussians.opacities[prune_idx]
        to_remove = prune_idx[torch.where(prune_opacities < self.opacity_thres)]
        to_modify = prune_idx[torch.where(prune_opacities >= self.opacity_thres)]

        means = gaussians.means
        scales = gaussians.scales
        rotations = gaussians.rotations
        covariances = gaussians.covariances
        harmonics = gaussians.harmonics
        opacities = gaussians.opacities
       
        opacities[to_modify] *= self.opacity_factor

        N = opacities.shape
        mask = torch.ones(N, dtype=torch.bool, device=opacities.device)
        mask[to_remove] = False

        return Gaussian(
            means=means[mask],
            scales=scales[mask],
            rotations=rotations[mask],
            covariances=covariances[mask],
            harmonics=harmonics[mask],
            opacities=opacities[mask]
        )

    def gaussian_splitter(self, gaussians, split_idx):

        means = gaussians.means
        scales = gaussians.scales
        rotations = gaussians.rotations
        covariances = gaussians.covariances
        harmonics = gaussians.harmonics
        opacities = gaussians.opacities

        dupli_means = means[split_idx]
        dupli_scales = scales[split_idx]
        dupli_rotations = rotations[split_idx]
        dupli_covs = covariances[split_idx]
        dupli_shs = harmonics[split_idx]
        dupli_opa = opacities[split_idx]

        return Gaussian(
            means=torch.cat([means, dupli_means], dim=0),
            scales=torch.cat([scales, dupli_scales], dim=0),
            rotations=torch.cat([rotations, dupli_rotations], dim=0),
            covariances=torch.cat([covariances, dupli_covs], dim=0),
            harmonics=torch.cat([harmonics, dupli_shs], dim=0),
            opacities=torch.cat([opacities, dupli_opa], dim=0)
        )
                
    def forward(self,
                origin_gaussians,
                score_maps,
                alphas,
                extrinsics, 
                intrinsics,
                image_size):

        device = score_maps.device
        gaussians = self.remove_redundancy(
            gaussians=origin_gaussians,
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            image_size=image_size
        )
        
        h, w = image_size
        b, v, _ , _ = extrinsics.shape
        assert len(gaussians) == b

        metas = self.get_deformable_metas(
            extrinsics=extrinsics.squeeze(),
            intrinsics=intrinsics.squeeze(),
            img_size=image_size,
            device=device
        )

        # split and prune
        for batch in range(b):
            cur_gaussians = gaussians[batch]
            cur_extrinsics = extrinsics[batch]
            cur_intrinsics = intrinsics[batch]
            
            for stage in range(self.stages):

                gaussian_scores = self.gaussian_scorer(
                    gaussian_centers=cur_gaussians.means,
                    score_maps=score_maps,
                    extrinsics=cur_extrinsics,
                    intrinsics=cur_intrinsics,
                    alphas=alphas,
                    image_size=image_size
                )
                
                tao_low, tao_high = self.hypers[stage](
                    gaussians=cur_gaussians,
                    gaussian_scores=gaussian_scores,
                    score_maps=score_maps,
                    metas=metas
                )

                split_idx = torch.where(gaussian_scores > tao_high)[0]
                prune_idx = torch.where(gaussian_scores < tao_low)[0]

                cur_gaussians = self.gaussian_splitter(cur_gaussians, split_idx)
                cur_gaussians = self.gaussian_pruner(cur_gaussians, prune_idx)

            gaussians[batch] = cur_gaussians  
            
        return gaussians
  