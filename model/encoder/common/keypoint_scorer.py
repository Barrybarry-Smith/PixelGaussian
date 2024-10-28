import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureScorer(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dims,
                 max_num_view, 
                 eps=1e-8
                ):

        super(KeypointScorer, self).__init__()

        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

        self.upsampler = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=False)
        self.max_num_view = max_num_view
        self.betas = nn.Parameter(torch.ones(max_num_view))
        self.eps = eps


    def forward(self, features, h, w):

        device = features.device

        features = features[0]
        N, input_dim, _, _ = features.shape
        assert self.max_num_view % N == 0

        upsampled_features = self.upsampler(features)
        betas = self.betas.view(N, -1).mean(dim=-1)
        alphas = F.softmax(betas, dim=0)
        weighted_features = alphas.view(N, 1, 1, 1) * upsampled_features

        score_maps = self.mlp(weighted_features.permute(0, 2, 3, 1).reshape(-1, input_dim))
        score_maps = score_maps.view(N, h, w)

        min_score = score_maps.view(N, -1).min(dim=1, keepdim=True)[0].view(N, 1, 1)
        max_score = score_maps.view(N, -1).max(dim=1, keepdim=True)[0].view(N, 1, 1)

        score_maps = (score_maps - min_score) / (max_score - min_score + self.eps)

        return score_maps, alphas


class ContextScorer(nn.Module):
    def __init__(self,
                 channels,
                 num_layers,
                 max_num_view,
                 mode='rgb',
                 eps=1e-8):

        super(ContextScorer, self).__init__()
        self.num_layers = num_layers
        self.channels = channels
        self.eps = eps
        self.max_num_view = max_num_view
        self.betas = nn.Parameter(torch.ones(max_num_view))

        assert len(self.channels) == self.num_layers
        in_channels = 3 if mode == 'rgb' else None

        layers = []
        for idx in range(num_layers):
            out_channels = self.channels[idx]
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels

        self.layers = nn.Sequential(*layers)

    
    def forward(self, context):
        b, v, c, h, w = context.shape
        context = context.view(b * v, c, h, w)
        score_maps = self.layers(context).squeeze()

        min_score = score_maps.view(v, -1).min(dim=1, keepdim=True)[0].view(v, 1, 1)
        max_score = score_maps.view(v, -1).max(dim=1, keepdim=True)[0].view(v, 1, 1)

        betas = self.betas.view(v, -1).mean(dim=-1)
        alphas = F.softmax(betas, dim=0)
        score_maps = (score_maps - min_score) / (max_score - min_score + self.eps)

        return score_maps, alphas


if __name__ == '__main__':
    scorer = TransformerScoreNetwork()
    inputs = torch.randn((1, 2, 3, 256, 256))
    scores = scorer(inputs)
