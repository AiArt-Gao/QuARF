import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, num_groups, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.num_experts = num_experts
        self.num_groups = num_groups

        # 单一的路由层
        self.route_linear = nn.Linear(n_embed, num_experts * num_groups)
        self.noise_linear = nn.Linear(n_embed, num_experts * num_groups)

    def forward(self, mh_output):
        # mh_output shape: [batch_size, n_embed]
        batch_size = mh_output.shape[0]

        logits = self.route_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)

        # Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits) * F.softplus(noise_logits)
        noisy_logits = logits + noise

        # Reshape to [batch_size, num_groups, num_experts]
        noisy_logits = noisy_logits.view(batch_size, self.num_groups, self.num_experts)

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)

        return router_output, indices

class QuARF(nn.Module):
    def __init__(self, in_channels,quality_dim=4096, num_experts=4, top_k=4, num_groups=4，mode='soft'):
        super(QuARF, self).__init__()
        self.router = NoisyTopkRouter(quality_dim, num_experts, num_groups, top_k)
        self.num_groups = num_groups
        self.channels_per_group = in_channels // num_groups
        self.top_k = top_k
        self.mode = mode
        self.num_experts = num_experts

        kernel_size = 3
        self.experts = nn.ModuleList([
            nn.Conv2d(self.channels_per_group, self.channels_per_group, kernel_size, padding=1, dilation=1),
            nn.Conv2d(self.channels_per_group, self.channels_per_group, kernel_size, padding=2, dilation=2),
            nn.Conv2d(self.channels_per_group, self.channels_per_group, kernel_size, padding=4, dilation=4),
            nn.Conv2d(self.channels_per_group, self.channels_per_group, kernel_size, padding=8, dilation=8)
        ])

    def forward(self, x, quality_feature):

        b, c, h, w = x.shape
        # 使用路由器获取每个组的专家选择
        router_output, indices = self.router(quality_feature)  # [b, num_groups, num_experts], [b, num_groups, top_k]

        x_groups = x.chunk(self.num_groups, dim=1)

        outputs = []
        group_selections = []
        for group_idx, x_group in enumerate(x_groups):
            if self.mode == 'hard':
                # print("hard")
                # Hard selection: 只选择概率最高的专家
                expert_indices = indices[:, group_idx, 0]  # [batch_size]
                group_output = torch.zeros_like(x_group)
                for i in range(b):
                    expert = self.experts[expert_indices[i]]
                    group_output[i:i + 1] = expert[1](expert[0](x_group[i:i + 1]))
                group_selections.append(expert_indices.cpu().numpy())

            elif self.mode == 'soft':
                # Soft selection: 使用所有专家的加权和
                group_output = torch.zeros_like(x_group)
                for i in range(self.num_experts):
                    expert = self.experts[i]
                    gating_scores = router_output[:, group_idx, i].view(b, 1, 1, 1)
                    expert_output = expert(x_group)
                    group_output += gating_scores * expert_output
                group_selections.append(router_output[:, group_idx].detach().cpu().numpy())

            outputs.append(group_output)

        return torch.cat(outputs, dim=1)
