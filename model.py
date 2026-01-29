import torch.nn as nn
import torch
import os
from transformers import AutoModel
from peft import get_peft_model, LoraConfig


def load_lora_backbone():
    model = AutoModel.from_pretrained(
        "facebook/dinov3-vitl16-pretrain-lvd1689m",
    )
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["k_proj", "v_proj", "q_proj", "o_proj"],
        lora_dropout=0.01, 
        use_dora=True,
        bias="none",
    )
    lora_model = get_peft_model(model, peft_config)
    return lora_model

def save_checkpoint(checkpoint_path, model):
    st = model.state_dict()
    torch.save(st, checkpoint_path)


class LocalMambaBlock(nn.Module):
    def __init__(self, dim, kernel_size=5, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        g = torch.sigmoid(self.gate(x))
        x = x * g
        x = x.transpose(1, 2)  # -> (B, D, N)
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # -> (B, N, D)
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x


class RegressionHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fusion = nn.Sequential(
            LocalMambaBlock(dim, kernel_size=5, dropout=0.1),
            LocalMambaBlock(dim, kernel_size=5, dropout=0.1),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head_green_raw = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim // 2, 1),
            nn.Softplus(),
        )
        self.head_clover_raw = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim // 2, 1),
            nn.Softplus(),
        )
        self.head_dead_raw = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim // 2, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        x_fused = self.fusion(x)
        x_pool = self.pool(x_fused.transpose(1, 2)).flatten(1)
        green = self.head_green_raw(x_pool)
        clover = self.head_clover_raw(x_pool)
        dead = self.head_dead_raw(x_pool)
        gdm = green + clover
        total = gdm + dead
        return (
            green.squeeze(-1),
            dead.squeeze(-1),
            clover.squeeze(-1),
            gdm.squeeze(-1),
            total.squeeze(-1),
        )


class BiomassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = load_lora_backbone()
        # self.backbone = load_convnet_backbone()
        nf = self.backbone.config.hidden_size
        # nf = self.backbone.config.hidden_sizes[-1]
        self.heads = RegressionHead(nf)

    def forward(
        self,
        left,
        right,
        get_embed=False,
    ):
        x_l = self.backbone(left).last_hidden_state[:, -1024:]  #
        x_r = self.backbone(right).last_hidden_state[:, -1024:]  # -1024
        x_cat = torch.cat([x_l, x_r], dim=1)
        return self.heads(x_cat, get_embed)

    def get_all_values(self, total_pred, props):
        # Tính toán giá trị thành phần: Total * Proportion
        green = total_pred * props[:, 0]
        dead = total_pred * props[:, 1]
        clover = total_pred * props[:, 2]
        gdm = green + clover
        return (
            green.squeeze(-1),
            dead.squeeze(-1),
            clover.squeeze(-1),
            gdm.squeeze(-1),
            total_pred.squeeze(-1),
        )
