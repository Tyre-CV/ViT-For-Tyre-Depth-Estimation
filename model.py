# TODO: 
# - layernorm, when?
# - dropout, when?

import torch
import torch.nn as nn
from torchinfo import summary


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim=1000, patch_size=(5,5), img_size=(1000, 2000), in_chans=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.img_size = img_size
        self.in_chans = in_chans

        self.embedding = nn.Conv2d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # x: (B, H, W)
        B, H, W = x.shape
        x = x.unsqueeze(1)                # (B, 1, H, W)
        x = self.embedding(x)             # (B, C, H', W')
        x = torch.flatten(x, start_dim=2) # (B, C, P)
        x = x.permute(0, 2, 1)            # (B, P, C)
        return x


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.layerNorm1 = nn.LayerNorm(embed_dim)
        self.layerNorm2 = nn.LayerNorm(embed_dim)
        self.mhsa = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.attn_drop = nn.Dropout(p=dropout)
        self.ffwd = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(embed_dim*4, embed_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        # x: (B, P, C)
        residual = x
        x = self.layerNorm1(x)
        x = x.permute(1, 0, 2)                # (P, B, C)
        attn_output, attn_weights = self.mhsa(
            x, x, x,
            need_weights=True,
            average_attn_weights=True)
        x = attn_output.permute(1, 0, 2) # back to (B, P, C)
        x = residual + self.attn_drop(x) 

        residual = x
        x = self.layerNorm2(x)
        x = self.ffwd(x)                     # (B, P, C)
        x = residual + x                   # second residual
        return x, attn_weights


class Transformer(nn.Module):
    def __init__(self,
                 embed_dim=1000,
                 num_layers=6,
                 num_heads=8,
                 dropout=0.1,
                 patch_size=(5,5), # (h,w)
                #  patch_dir='horizontal',
                 img_size=(1000, 2000),
                 in_chans=1,
                 num_classes=6):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.patch_size = patch_size
        # self.patch_dir = patch_dir
        self.img_size = img_size
        self.in_chans = in_chans
        self.num_classes = num_classes

        # Patch embedding
        self.patch_embedding = PatchEmbedding(embed_dim, patch_size, img_size, in_chans)

        # Compute number of patches P
        grid_h = img_size[0] // patch_size[0]
        grid_w = img_size[1] // patch_size[1]
        P = grid_h * grid_w

        self.pos_drop = nn.Dropout(p=dropout)

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Position embedding for CLS + patches
        self.pos_embed = nn.Parameter(torch.zeros(1, P+1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoders
        self.encoders = nn.ModuleList([
            Encoder(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

        # Classification head
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_attn = False):
        attn_maps = []

        # x: (B, H, W)
        B = x.size(0)
        x = self.patch_embedding(x)            # (B, P, C)

        # prepend cls token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, C)
        x = torch.cat([cls, x], dim=1)           # (B, P+1, C)

        # add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # transformer layers
        for enc in self.encoders:
            x, layer_attn = enc(x)
            attn_maps.append(layer_attn)

        # extract cls and classify
        cls_out = x[:, 0]                   # (B, C)
        logits = self.classifier(cls_out)   # (B, num_classes)
        return logits, attn_maps if return_attn else logits

    def classify(self, x):
        logits = self.forward(x)
        # Normalise logits to make less spiky?
        return torch.softmax(logits, dim=-1)

    
# Input
# B x P x C
# Output
# B x #Classes
if __name__ == "__main__":
    # Test model
    sample_input = torch.randn(32, 1000, 2000)  # Example input
    model = Transformer(embed_dim=1000, num_layers=6, num_heads=8, dropout=0.1, patch_size=(5,5), img_size=(1000, 2000), in_chans=1, num_classes=6)
    output = model(sample_input)
    predict = model.classify(sample_input)
    summary(
        model,
        input_size=(32, 1000, 2000),  # batch, height, width
        col_names=("input_size", "output_size", "num_params"),
        depth=4
    )