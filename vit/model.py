import torch
from torch import nn


class ViT(nn.Module):
    def __init__(self, hid_dim=512, cifar_classes=10):
        super(ViT, self).__init__()
        self.position_emb = PositionEmbedding()
        self.transformer_encoder = TransformerEncoder()
        self.linear = nn.Linear(hid_dim, cifar_classes)

    def forward(self, inputs):
        out = self.position_emb(inputs)
        enc_out = self.transformer_encoder(out)  # (bs, n+1, hid_dim)
        cls_out = self.linear(enc_out[:, 0, :])
        return cls_out


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dims=512, num_heads=8):
        super(TransformerBlock, self).__init__()
        self.norm = nn.LayerNorm(hidden_dims)  #!TODO: try not sharing weights
        self.mlp = nn.Linear(hidden_dims, hidden_dims)
        self.mha = nn.MultiheadAttention(
            hidden_dims, num_heads, batch_first=True
        )  # seqlen, bs, feat

    def forward(self, inputs):
        norm_out = self.norm(inputs)
        mha_out, _ = self.mha(norm_out, norm_out, norm_out)

        mid_out = mha_out + inputs
        norm_out_2 = self.norm(mid_out)
        mlp_out = self.mlp(norm_out_2)
        return mlp_out + mid_out


class TransformerEncoder(nn.Module):
    def __init__(self):
        super(TransformerEncoder, self).__init__()
        self.blocks = nn.Sequential(TransformerBlock())

    def forward(self, inputs):
        out = self.blocks(inputs)
        return out


class PositionEmbedding(nn.Module):  # TODO learned embeddings
    def __init__(self, patch_per_side=4, hidden_dims=512):
        super(PositionEmbedding, self).__init__()
        self.clstoken = nn.Parameter(torch.randn(1, 1, hidden_dims))
        self.patch_per_side = patch_per_side
        self.pos_emb = torch.arange(patch_per_side + 1).unsqueeze(-1)  # (n+1, 1)
        self.hidden_dims = hidden_dims
        self.lin = nn.Linear(768, self.hidden_dims)

    def forward(self, inputs):
        x = self.lin(inputs)
        batch_size, seq_len, hidden_dims = x.shape
        clstok_added = torch.cat([self.clstoken.repeat(batch_size, 1, 1), x], dim=1)
        clstok_added += self.pos_emb.repeat(batch_size, 1, hidden_dims)
        return clstok_added


if __name__ == "__main__":
    inp_tensor = torch.randn(8, 4, 768)
    model = ViT()
    print(model(inp_tensor).shape)
