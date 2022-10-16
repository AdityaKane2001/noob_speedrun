from itertools import count
import torch
from torch import nn


class ViT(nn.Module):
    """
    Main ViT model
    """
    def __init__(self, hid_dim=512, cifar_classes=10, num_layers=8):
        super(ViT, self).__init__()
        self.position_emb = PositionEmbedding()
        self.transformer_encoder = TransformerEncoder(num_layers)
        self.linear = nn.Linear(hid_dim, cifar_classes)

    def forward(self, inputs):
        out = self.position_emb(inputs) # (bs, n+1, hid_dim)
        enc_out = self.transformer_encoder(out)  # (bs, n+1, hid_dim)
        cls_out = self.linear(enc_out[:, 0, :]) # (bs, n+1, cifar_classes)
        return cls_out


class PositionEmbedding(nn.Module):  # TODO learned embeddings
    """
    Position Embedding, a very naive implementation
    CLS is init random
    Position Embedding is torch.arange(num_patches)
    Need to make Position Embedding Learnable
    """
    def __init__(self, num_patches=4, hid_dim=512):
        super(PositionEmbedding, self).__init__()
        self.clstoken = nn.Parameter(torch.randn(1, 1, hid_dim)) # [CLS] token added to the beginning
        self.lin = nn.Linear(768, hid_dim)
        self.register_buffer(name="pos_emb", tensor=torch.arange(num_patches + 1).unsqueeze(-1))

    def forward(self, inputs):
        x = self.lin(inputs) # (bs, n, hid_dim)
        batch_size, seq_len, hid_dim = x.shape
        clstok_added = torch.cat([self.clstoken.repeat(batch_size, 1, 1), x], dim=1) # (bs, n_1, hid_dim)
        clstok_added += self.pos_emb.repeat(batch_size, 1, hid_dim) # Adds the position embedding
        return clstok_added


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    Uses Transformer Block and stacks it for num_layers
    """
    def __init__(self, num_layers=1):
        super(TransformerEncoder, self).__init__()
        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(num_layers)])

    def forward(self, inputs):
        for layer in self.blocks:
            out = layer(inputs)
        return out


class TransformerBlock(nn.Module):
    """
    Transformer Block for Transformer Encoder
    """
    def __init__(self, hid_dim=512, num_heads=8):
        super(TransformerBlock, self).__init__()
        self.norm = nn.LayerNorm(hid_dim)  #!TODO: try not sharing weights
        self.mlp = nn.Linear(hid_dim, hid_dim)
        self.mha = nn.MultiheadAttention(
            hid_dim, num_heads, batch_first=True
        )  # (bs, seqlen, feat) == (bs, n+1, hid_dim)

    def forward(self, inputs):
        norm_out = self.norm(inputs) # (bs, n+1, hid_dim)
        mha_out, _ = self.mha(norm_out, norm_out, norm_out) # (bs, n+1, hid_dim)
        mha_out = mha_out + inputs # (bs, n+1, hid_dim)
        # mlp_out = self.mlp(self.norm(mha_out)) # This is implemented in the paper, but it throws error for mps, not for cpu or cuda
        ## RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        mlp_out = self.norm(self.mlp(mha_out)) # Reversing the norm and MLP fixes it
        return mlp_out + mha_out 


if __name__ == "__main__":
    inp_tensor = torch.randn(8, 4, 768)
    model = ViT(num_layers=1) # 1.7M for 1, 10.9M for 8 layers
    print(inp_tensor.shape, model(inp_tensor).shape)
    from utils import count_parameters
    count_parameters(model)
