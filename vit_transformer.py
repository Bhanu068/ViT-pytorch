import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """ Split the image into patches and embed them
    Parameters
    ----------
    img_size: size of the image
    patch_size: size of each patch
    inp_channels: Number of channels of input (rgb-3, grayscale-1)
    embed_dim: Embedding dimensions

    Attributes
    ----------
    n_patches: Number of patches
    proj: nn.Conv2d - Convolutional layer that does both the splitting and embedding of the patches
    """

    def __init__(self, img_size, patch_size, inp_channels=3, embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(inp_channels,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    # x shape: (batch_size, inp_channels, img_size, img_size)
    def forward(self, x):
        # (batch_size, embed_dim, n_patches ** 0.5, n_patches ** 0.5)
        x = self.proj(x)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)
        return x


class Attention(nn.Module):
    """Attention mechanism.
    Parameters
    ----------
    dim : int
        The input and out dimension of per token features.
    n_heads : int
        Number of attention heads.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    dropout_attn : float
        Dropout probability applied to the query, key and value tensors.
    dropout_output : float
        Dropout probability applied to the output tensor.
    Attributes
    ----------
    scale : float
        Normalizing consant for the dot product.
    qkv : nn.Linear
        Linear projection for the query, key and value.
    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space.
    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """

    def __init__(self, dim, n_heads=12, qkv_bias=True, dropout_attn=0., dropout_output=0.0):
        super(Attention, self).__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = self.dim // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim ** 3, bias = qkv_bias)
        self.qkv_bias = qkv_bias
        self.dropout_attn = nn.Dropout(dropout_attn)
        self.softmax = nn.Softmax(dim = -1)

        self.dropout_output = nn.Dropout(dropout_output)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        batch_size, n_patches, dim = x.shape
        if dim != dim:
            raise ValueError
        
        qkv = self.qkv(x) # batch_size, n_patches + 1, 3 * dim
        qkv = qkv.reshape(batch_size, n_patches, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4) # 3, batch_size, n_heads, n_patches + 1, head_dim
        q, k, v = qkv[0], qkv[0], qkv[0]
        
        k = k.transpose(-2, -1)     # (batch_size, n_heads, head_dim, n_patches + 1)        
        attn = self.softmax((q * k) * self.scale)   # (batch_size, n_heads, n_patches + 1, n_patches + 1)
        
        attn = self.dropout_attn(attn)
        value = attn @ v       # (batch_size, n_heads, n_patches +1, head_dim)
        value = value.transpose(1, 2) # (batch_size, n_patches + 1, n_heads, head_dim)
        value = value.flatten(2)    # (n_samples, n_patches + 1, dim)
        
        out = self.proj(value)  # (n_samples, n_patches + 1, dim)
        out = self.dropout_output(out)
        
        return out
    

class MLP(nn.Module):
    """Multilayer perceptron.
    Parameters
    ----------
    in_features : int
        Number of input features.
    hidden_features : int
        Number of nodes in the hidden layer.
    out_features : int
        Number of output features.
    p : float
        Dropout probability.
    Attributes
    ----------
    fc : nn.Linear
        The First linear layer.
    act : nn.GELU
        GELU activation function.
    fc2 : nn.Linear
        The second linear layer.
    drop : nn.Dropout
        Dropout layer.
    """
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(batch_size, n_patches + 1, in_features)`.
        Returns
        -------
        torch.Tensor
            Shape `(batch_size, n_patches +1, out_features)`
        """
        x = self.fc1(
                x
        ) # (batch_size, n_patches + 1, hidden_features)
        x = self.act(x)  # (batch_size, n_patches + 1, hidden_features)
        x = self.drop(x)  # (batch_size, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (batch_size, n_patches + 1, hidden_features)
        x = self.drop(x)  # (batch_size, n_patches + 1, hidden_features)

        return x   
    
class VisionTransformerBlock(nn.Module):
    """Transformer block.
    Parameters
    ----------
    dim : int
        Embeddinig dimension.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module with respect
        to `dim`.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    dropout_output, dropout_attn : float
        Dropout probability.
    Attributes
    ----------
    norm1, norm2 : LayerNorm
        Layer normalization.
    attn : Attention
        Attention module.
    mlp : MLP
        MLP module.
    """
    def __init__(self, dim, n_heads, mlp_ratio = 4.0, qkv_bias = True, dropout_output = 0., dropout_attn = 0.):
        super(VisionTransformerBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
                dim,
                n_heads = n_heads,
                qkv_bias = qkv_bias,
                dropout_attn = dropout_attn,
                dropout_output = dropout_output
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
                in_features=dim,
                hidden_features=hidden_features,
                out_features=dim,
        )

    def forward(self, x):
        """Run forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(batch_size, n_patches + 1, dim)`.
        Returns
        -------
        torch.Tensor
            Shape `(batch_size, n_patches + 1, dim)`.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x   
    
    
class VisionTransformer(nn.Module):
    """Simplified implementation of the Vision transformer.
    Parameters
    ----------
    img_size : int
        Both height and the width of the image (it is a square).
    patch_size : int
        Both height and the width of the patch (it is a square).
    in_chans : int
        Number of input channels.
    n_classes : int
        Number of classes.
    embed_dim : int
        Dimensionality of the token/patch embeddings.
    num_layers : int
        Number of VisionTransformer blocks.
    n_heads : int
        Number of attention heads.
    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module.
    qkv_bias : bool
        If True then we include bias to the query, key and value projections.
    dropout_output, dropout_attn : float
        Dropout probability.
    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of `PatchEmbed` layer.
    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has `embed_dim` elements.
    pos_emb : nn.Parameter
        Positional embedding of the cls token + all the patches.
        It has `(n_patches + 1) * embed_dim` elements.
    pos_drop : nn.Dropout
        Dropout layer.
    blocks : nn.ModuleList
        List of `Block` modules.
    norm : nn.LayerNorm
        Layer normalization.
    """
    def __init__(
            self,
            img_size = 384,
            patch_size = 16,
            in_chans = 3,
            n_classes = 1000,
            embed_dim = 768,
            num_layers = 12,
            n_heads = 12,
            mlp_ratio = 4.,
            qkv_bias = True,
            dropout_output = 0.,
            dropout_attn = 0.,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
                img_size = img_size,
                patch_size = patch_size,
                inp_channels = in_chans,
                embed_dim = embed_dim,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
                torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(dropout_output)

        self.blocks = nn.ModuleList(
            [
                VisionTransformerBlock(
                    dim = embed_dim,
                    n_heads = n_heads,
                    mlp_ratio = mlp_ratio,
                    qkv_bias = qkv_bias,
                    dropout_output = dropout_output,
                    dropout_attn = dropout_attn,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps = 1e-6)
        self.head = nn.Linear(embed_dim, n_classes)


    def forward(self, x):
        """Run the forward pass.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_size, img_size)`.
        Returns
        -------
        logits : torch.Tensor
            Logits over all the classes - `(n_samples, n_classes)`.
        """
        n_samples = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(
                n_samples, -1, -1
        )  # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        x = x + self.pos_embed  # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        cls_token_final = x[:, 0]  # just the CLS token
        x = self.head(cls_token_final)

        return x  
         
        
        
        
        
