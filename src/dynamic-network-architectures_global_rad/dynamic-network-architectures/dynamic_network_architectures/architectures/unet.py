from typing import Union, Type, List, Tuple

import torch
from dynamic_network_architectures.building_blocks.helper import convert_conv_op_to_dim
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder
from dynamic_network_architectures.building_blocks.residual import BasicBlockD, BottleneckD
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.unet_residual_decoder import UNetResDecoder
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

# class CrossAttentionBlock(nn.Module):
#     """
#     A cross-attention block that fuses global features with spatial features.
    
#     Given:
#       - bottleneck: Tensor of shape [B, C, H, W] (e.g., UNet bottleneck features)
#       - global_feats: Tensor of shape [B, global_dim]
      
#     The module:
#       1. Flattens the spatial dimensions of the bottleneck to form a sequence of tokens.
#       2. Projects the global feature vector into the same embedding space (C-dim).
#       3. Uses multi-head attention with the projected global features as queries and
#          the spatial tokens as keys and values.
         
#     Returns:
#       - attn_output: Updated global features with attended spatial information (shape [B, 1, C]).
#       - attn_weights: Attention weights for inspection (shape [1, B, H*W]).
#     """
    
#     def __init__(self, bottleneck_channels, global_feat_dim, num_heads=8):
#         super(CrossAttentionBlock, self).__init__()
#         # Linear projection to map global features into the same space as bottleneck channels
#         self.proj = nn.Linear(global_feat_dim, bottleneck_channels)
#         # Multi-head attention module
#         self.attn = nn.MultiheadAttention(embed_dim=bottleneck_channels, num_heads=num_heads)
    
#     def forward(self, bottleneck, global_feats):
#         # bottleneck: [B, C, H, W]
#         # global_feats: [B, global_feat_dim]
#         # print(bottleneck.size(), global_feats.size())
#         B, C, H, W = bottleneck.shape
        
#         # Flatten spatial dimensions: reshape to [B, H*W, C]
#         bottleneck_flat = bottleneck.view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        
#         # Project global features to [B, C] then add a singleton dimension to form [B, 1, C]
#         global_feats_proj = self.proj(global_feats).unsqueeze(1)  # [B, 1, C]
        
#         # MultiheadAttention in PyTorch expects input shape [seq_len, batch, embed_dim]
#         query = global_feats_proj.transpose(0, 1)  # [1, B, C]
#         key   = bottleneck_flat.transpose(0, 1)      # [H*W, B, C]
#         value = bottleneck_flat.transpose(0, 1)      # [H*W, B, C]
        
#         # Apply cross-attention: global features attend to spatial tokens
#         attn_output, attn_weights = self.attn(query, key, value)
#         # attn_output shape: [1, B, C] -> transpose back to [B, 1, C]
#         attn_output = attn_output.transpose(0, 1)
#         # print(attn_output.size(), attn_weights.size())
        
#         return attn_output, attn_weights

import torch
from torch import nn

class CrossAttentionBlock(nn.Module):
    """
    A cross-attention block that fuses global features with spatial features.
    Adapted for both 2D (4D bottleneck) and 3D (5D bottleneck) spatial inputs.
    
    Given:
      - bottleneck: Tensor of shape [B, C, H, W] (for 2D) or [B, C, D, H, W] (for 3D)
      - global_feats: Tensor of shape [B, global_dim]
      
    The module:
      1. Flattens the spatial dimensions of the bottleneck to form a sequence of tokens.
      2. Projects the global feature vector into the same embedding space (C-dim).
      3. Uses multi-head attention with the projected global features as queries and
         the spatial tokens as keys and values.
         
    Returns:
      - attn_output: Updated global features with attended spatial information (shape [B, 1, C]).
      - attn_weights: Attention weights for inspection (shape [1, B, H*W] or [1, B, D*H*W]).
    """
    
    def __init__(self, bottleneck_channels, global_feat_dim, num_heads=8):
        super(CrossAttentionBlock, self).__init__()
        # Linear projection to map global features into the same space as bottleneck channels
        self.proj = nn.Linear(global_feat_dim, bottleneck_channels)
        # Multi-head attention module
        self.attn = nn.MultiheadAttention(embed_dim=bottleneck_channels, num_heads=num_heads)
    
    def forward(self, bottleneck, global_feats):
        # bottleneck: [B, C, S1, S2, ...] where S are spatial dimensions
        # global_feats: [B, global_feat_dim]

        B_bottleneck, C, *spatial_dims = bottleneck.shape
        num_spatial_dims = len(spatial_dims)
        
        # Calculate the product of spatial dimensions
        num_spatial_tokens = 1
        for dim_size in spatial_dims:
            num_spatial_tokens *= dim_size
        
        # Dynamically flatten spatial dimensions: reshape to [B, num_spatial_tokens, C]
        # First, rearrange to [B, C, num_spatial_tokens] then permute
        bottleneck_flat = bottleneck.view(B_bottleneck, C, num_spatial_tokens).permute(0, 2, 1)  # [B, num_spatial_tokens, C]
        
        # Ensure global_feats batch size matches bottleneck's batch size
        if global_feats.shape[0] == 1 and B_bottleneck > 1:
            global_feats = global_feats.expand(B_bottleneck, -1) 
        elif global_feats.shape[0] != B_bottleneck:
            raise ValueError(f"Batch size of global_feats ({global_feats.shape[0]}) does not match "
                             f"bottleneck ({B_bottleneck}) and is not 1 for expansion.")
                             
        # Project global features to [B, C] then add a singleton dimension to form [B, 1, C]
        global_feats_proj = self.proj(global_feats).unsqueeze(1)  # [B, 1, C]
        
        # MultiheadAttention in PyTorch expects input shape [seq_len, batch, embed_dim]
        query = global_feats_proj.transpose(0, 1)        # [1, B, C]
        key   = bottleneck_flat.transpose(0, 1)          # [num_spatial_tokens, B, C]
        value = bottleneck_flat.transpose(0, 1)          # [num_spatial_tokens, B, C]
        
        # Apply cross-attention: global features attend to spatial tokens
        attn_output, attn_weights = self.attn(query, key, value)
        # attn_output shape: [1, B, C] -> transpose back to [B, 1, C]
        attn_output = attn_output.transpose(0, 1)
        
        return attn_output, attn_weights


class PlainConvUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_conv_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 nonlin_first: bool = False
                 ):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_conv_per_stage: {n_conv_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                        n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                        dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
                                        nonlin_first=nonlin_first)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
                                   nonlin_first=nonlin_first)
        # make attention optional/lazy: do not create unless explicitly enabled
        self.attn_layer = None
        self._bottleneck_channels = features_per_stage[-1]

    def enable_attention_from_dim(self, global_feat_dim: int, num_heads: int = 8) -> None:
        if self.attn_layer is None:
            self.attn_layer = CrossAttentionBlock(self._bottleneck_channels, global_feat_dim, num_heads=num_heads)
#     def forward(self, x, global_info):
#         # print(type(global_info))
#         # print(x.size(), global_info.size())
#         skips = self.encoder(x)
#         print(f'skips size: {skips[-1].size()}')
#         print(f'global_info size: {global_info.size()}')
#         global_info = global_info.repeat(skips[0].size(0), 1)
#         # print(global_info.size())
#         attn_out, _ = self.attn_layer(skips[-1], global_info)
#         # print(f'attn_out size: {attn_out.size()}')
#         attn_gate = attn_out.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # shape: [B, C, 1, 1]
#         bottleneck = skips[-1]

# # Apply the attention as a channel-wise gate (broadcasting over H and W):
#         modified_bottleneck = bottleneck * attn_gate

#         # print(f'after attn size: {modified_bottleneck.size()}')
#         # print('Model output:')
#         assert modified_bottleneck != skips[-1], 'already replaced!'
#         skips[-1] = modified_bottleneck
#         # print(len(skips))
#         # for t in skips:
#         #     print(t.size())
#         # print(skips[-1].size())
#         # [320,6,6,6]
#         return self.decoder(skips)
    def forward(self, x, global_info=None):
        # Encoder forward pass
        skips = self.encoder(x)
        if (self.attn_layer is None) or (global_info is None):
            return self.decoder(skips)

        bottleneck = skips[-1]
        B_bottleneck = bottleneck.shape[0]
        if global_info.shape[0] == 1 and B_bottleneck > 1:
            global_info = global_info.expand(B_bottleneck, -1)
        elif global_info.shape[0] != B_bottleneck:
            raise ValueError(f"Batch size of global_info ({global_info.shape[0]}) does not match bottleneck ({B_bottleneck}) and is not 1 for expansion.")

        attn_out, _ = self.attn_layer(bottleneck, global_info)
        attn_gate = attn_out.squeeze(1)
        num_spatial_dims = len(bottleneck.shape) - 2
        for _ in range(num_spatial_dims):
            attn_gate = attn_gate.unsqueeze(-1)
        skips[-1] = bottleneck * attn_gate
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                            "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                            "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)


class ResidualEncoderUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        ## 
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


class ResidualUNet(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 num_classes: int,
                 n_conv_per_stage_decoder: Union[int, Tuple[int, ...], List[int]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 deep_supervision: bool = False,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 stem_channels: int = None
                 ):
        super().__init__()
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(n_conv_per_stage_decoder, int):
            n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have as many entries as we have " \
                                                  f"resolution stages. here: {n_stages}. " \
                                                  f"n_blocks_per_stage: {n_blocks_per_stage}"
        assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
                                                                f"as we have resolution stages. here: {n_stages} " \
                                                                f"stages, so it should have {n_stages - 1} entries. " \
                                                                f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        self.encoder = ResidualEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
                                       n_blocks_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
                                       dropout_op_kwargs, nonlin, nonlin_kwargs, block, bottleneck_channels,
                                       return_skips=True, disable_default_stem=False, stem_channels=stem_channels)
        self.decoder = UNetResDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

    def forward(self, x):
        skips = self.encoder(x)
        return self.decoder(skips)

    def compute_conv_feature_map_size(self, input_size):
        assert len(input_size) == convert_conv_op_to_dim(self.encoder.conv_op), "just give the image size without color/feature channels or " \
                                                                                "batch channel. Do not give input_size=(b, c, x, y(, z)). " \
                                                                                "Give input_size=(x, y(, z))!"
        return self.encoder.compute_conv_feature_map_size(input_size) + self.decoder.compute_conv_feature_map_size(input_size)

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
        init_last_bn_before_add_to_0(module)


if __name__ == '__main__':
    data = torch.rand((1, 4, 128, 128, 128))

    model = PlainConvUNet(4, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 4,
                                (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True)

    if False:
        import hiddenlayer as hl

        g = hl.build_graph(model, data,
                           transforms=None)
        g.save("network_architecture.pdf")
        del g

    print(model.compute_conv_feature_map_size(data.shape[2:]))

    data = torch.rand((1, 4, 512, 512))

    model = PlainConvUNet(4, 8, (32, 64, 125, 256, 512, 512, 512, 512), nn.Conv2d, 3, (1, 2, 2, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2, 2, 2), 4,
                                (2, 2, 2, 2, 2, 2, 2), False, nn.BatchNorm2d, None, None, None, nn.ReLU, deep_supervision=True)

    if False:
        import hiddenlayer as hl

        g = hl.build_graph(model, data,
                           transforms=None)
        g.save("network_architecture.pdf")
        del g

    print(model.compute_conv_feature_map_size(data.shape[2:]))
