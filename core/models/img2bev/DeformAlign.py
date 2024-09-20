import packages.ops_dcnv3.modules.dcnv3 as dcnv3
import torch
import torch.nn as nn
from torch.nn import CosineSimilarity
import torch.nn.functional as F
import packages.ops_dcnv3.functions.dcnv3_func as dcnv3_func

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FeatureAlign(nn.Module):
    def __init__(self,
                 fea_dim=64,
                 kernel_size=3,
                 group=4,
                 offset_scale=1.0,
                 align_conv_dim=32):
        super(FeatureAlign, self).__init__()

        # Deformable convolution
        # self.deformable_conv = dcnv3.DCNv3(channels=fea_dim, kernel_size=kernel_size, group=group,
        #                                   offset_scale=offset_scale)
        self.deformable_conv = deform_align(channels=fea_dim, kernel_size=kernel_size, group=group,
                                           offset_scale=offset_scale)

        # Feature alignment convolutions with batch normalization and ReLU activation
        self.alignConv1 = nn.Sequential(
            nn.Conv3d(2, align_conv_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(align_conv_dim),
            nn.ReLU(inplace=True)
        )
        self.alignConv2 = nn.Sequential(
            nn.Conv3d(align_conv_dim, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )
    def forward(self, current_feat, reference_feat):
        """
        current_feat: [B, N, C, H, W]
        reference_feat: [B, N, C, H, W]
        """
        # Concatenate features along the channel dimension
        x = torch.cat((current_feat, reference_feat), dim=1)

        # Feature alignment
        x = self.alignConv2(self.alignConv1(x))

        # Squeeze and permute for deformable convolution
        x = x.squeeze(0).permute(0, 2, 3, 1)
        reference_feat = reference_feat.squeeze(0).permute(0, 2, 3, 1).contiguous()
        # Deformable convolution
        out = self.deformable_conv(reference_feat, x)

        # Permute and add batch dimension
        out = out.permute(0, 3, 1, 2).unsqueeze(0)

        return out


class deform_align(dcnv3.DCNv3):
    def forward(self, target, input):
        """
        :param query                       (N, H, W, C)
        :return output                     (N, H, W, C)
        """
        N, H, W, _ = input.shape

        x = self.input_proj(input)
        x_proj = x
        dtype = x.dtype

        x1 = input.permute(0, 3, 1, 2)
        x1 = self.dw_conv(x1)
        offset = self.offset(x1)
        mask = self.mask(x1).reshape(N, H, W, self.group, -1)
        mask = F.softmax(mask, -1).reshape(N, H, W, -1).type(dtype)

        x = dcnv3_func.DCNv3Function.apply(
            target, offset, mask,
            self.kernel_size, self.kernel_size,
            self.stride, self.stride,
            self.pad, self.pad,
            self.dilation, self.dilation,
            self.group, self.group_channels,
            self.offset_scale,
            256)

        if self.center_feature_scale:
            center_feature_scale = self.center_feature_scale_module(
                x1, self.center_feature_scale_proj_weight, self.center_feature_scale_proj_bias)
            # N, H, W, groups -> N, H, W, groups, 1 -> N, H, W, groups, _d_per_group -> N, H, W, channels
            center_feature_scale = center_feature_scale[..., None].repeat(
                1, 1, 1, 1, self.channels // self.group).flatten(-2)
            x = x * (1 - center_feature_scale) + x_proj * center_feature_scale
        x = self.output_proj(x)

        return x


