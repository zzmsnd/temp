import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
import torch.nn.functional as F


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Encoder1(nn.Module):
    def __init__(self, bands, feature_dim):
        super(Encoder1, self).__init__()
        self.dim1 = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(bands, self.dim1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.dim1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.dim1, self.dim1 * 2, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.dim1 * 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.dim1 * 2, self.dim1 * 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.dim1 * 4),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.dim1 * 4, self.dim1 * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(self.dim1 * 8),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x1 = self.conv1(x)  # [90, 64, 12, 12]
        x2 = self.conv2(x1)  # [90, 128, 6, 6]
        x3 = self.conv3(x2)  # [90, 256, 3, 3]
        return x3


class Projector(nn.Module):
    def __init__(self, low_dim):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(low_dim*2, 16)
        self.relu_mlp = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        # self.relu_mlp = nn.ReLU()
        self.fc2 = nn.Linear(16, low_dim)
        self.l2norm = Normalize(2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu_mlp(x)
        x = self.fc2(x)
        x = self.l2norm(x)

        return x


class Classifier0(nn.Module):
    def __init__(self, n_classes):
        super(Classifier0, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(256, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.conv(x)
        # x1 = x.view(x.size(0), -1)
        # x = self.avg(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)

        # x = torch.softmax(x, dim=1)
        return x


class Supervisednetwork(nn.Module):
    def __init__(self, bands, n_classes, low_dim):
        super(Supervisednetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(low_dim*2, low_dim*2, 1),
            nn.BatchNorm2d(low_dim*2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.encoder = Encoder1(bands, low_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(low_dim*2, n_classes)

    def forward(self, x):
        feature = self.encoder(x)
        x = self.conv(feature)
        # x = self.avgpool(feature)
        x = torch.flatten(x, start_dim=1)
        y = self.head(x)
        return y


class Classifier(nn.Module):

    def __init__(self, num_classes=10, feature_dim=256):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.features_dim = feature_dim
        # pseudo head and worst-case estimation head
        mlp_dim = 2 * self.features_dim
        self.head = nn.Linear(self.features_dim, num_classes)
        self.pseudo_head = nn.Sequential(
            nn.Linear(self.features_dim, mlp_dim),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_dim),
            # nn.Dropout(0.5),
            nn.Linear(mlp_dim, num_classes)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x1 = self.conv(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, start_dim=1)
        x1 = torch.flatten(x1, start_dim=1)
        y = self.head(x1)
        y_pseudo = self.pseudo_head(x1)
        # y_pseudo = self.head(x1)
        return y, y_pseudo
        # return y




class Network2(nn.Module):
    def __init__(self, bands, n_classes, low_dim):
        super(Network2, self).__init__()

        self.encoder = Encoder1(bands, low_dim)
        self.projector = Projector(low_dim)
        self.classifier = Classifier0(n_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, u_w=None, u_s=None):
        if u_w==None and u_s==None :
            feature = self.encoder(x)
            cx = self.classifier(feature)
            feature = torch.mean(feature, dim=(2, 3))
            feature = self.projector(feature)
            return feature, cx

        feature1_3 = self.encoder(x)
        cx = self.classifier(feature1_3)
        feature1_3 = self.avgpool(feature1_3)
        feature1_3 = torch.mean(feature1_3, dim=(2, 3))
        feature1_3 = self.projector(feature1_3)

        feature2_3 = self.encoder(u_w)
        cuw = self.classifier(feature2_3)
        feature2_3 = self.avgpool(feature2_3)
        feature2_3 = torch.mean(feature2_3, dim=(2, 3))
        feature2_3 = self.projector(feature2_3)

        feature3_3 = self.encoder(u_s)
        cus = self.classifier(feature3_3)
        feature3_3 = self.avgpool(feature3_3)
        feature3_3 = torch.mean(feature3_3, dim=(2, 3))
        feature3_3 = self.projector(feature3_3)

        return feature1_3, feature2_3, feature3_3, cx, cuw, cus


class Network(nn.Module):
    def __init__(self, bands, n_classes, feature_dim):
        super(Network, self).__init__()
        self.encoder = Encoder1(bands, feature_dim)
        self.classifier = Classifier(n_classes, feature_dim)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        f = self.encoder(x)
        y, y_pseudo = self.classifier(f)

        return y, y_pseudo
        # return y


class dce_loss(torch.nn.Module):
    def __init__(self, n_classes, feat_dim):
        super(dce_loss, self).__init__()
        self.n_classes = n_classes
        self.feat_dim = feat_dim
        self.centers = nn.Parameter(torch.randn(self.feat_dim, self.n_classes).cuda(), requires_grad=True)

    def forward(self, x):
        features_square = torch.sum(torch.pow(x, 2), 1, keepdim=True)
        centers_square = torch.sum(torch.pow(self.centers, 2), 0, keepdim=True)
        features_into_centers = 2 * torch.matmul(x, (self.centers))
        dist = features_square + centers_square - features_into_centers

        return self.centers, -dist


def regularization(features, centers, labels):
    distance = (features - torch.t(centers)[labels])

    distance = torch.sum(torch.pow(distance, 2), 1, keepdim=True)

    distance = (torch.sum(distance, 0, keepdim=True)) / features.shape[0]

    return distance

#我的更改 将编码器改成Transformer
class ConvolutionalEmbedding(nn.Module):
    """动态填充的卷积嵌入层"""

    def __init__(self, in_channels, embed_dim, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        # 保留原始卷积输出（不展平）
        return self.proj(x)  # [B, D, H_out, W_out]


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads,dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim  # 显式存储embed_dim
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 统一输入为三维序列格式 [B, num_patches, D]
        if len(x.shape) == 4:  # 若输入为空间结构 [B, H, W, D]
            B, H, W, D = x.shape
            x_seq = x.reshape(B, H * W, D)
        else:  # 若输入已是序列 [B, num_patches, D]
            x_seq = x

        # 自注意力处理
        attn_output, _ = self.attn(
            self.norm1(x_seq),
            self.norm1(x_seq),
            self.norm1(x_seq),
        )
        x_seq = x_seq + attn_output

        # MLP处理
        x_seq = x_seq + self.mlp(self.norm2(x_seq))

        # 恢复原始结构（若输入是空间结构）
        if len(x.shape) == 4:
            return x_seq.view(B, H, W, self.embed_dim)
        return x_seq


class HierarchicalVisionTransformer(nn.Module):
    def __init__(self, bands, feature_dim=256):
        super().__init__()
        # 超参数配置（关键修改：第三阶段patch_size=1）
        self.embed_dims = [64, 128, feature_dim]
        self.depths = [2, 2, 2]
        self.patch_sizes = [4, 2, 1]  # 第三阶段改为1x1卷积[5](@ref)

        # 阶段1-3初始化
        self.stage1 = self._make_stage(bands, self.embed_dims[0], self.patch_sizes[0], self.depths[0])
        self.stage2 = self._make_stage(self.embed_dims[0], self.embed_dims[1], self.patch_sizes[1], self.depths[1])
        self.stage3 = self._make_stage(self.embed_dims[1], self.embed_dims[2], self.patch_sizes[2], self.depths[2])

        # 空间恢复卷积
        self.final_conv = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)

    def _make_stage(self, in_dim, out_dim, patch_size, depth):
        return nn.Sequential(
            ConvolutionalEmbedding(in_dim, out_dim, patch_size),
            *[TransformerBlock(out_dim, num_heads=4) for _ in range(depth)]
        )
    def forward(self, x):
        B, C, H0, W0 = x.shape

        # 阶段1
        x = self.stage1[0](x)  # [B, 64, H1, W1]
        H1 = (H0 + 2 * 0 - 1 * (self.patch_sizes[0] - 1) - 1) // self.patch_sizes[0] + 1
        W1 = (W0 + 2 * 0 - 1 * (self.patch_sizes[0] - 1) - 1) // self.patch_sizes[0] + 1
        x = x.permute(0, 2, 3, 1)  # [B, H1, W1, 64]
        x = self.stage1[1:](x)  # 处理TransformerBlock

        # 阶段2
        x = self.stage2[0](x.permute(0, 3, 1, 2))  # 恢复 [B, C, H, W]
        H2 = (H1 + 2 * 0 - 1 * (self.patch_sizes[1] - 1) - 1) // self.patch_sizes[1] + 1
        W2 = (W1 + 2 * 0 - 1 * (self.patch_sizes[1] - 1) - 1) // self.patch_sizes[1] + 1
        x = x.permute(0, 2, 3, 1)  # [B, H2, W2, 128]
        x = self.stage2[1:](x)

        # 阶段3
        x = self.stage3[0](x.permute(0, 3, 1, 2))
        H3, W3 = H2, W2  # 1x1卷积不改变尺寸
        x = self.stage3[1:](x.permute(0, 2, 3, 1))  # [B, H3, W3, 256]

        # 输出转换
        x = x.permute(0, 3, 1, 2)  # [B, 256, H3, W3]
        return self.final_conv(x)

# 修改后的网络架构（保持原接口不变）
class TRNetwork(nn.Module):
    def __init__(self, bands, n_classes, feature_dim=256):
        super().__init__()
        self.encoder = HierarchicalVisionTransformer(bands, feature_dim)
        self.classifier = Classifier(n_classes, feature_dim)  # 原分类器不变

    def forward(self, x):
        f = self.encoder(x)
        y, y_pseudo = self.classifier(f)
        return y, y_pseudo
#改transformer
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class newClassifier(nn.Module):
    def __init__(self, n_classes, feature_dim):
        super().__init__()
        self.main = nn.Linear(feature_dim, n_classes)
        self.pseudo = nn.Linear(feature_dim, n_classes)

    def forward(self, x):
        # 取序列的第一个元素作为特征（类似CLS token）
        cls_token = x[:, 0]
        y = self.main(cls_token)
        y_pseudo = self.pseudo(cls_token)
        return y, y_pseudo

class Transformer(nn.Module):
    def __init__(self, bands, n_classes, feature_dim=256,
                 depth=6, heads=8, dim_head=32, mlp_head=512,
                 dropout=0.1, num_channel=1):
        super().__init__()
        #-------------------------------方案2：---------------------
        # 新增 CNN 层处理空间维度
        self.cnn = nn.Sequential(
            nn.Conv2d(bands, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 压缩空间维度到 1×1
        )
        # 调整 proj 输入维度：CNN 输出是 [batch, 64, 1, 1] → 展平后是 [batch, 64]
        self.proj = nn.Linear(64, feature_dim)
        #----------------------------------------------------------
        # 输入投影：将波段数映射到特征维度
        #self.proj = nn.Linear(bands, feature_dim)

        # 分类token（用于最终分类）
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.classifier = newClassifier(n_classes, feature_dim)  # 原分类器不变
        # Transformer主体
        self.transformer = nn.ModuleList([])
        for _ in range(depth):
            self.transformer.append(nn.ModuleList([
                Residual(PreNorm(feature_dim, Attention(
                    feature_dim, heads=heads, dim_head=dim_head, dropout=dropout
                ))),
                Residual(PreNorm(feature_dim, FeedForward(
                    feature_dim, mlp_head, dropout=dropout
                )))
            ]))

        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(num_channel + 1, num_channel + 1, [1, 2], 1, 0))

    def forward(self, x):
        #方案1：将空间维度转化为序列 当作序列数据处理----------------------------------------
        # # 输入形状: [batch, bands, height, width] → 期望转成 [batch, seq_len, bands]
        # # 这里 seq_len = height × width = 24×24=576
        # batch, bands, height, width = x.shape
        # # 维度调整：将 [batch, 103, 24, 24] → [batch, 24×24, 103]
        # x = x.permute(0, 2, 3, 1).reshape(batch, height * width, bands)  # 关键：调整维度顺序并展平
        #----------------------------------------------------------------------------------

        #方案2：保留空间信息 在 Transformer 前加 CNN 处理空间维度，再将特征送入 Transformer
        # 1. CNN 处理空间维度
        x_cnn = self.cnn(x)  # 输出: [batch, 64, 1, 1]
        x_cnn = x_cnn.view(x_cnn.shape[0], -1)  # 展平: [batch, 64]

        # 2. 转成 Transformer 可处理的序列（这里序列长度为 1）
        x = x_cnn.unsqueeze(1)  # 变成 [batch, 1, 64]
        #-------------------------------------------------------------------------------

        # 输入投影
        x = self.proj(x)  # [batch, seq_len, feature_dim]

        # 添加分类token
        b = x.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)  # [batch, 1, feature_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [batch, seq_len+1, feature_dim]

        # Transformer处理
        for attn, ff in self.transformer:
            x = attn(x)
            x = ff(x)

        # 分类器输出
        y, y_pseudo = self.classifier(x)
        return y, y_pseudo
