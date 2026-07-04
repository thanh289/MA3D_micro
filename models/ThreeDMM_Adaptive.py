import torch
import torch.nn as nn
import torch.nn.functional as F

class ThreeDMMEncoderCNN(nn.Module):
    """
    CNN encoder cho prior dạng spatial map (thay MLP khi input là flow map
    chưa pool, kiểu STSTNet: [B, C, H, W] thay vì vector phẳng).

    input: x_3d shape [B, n_roi, C, H, W] (vd [B, 2, 3, 28, 28] cho
        2 ROI x (u, v, os) x 28x28) HOẶC [B, C, H, W] nếu đã gộp n_roi vào C
        từ bước tiền xử lý. Hai ROI được gộp vào batch-dim rồi conv chung
        (weight-sharing giữa các ROI) thay vì mỗi ROI một nhánh riêng --
        đơn giản hơn và ít tham số hơn, hợp lý khi n_roi nhỏ (2) và data ít.

    Output: [B, embed_dim] -- cùng shape với ThreeDMMEncoder (MLP) nên có
    thể cắm thẳng vào ThreeDMMConditionGenerator không cần sửa gì thêm.
    """

    def __init__(self, in_channels=3, embed_dim=256, dropout=0.2):
        super().__init__()

        # rất nông có chủ đích (giống tinh thần STSTNet ~0.00167M param) --
        # dataset nhỏ, không cần/không nên đi sâu.
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2),                       # 28x28 -> 14x14

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),               # global pool -> [B*n_roi, 32, 1, 1]
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(32, embed_dim)

    def forward(self, x_3d):
        if x_3d.dim() == 5:
            # [B, n_roi, C, H, W] -> gộp n_roi vào batch để weight-share
            # giữa các ROI, rồi mean-pool embedding của các ROI lại theo B
            B, R, C, H, W = x_3d.shape
            x = x_3d.view(B * R, C, H, W)
            feat = self.conv(x).flatten(1)         # [B*R, 32]
            feat = feat.view(B, R, -1).mean(dim=1)  # [B, 32] -- gộp ROI
        else:
            # [B, C, H, W] -- đã gộp ROI vào C từ tiền xử lý
            feat = self.conv(x_3d).flatten(1)      # [B, 32]

        feat = self.dropout(feat)
        return self.proj(feat)


class ThreeDMMEncoder(nn.Module):

    def __init__(self, input_dim=358, embed_dim=256, dropout=0.2, hidden_dim=None):
        """
        input_dim: chiều của vector prior đầu vào.
            - 358  : SMIRK prior cũ (exp50 + jaw3 + eyelid2 + pose3 + shape300)
            - 16   : flow prior mới (2 ROI x 8 stat:
                     mean(mag), std(mag), mean(|u|), mean(|v|),
                     mean(u), mean(v), mean(os), std(os))
        hidden_dim: chiều lớp ẩn. Mặc định 512 (phù hợp input_dim lớn kiểu SMIRK).
            Khi input_dim nhỏ (vd flow prior 16-dim), nên truyền hidden_dim nhỏ hơn
            (vd 64) để tránh bottleneck ngược (input << hidden) làm mất ý nghĩa
            "prior nhỏ, đặc trưng" của flow vector.
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 512

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x_3d):
        return self.net(x_3d)

class ThreeDMMConditionGenerator(nn.Module):

    def __init__(self, embed_dim, num_blocks, channels):
        super().__init__()
        self.num_blocks = num_blocks
        self.channels = channels

        self.net = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_blocks * channels * 2)
        )

    def forward(self, embed):
        B = embed.shape[0]

        gamma_beta = self.net(embed)
        gamma_beta = gamma_beta.view(
            B, self.num_blocks, 2, self.channels
        )

        gamma = gamma_beta[:, :, 0]
        beta = gamma_beta[:, :, 1]

        return gamma, beta

class SEBlock(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()

        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.net(x)
        return x * scale

class ThreeDMMAdaptiveBlock(nn.Module):

    def __init__(self, channels, bottleneck_ratio=4, use_se=False, dropout=0.1):
        super().__init__()
        self.use_se = use_se
        hidden = channels // bottleneck_ratio

        # Bottleneck convs
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.conv2 = nn.Conv2d(hidden, hidden, 3, padding=1)
        self.conv3 = nn.Conv2d(hidden, channels, 1)

        self.norm1 = nn.InstanceNorm2d(channels, affine=False)
        self.norm2 = nn.InstanceNorm2d(hidden, affine=False)

        self.dropout = nn.Dropout2d(dropout)

        if use_se:
            self.se = SEBlock(channels)

    def forward(self, x, gamma, beta):

        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)

        # FiLM modulation
        out = self.norm1(x)
        out = out * (1 + gamma) + beta
        out = F.leaky_relu(out, 0.2)

        # Bottleneck
        out = self.conv1(out)

        out = self.norm2(out)
        out = F.leaky_relu(out, 0.2)

        out = self.conv2(out)
        out = self.dropout(out)

        out = F.leaky_relu(out, 0.2)
        out = self.conv3(out)

        if self.use_se:
            out = self.se(out)

        return x + out



class ThreeDMMFusion(nn.Module):

    def __init__(self, feat_nc=512, embed_dim=256, num_blocks=5,
                 x3d_dim=358, x3d_hidden_dim=None,
                 x3d_mode="mlp", x3d_channels=3):
        """
        x3d_dim: chiều vector prior 3D đầu vào -- CHỈ dùng khi x3d_mode="mlp"
            (358 cho SMIRK cũ, 16 cho flow-pooled-vector).
        x3d_hidden_dim: chiều hidden của ThreeDMMEncoder (MLP) -- CHỈ dùng khi
            x3d_mode="mlp". None -> mặc định 512. Nên set nhỏ (vd 64) khi
            x3d_dim nhỏ để tránh bottleneck ngược.
        x3d_mode: "mlp" (mặc định, giữ tương thích ngược với SMIRK/flow-pooled-vector)
            hoặc "cnn" (flow map thô chưa pool, kiểu STSTNet -- xem ThreeDMMEncoderCNN).
        x3d_channels: số kênh input cho CNN encoder -- CHỈ dùng khi x3d_mode="cnn".
            Vd 3 nếu x_3d shape [B, n_roi, 3, H, W] (u, v, os mỗi ROI).
        """

        super().__init__()

        self.num_blocks = num_blocks
        self.x3d_mode = x3d_mode

        if x3d_mode == "cnn":
            self.encoder_3dmm = ThreeDMMEncoderCNN(
                in_channels=x3d_channels,
                embed_dim=embed_dim,
            )
        else:
            # NOTE: trước đây gọi ThreeDMMEncoder(embed_dim) theo vị trí (positional).
            # Sau khi thêm input_dim làm tham số đầu tiên của ThreeDMMEncoder, phải gọi
            # rõ tên tham số ở đây để tránh truyền nhầm embed_dim vào input_dim.
            self.encoder_3dmm = ThreeDMMEncoder(
                input_dim=x3d_dim,
                embed_dim=embed_dim,
                hidden_dim=x3d_hidden_dim,
            )

        self.film_generator = ThreeDMMConditionGenerator(
            embed_dim,
            num_blocks,
            feat_nc
        )

        self.blocks = nn.ModuleList([
            ThreeDMMAdaptiveBlock(feat_nc, use_se=True),
            ThreeDMMAdaptiveBlock(feat_nc, use_se=True),
            ThreeDMMAdaptiveBlock(feat_nc, use_se=True),
            ThreeDMMAdaptiveBlock(feat_nc, use_se=True),
            ThreeDMMAdaptiveBlock(feat_nc, use_se=True)
        ])

    def forward(self, x, x_3d):
        mm_embed = self.encoder_3dmm(x_3d)
        gamma, beta = self.film_generator(mm_embed)

        for i, block in enumerate(self.blocks):
            x = block(x, gamma[:, i], beta[:, i])

        return x