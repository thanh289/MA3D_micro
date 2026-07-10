import torch
import torch.nn as nn
import torch.nn.functional as F

class ThreeDMMEncoderMEAN(nn.Module):
    """
    Khác ThreeDMMEncoderCNN cũ ở CẢ 2 điểm:
      1) 2 ROI (eyebrow, mouth) được GHÉP thành 1 ảnh composite duy nhất
         (42x42, nửa trên = eyebrow, nửa dưới = mouth) NGAY Ở BƯỚC TIỀN XỬ LÝ
         (xem run_inference_flow.py --feature_mode mean), không giữ tách
         riêng theo trục n_roi nữa.
      2) 3 kênh vật lý (u, v, os) vẫn tách nhánh riêng (điểm này giữ nguyên
         tinh thần từ bản cũ, và cũng đúng thiết kế MEAN_Spot gốc: mỗi kênh
         là 1 input (42,42,1) độc lập, filter 3/3/8).

    input: x_3d shape [B, 3, 42, 42] (kênh: u, v, os -- ĐÃ ghép sẵn 2 ROI
        thành composite từ bước tiền xử lý, không còn trục n_roi riêng).
    Output: [B, embed_dim] -- cùng interface ThreeDMMEncoder/ThreeDMMEncoderCNN
        cũ, cắm thẳng vào ThreeDMMConditionGenerator không cần sửa gì thêm.
    """

    def __init__(self, embed_dim=256, dropout_branch=0.3, dropout_head=0.2):
        super().__init__()

        def make_branch(out_ch):
            # kernel 5x5 pad=2 (tương đương padding='same' Keras với kernel
            # lẻ) + MaxPool k=3 s=3: 42 chia hết cho 3 -> đúng 14x14 giống
            # Keras MaxPooling2D(padding='same', strides=3) trên input 42x42,
            # không cần ceil_mode.
            return nn.Sequential(
                nn.Conv2d(1, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3, stride=3),
                nn.Dropout2d(dropout_branch),
            )

        self.branch_u  = make_branch(3)
        self.branch_v  = make_branch(3)
        self.branch_os = make_branch(8)

        merged_ch = 3 + 3 + 8  # = 14
        self.merge_conv = nn.Conv2d(merged_ch, 8, kernel_size=5, padding=2)
        self.merge_pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 14/2=7, chia hết sạch
        self.dropout_head = nn.Dropout(dropout_head)
        self.proj = nn.Linear(8 * 7 * 7, embed_dim)  # 392 -> embed_dim

    def forward(self, x_3d):  # x_3d: [B, 3, 42, 42]
        u, v, os_ = x_3d[:, 0:1], x_3d[:, 1:2], x_3d[:, 2:3]
        fu = self.branch_u(u)
        fv = self.branch_v(v)
        fo = self.branch_os(os_)
        merged = torch.cat([fu, fv, fo], dim=1)          # [B, 14, 14, 14]
        m = F.relu(self.merge_conv(merged))               # [B, 8, 14, 14]
        m = self.merge_pool(m)                             # [B, 8, 7, 7]
        m = self.dropout_head(m.flatten(1))                # [B, 392]
        return self.proj(m)                                # [B, embed_dim]


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
                 x3d_mode="mlp"):
        """
        x3d_dim: chiều vector prior 3D đầu vào -- CHỈ dùng khi x3d_mode="mlp"
            (358 cho SMIRK cũ, 16 cho flow-pooled-vector).
        x3d_hidden_dim: chiều hidden của ThreeDMMEncoder (MLP) -- CHỈ dùng khi
            x3d_mode="mlp". None -> mặc định 512. Nên set nhỏ (vd 64) khi
            x3d_dim nhỏ để tránh bottleneck ngược.
        x3d_mode: "mlp" (mặc định, giữ tương thích ngược với SMIRK/flow-pooled-vector)
            hoặc "mean" (composite flow map [B,3,42,42], port kiến trúc MEAN_Recog
            -- xem ThreeDMMEncoderMEAN). Thay thế "cnn" (ThreeDMMEncoderCNN,
            tách 2 ROI riêng) đã bỏ sau khi so sánh thực nghiệm 3 kiến trúc
            độc lập cho thấy MEAN_Recog tốt hơn.
        """

        super().__init__()

        self.num_blocks = num_blocks
        self.x3d_mode = x3d_mode

        if x3d_mode == "mean":
            self.encoder_3dmm = ThreeDMMEncoderMEAN(embed_dim=embed_dim)
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