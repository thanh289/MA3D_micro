import torch
import torch.nn as nn
import torch.nn.functional as F

class ThreeDMMEncoderCNN(nn.Module):
    """
    CNN encoder cho prior dạng spatial map (thay MLP khi input là flow map
    chưa pool, kiểu STSTNet/MEAN: [B, C, H, W] thay vì vector phẳng).

    Khác bản trước ở 2 điểm (theo tinh thần MEAN_Spot trong define_model.py gốc):
      1) Mỗi kênh (u, v, os) đi qua 1 nhánh Conv2d RIÊNG (không dùng chung 1
         Conv2d(in_channels,...) như xử lý ảnh RGB) -- vì u, v, os là 3 đại
         lượng vật lý khác nhau (không tương quan như RGB), tách nhánh cho
         phép mỗi kênh học bộ filter/độ sâu riêng (MEAN dùng 3, 3, 8 filter
         cho u, v, os) rồi mới concat lại ở lớp interpretation.
      2) 2 ROI (eyebrow, mouth) được encode riêng rồi CONCAT (không mean-pool
         nữa) -- tránh san bằng đóng góp của 2 vùng có tín hiệu AU rất khác
         nhau (vd ngạc nhiên chủ yếu ở lông mày, khinh bỉ chủ yếu ở miệng).

    input: x_3d shape [B, n_roi, C, H, W] (vd [B, 2, 3, 28, 28] cho
        2 ROI x (u, v, os) x 28x28) HOẶC [B, C, H, W] nếu chỉ có 1 ROI.
    Output: [B, embed_dim] -- cùng shape với ThreeDMMEncoder (MLP) nên có
    thể cắm thẳng vào ThreeDMMConditionGenerator không cần sửa gì thêm.
    """

    def __init__(self, in_channels=3, embed_dim=256, dropout=0.2,
                 roi_embed_dim=64, branch_channels=(3, 3, 8), n_roi=2):
        """
        n_roi: số ROI CỐ ĐỊNH sẽ nhận vào forward (vd 2: eyebrow + mouth).
            Khai báo tường minh ở __init__ (không lazy-infer lúc forward),
            vì optimizer (SAM trong train.py) được khởi tạo bằng
            model.parameters() NGAY SAU khi tạo model, TRƯỚC lần forward
            đầu tiên -- một layer tạo lười trong forward() sẽ bị bỏ sót
            khỏi optimizer.param_groups và không bao giờ được cập nhật
            gradient (silent bug, không raise exception, chỉ âm thầm học
            kém đi mà không ai biết).
        """
        super().__init__()
        assert in_channels == 3, (
            "Thiết kế 3-nhánh (u, v, os) yêu cầu in_channels=3; "
            "nếu bạn có số kênh khác, sửa branch_channels cho khớp."
        )
        c_u, c_v, c_os = branch_channels
        self.n_roi = n_roi

        def make_branch(out_ch):
            # kernel 5x5 + BN + pool, phỏng theo define_model.py::MEAN_Spot gốc
            return nn.Sequential(
                nn.Conv2d(1, out_ch, kernel_size=5, padding=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3, stride=3, ceil_mode=True),
                nn.Dropout2d(0.3),
            )

        self.branch_u  = make_branch(c_u)
        self.branch_v  = make_branch(c_v)
        self.branch_os = make_branch(c_os)

        merged_ch = c_u + c_v + c_os
        self.merge = nn.Sequential(
            nn.Conv2d(merged_ch, 8, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            # AdaptiveAvgPool thay vì tính tay kích thước còn lại sau 2 lần
            # pool -- tránh lỗi shape khi FLOW_MAP_SIZE thay đổi, vì Keras
            # padding='same' và PyTorch không chia hết giống hệt nhau.
            nn.AdaptiveAvgPool2d(4),
        )
        self.dropout = nn.Dropout(dropout)
        self.roi_proj = nn.Linear(8 * 4 * 4, roi_embed_dim)
        # final_proj khai báo NGAY ở đây (không lazy) vì n_roi đã biết trước
        self.final_proj = nn.Linear(n_roi * roi_embed_dim, embed_dim)

    def _encode_one_roi(self, x):  # x: [B, 3, H, W]
        u, v, os_ = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        f = torch.cat([self.branch_u(u), self.branch_v(v), self.branch_os(os_)], dim=1)
        f = self.merge(f)
        f = self.dropout(f.flatten(1))
        return self.roi_proj(f)  # [B, roi_embed_dim]

    def forward(self, x_3d):
        if x_3d.dim() == 5:
            B, R, C, H, W = x_3d.shape
            assert R == self.n_roi, (
                f"n_roi lúc __init__ ({self.n_roi}) không khớp dữ liệu thực "
                f"tế (R={R}). Sửa tham số n_roi khi khởi tạo MA3D/ThreeDMMFusion."
            )
            roi_embeds = [self._encode_one_roi(x_3d[:, r]) for r in range(R)]
            feat = torch.cat(roi_embeds, dim=1)   # [B, R * roi_embed_dim] -- concat, không mean
        else:
            assert self.n_roi == 1, "input 4D (1 ROI) nhưng n_roi khởi tạo != 1"
            feat = self._encode_one_roi(x_3d)      # [B, roi_embed_dim]

        return self.final_proj(feat)


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
                 x3d_mode="mlp", x3d_channels=3, x3d_n_roi=2):
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
        x3d_n_roi: số ROI CỐ ĐỊNH của flow map -- CHỈ dùng khi x3d_mode="cnn".
            Mặc định 2 (eyebrow + mouth, xem run_inference_flow.py). Phải
            khớp đúng n_roi thực tế trong flow_map.npy, nếu không sẽ assert
            fail ngay ở forward (xem ThreeDMMEncoderCNN.forward).
        """

        super().__init__()

        self.num_blocks = num_blocks
        self.x3d_mode = x3d_mode

        if x3d_mode == "cnn":
            self.encoder_3dmm = ThreeDMMEncoderCNN(
                in_channels=x3d_channels,
                embed_dim=embed_dim,
                n_roi=x3d_n_roi,
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