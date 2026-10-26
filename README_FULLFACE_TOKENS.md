# ROI motion tokens + full-face context (v1)

Bản cập nhật dựa trên ZIP `MA3D-Net(1).zip` người dùng gửi. Giải nén gói update
vào thư mục gốc repo, ghi đè đúng đường dẫn. Gói chỉ chứa các file thêm/sửa;
các file khác, pretrained checkpoint và dữ liệu vẫn lấy từ repo hiện tại.
Không dùng branch GitHub cũ làm nguồn cho bản cập nhật này.

## Luồng model

- ROI flow [B,R,3,28,28] -> CNN chung -> [B,R,512,g,g]. Giữ grid mỗi ROI,
  bỏ bước expand/pack thành 7x7. Mặc định g=3.
- Tọa độ mỗi motion token = tâm ô trên lưới ROI chuẩn hóa, ánh xạ qua bbox.
  Đây là tọa độ đại diện cho bin, không phải receptive field chính xác của CNN.
- Apex toàn mặt -> MobileFaceNet frozen cả parameters và BatchNorm -> [B,512,7,7].
  Không crop geometry. Geometry cross-attention: Q=motion+tọa độ,
  K=geometry+tọa độ lưới toàn mặt, V=geometry. Math SDPA backend chỉ cho layer này.
- Context theo motion token -> sinh gamma/beta -> các Spatial FiLM blocks cũ.
  FiLM vẫn dùng convolution cục bộ trong từng ROI g x g (fold ROI vào batch),
  tránh tạo láng giềng giả giữa các vùng. Geometry attention tính một lần/phase,
  sinh context cho toàn bộ các blocks; không thêm một attention cho mỗi block.
- Onset toàn mặt -> shallow encoder cũ -> [B,49,512], tính một lần/sample.
- Motion đã modulate -> [B,R*g*g,512] -> một pyramid cross-interaction/phase
  với appearance [B,49,512]. Giữ pyramid scales/depth/MLP/token-mixing conv.
- Motion summary -> SE cũ -> AU embedding tùy chọn -> classifier.
- Không gọi `ROIContextAggregator` trong `roi_alignment.py` nữa. File đó có thể
  còn trong repo để tham khảo nhưng không tham gia kiến trúc mới.

### Thay đổi quan trọng trong cross-interaction

Đường chạy mới bật `query_aligned=True`: motion Q / appearance K,V tạo update
cho motion; appearance Q / motion K,V tạo update cho appearance. Hai chuỗi
không cần bằng chiều dài. Bản cũ dùng query nhánh đối diện rồi cộng output vào
nhánh đang cập nhật; chỉ chạy được khi số tokens bằng nhau. Vì vậy đây là
thay đổi cơ chế update có chủ đích, không chỉ sửa reshape.
Constructor HyVisionTransformer mặc định query_aligned=False vẫn giữ đường
legacy để đối chiếu; MA3D mới luôn bật True.

## Dữ liệu và cách chạy

Notebook `4dme-preprocess-dlci-net.ipynb` đã có bbox/metadata cần thiết: không
cần sửa notebook hoặc preprocess lại nếu output hiện tại có các file đó.
Đổi g không cần preprocess; đổi R=3/6 phải dùng dataset được tạo với ROI_MODE
merged3/split6 tương ứng. Không đổi mỗi flag n_roi trên cùng bộ flow 6 ROI.
Augment đồng bộ, normalized-flow flip, AU và random erasing được giữ nguyên.

Giữ lệnh train/notebook hiện có; thêm các options mới nếu muốn thay mặc định.
Ví dụ **phần cấu hình model**, ghép vào lệnh hiện có, không phải lệnh chạy đầy đủ:

```bash
--motion_backbone cnn --n_roi 6 --num_film_blocks 3 \
--roi_pool_size 3 --geometry_heads 4 \
--geometry_conditioning cross_attention \
--use_rise_fall --rise_fall_mode decision_level
```

Rise-only: bỏ `--use_rise_fall` (CLI store_true, không viết `--use_rise_fall False`).
Decision-level: fall đóng góp loss khi train, inference không chạy fall.
Mặc định CLI `rise_fall_mode=feature_gate` của file gốc được giữ nguyên,
vì vậy nhớ ghi rõ `--rise_fall_mode decision_level` khi bật hai phase.
Feature_gate được giữ tương thích nhưng không nằm trong kế hoạch ablation này.
RMT không có implementation trong ZIP; model mới báo lỗi rõ nếu chọn RMT.

Không đổi defaults cũ của lr/batch/epochs/loss/etc. Defaults trong train.py
vẫn khác cấu hình notebook của bạn; tiếp tục dùng explicit flags từ lệnh cũ.
Không có notebook train trong lần đính kèm này nên chưa chỉnh notebook train.

## Các cấu hình để kiểm chứng

| Tham số | Ban đầu / giữ theo run của bạn | Giá trị để thử | Ý nghĩa |
|---|---|---|---|
| `--n_roi` | theo dataset | 3, 6 | Phân vùng motion; cần preprocess tương ứng |
| `--roi_pool_size` (mới) | 3 | 2, 3, 4, 5, 7 | g x g tokens/ROI; N=R*g*g |
| `--num_film_blocks` | giữ flag cũ | 3, 5 | Số blocks modulation; weights chung giữa ROI/phases |
| `--use_rise_fall` | giữ flag cũ | bỏ/bật | Rise-only vs rise+fall; khi bật ghi decision_level |
| `--geometry_heads` (mới) | 4 | 1, 2, 4, 8 | Heads của geometry attention, không phải số ROI |
| `--geometry_conditioning` (mới) | cross_attention | cross_attention, global_mean, none | Token-conditioned FiLM, pooled-global/channel FiLM, bỏ geometry+FiLM |
| `--no_roi_position` (mới) | không bật | bỏ/bật | Bỏ explicit xy encoding; learned pyramid positions vẫn còn |
| `--use_au` | giữ flag cũ | bỏ/bật | Có/không AU input; dùng đúng AU_DIM của dataset |
| `--aux_loss_weight` | giữ flag cũ | ví dụ 0.05, 0.1, 0.2 | Trọng số fall loss, chỉ khi bật rise/fall |
| `--model_type` | giữ flag cũ | small, base, large | Depth pyramid 4/6/8; appearance depth vẫn 2 |

Khuyến nghị chạy một baseline, rồi thay từng yếu tố: geometry mode -> explicit
coordinates -> ROI/grid -> FiLM depth -> rise/fall -> heads. Không nhất thiết
quét mọi tổ hợp. Giữ folds/seeds/training protocol, ghi UF1/UAR, số tham số,
VRAM và thời gian mỗi epoch. Train lại từ đầu cho từng cấu hình.

Lưu ý diễn giải:
- Đổi g hoặc R làm thay đổi N và số tham số của token-mixing convolutions trong
  pyramid gốc, không chỉ tăng độ phân giải. 6 ROI với g=3 cho 54 motion tokens;
  g=5 cho 150. Các lưới lớn có thể chậm/tốn bộ nhớ đáng kể.
- global_mean dùng LandmarkModulationFusion channel-only cũ; đây là ablation
  cả kiểu conditioning, không phải so sánh cùng parameter count.
- none bỏ geometry conditioning VÀ FiLM; không phải chỉ bỏ attention.
- no_roi_position bỏ shared face-coordinate encoding ở geometry Q/K và trước
  pyramid, không bỏ learned positional embeddings/token-mixing của pyramid.
- Geometry attention cho phép học liên hệ, không bảo đảm alignment pixel.
- Full-face context chỉ encode một lần nhưng FiLM vẫn xử lý từng ROI trên grid
  nhỏ. Không tuyên bố tốc độ hay accuracy trước khi benchmark GPU/dataset thật.

## Checkpoint và kiểm tra

Kiến trúc mới cần train lại; không resume checkpoint bản packed hay all-ROI.
LOSO checkpoint mới lưu architecture/model_config và kiểm tra khớp khi resume.
Transfer cũng lưu/so sánh model_config, đã nối thêm các options mới. Chọn thư
mục log/checkpoint riêng cho từng cấu hình, không ghi đè kết quả cũ.
MobileFaceNet vẫn lấy pretrained checkpoint ở đường dẫn cũ:
`checkpoints/mobilefacenet_model_best.pth_o1.tar`.
`eval.py` cũ dành cho dataset ảnh tĩnh không được chuyển thành evaluator mới;
dùng validation LOSO hoặc cross-dataset trong entrypoints đã cập nhật.

```bash
python -m unittest discover -s tests -v
```

10 tests CPU pass: actual model forward/backward (random MobileFaceNet weights,
synthetic inputs), 3/6 ROI, FiLM 3/5, grid/head variants, freeze BN/grad,
fall-free eval, bbox reflection coordinates, geometry ablations, checkpoint
roundtrip, feature_gate smoke, CLI wiring và preservation of legacy pyramid.
Test legacy comparison sẽ skip khi không có thư mục source gốc `../original`.
Chưa train benchmark thật hoặc kiểm chứng trực tiếp GPU trong phiên này.

## File thay đổi

- models/MA3D.py: nối kiến trúc mới, giữ freeze/eval, bỏ aggregator cũ khỏi đường chạy.
- models/fullface_context.py: geometry attention, coordinate helper, CLI options mới.
- models/motion_encoder.py: xuất ROI grid g x g trực tiếp.
- models/hyp_crossvit.py: query-aligned cross-attention với hai độ dài khác nhau.
- models/ThreeDMM_Adaptive.py: chỉ cập nhật comment/documentation; phép tính giữ nguyên.
- train.py: options mới và kiểm tra cấu hình resume.
- train_transfer.py: options mới và config checkpoint.
- cross_dataset.py: truyền roi_boxes vào model ở evaluation.
- tests/test_fullface_tokens.py và hướng dẫn này.
