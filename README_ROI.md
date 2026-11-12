# MA3D — motion-first ROI baseline

Bản sửa theo ưu tiên **hướng 1 → hướng 3** trong `DICL-context(1).txt`.

## Những gì đã triển khai

- Mặc định `motion_aggregation=roi_transformer`: dùng chung CNN cho từng ROI,
  tạo feature `[B,R,128,4,4]` (có thể chọn 192 chiều), pooling có trọng số học được
  riêng trong mỗi ROI, thêm embedding định danh ROI và tâm/kích thước box,
  qua 2 transformer block nhỏ giữa các ROI, mean pooling rồi classifier.
  Có thể chọn 1 block. Mặc định 3 ROI, không geometry, appearance, AU hay fall.
- Bộ tổng hợp motion luôn tồn tại khi bật/tắt geometry. Không còn phụ thuộc
  vào nhánh appearance hay pyramid trong mô hình mới.
- `geometry_conditioning=roi_residual`: lấy feature MobileFaceNet trong đúng
  bounding box bằng `sample_rois`, một block modulation, đường motion residual
  trực tiếp và hệ số học được khởi tạo 0.01. Backbone đóng băng cả trọng số lẫn
  thống kê BatchNorm. Đây là tương ứng theo vùng, không phải theo pixel vật lý.
- Tắt random erasing trên apex mặc định; `--erase_apex` để thử lại. Color jitter
  đồng bộ và erasing trên onset/offset giữ nguyên. Flip vẫn đồng bộ RGB/motion/box.
- Fall auxiliary tùy chọn dùng chung encoder, aggregator và classifier. Fall chỉ
  tham gia loss khi train; lúc inference chỉ cần rise. Không dùng cosine gate
  sau ReLU trong kiến trúc mới.
- Tham số mới có ở cả `train.py` và `train_transfer.py`. Checkpoint lưu cấu hình;
  `eval.py` dựng lại đúng kiến trúc và lấy backbone từ checkpoint đầy đủ.
- Giữ checkpoint best/last sau khi kết thúc fold; last được lưu mỗi epoch.
  Bản cũ tự xóa các file này khi hoàn thành fold.

Luồng chính nằm ở `models/compact_roi.py`; pooling, ROI transformer và geometry
residual ở `models/roi_alignment.py`. `models/MA3D.py` chọn đường mới hoặc legacy.

## Dữ liệu và notebook

**Không cần sửa hoặc chạy lại notebook chỉ để đổi mô hình 3 ROI.** Bản notebook
đã gửi tạo đủ `flow_map.npy`, `roi_boxes.npy`, `motion_meta.json` cho kiến trúc này.
Nó đang dùng `NUM_CLASSES=5`, `ROI_MODE="merged3"`, `MOTION_MODE="pixeldiff"`.
Lưới 4×4 là feature grid bên trong model; input motion vẫn `[R,3,28,28]`.
Model mới nhận được cả dữ liệu pixeldiff lẫn optical flow theo schema hiện có.

Muốn thử 6 ROI, đổi **duy nhất** `ROI_MODE="split6"` ở cell đầu notebook và chạy
lại pipeline. Dùng thư mục output 6 ROI cùng `--n_roi 6`. Không đổi tên/copy
motion 3 ROI thành 6 ROI. Giữ nguyên class protocol, motion mode, seed, GAMDSS,
sampler, augmentation và hyperparameter khi so sánh; kiểm tra cohort
`saved_folders` của hai lần preprocessing có cùng mẫu.

## Chạy trên Kaggle hoặc môi trường hiện có

Giải nén ZIP, chạy từ thư mục chứa `train.py`. Dùng môi trường PyTorch/torchvision
phù hợp của bạn cùng numpy, Pillow, scikit-learn, matplotlib, tqdm, wandb.
`pytest` chỉ cần khi chạy bộ kiểm tra. Không cần checkpoint landmark cho baseline.

Baseline 3 ROI, 5 lớp (đường dẫn khớp cấu hình notebook đã gửi):

```bash
python train.py \
  --data_type 4DME_MOTION \
  --data_root /kaggle/working/4dme_ma3d_motion_3ROI_pixeldiff \
  --num_classes 5 --n_roi 3 \
  --motion_aggregation roi_transformer --motion_dim 128 \
  --roi_pool_size 4 --roi_depth 2 \
  --geometry_conditioning none --appearance_fusion none \
  --seed 42 --epochs 100 --batch_size 8 \
  --resume_dir runs/roi3/checkpoints --backup_dir runs/roi3/backup \
  --log_file roi3.txt
```

LR, scheduler, loss schedule và class protocol không được đổi để tránh trộn
thêm yếu tố vào phép thử kiến trúc. Lệnh trên dùng LR cũ 1e-5 của `train.py`;
đây chưa phải LR đã được tối ưu cho kiến trúc mới. Nếu bạn đang dùng LR/sampler/
GAMDSS khác, truyền cấu hình đó đồng nhất cho mọi ablation. Chỉ thêm
`--use_gamdss` khi muốn train trên bộ dữ liệu reselection; val vẫn dùng bản gốc.

Các phép thử tiếp theo, giữ nguyên các tham số còn lại và dùng thư mục run riêng:

| Phép thử | Tham số thay đổi |
|---|---|
| Motion 6 ROI | `--n_roi 6 --data_root /kaggle/working/4dme_ma3d_motion_6ROI_pixeldiff` |
| Motion 192 chiều | `--motion_dim 192` |
| Thêm geometry cùng vùng | `--geometry_conditioning roi_residual --num_film_blocks 1 --geometry_scale_init 0.01 --landmark_checkpoint checkpoints/mobilefacenet_model_best.pth_o1.tar` |
| Fall auxiliary | `--use_rise_fall --rise_fall_mode decision_level --aux_loss_weight 0.2` |
| Apex erasing ablation | `--erase_apex` |

Geometry phải có checkpoint MobileFaceNet pretrained thật. ZIP gốc không chứa
checkpoint này. Không dùng trọng số ngẫu nhiên để đánh giá chất lượng geometry.

Chạy một fold trước: thêm `--loso_debug_subject SUBJECT_ID` với ID đúng trong tên
folder dữ liệu. Tiếp tục run đang có bằng cùng cấu hình và `--resume`. Không resume
checkpoint kiến trúc cũ vào mô hình mới; phải train mới. `--model_type` và
`--landmark_embed_dim` là tham số legacy, không điều khiển kích thước compact model.

Đánh giá một checkpoint trên subject đã giữ ngoài train:

```bash
python eval.py \
  --ckpt_path runs/roi3/backup/SUBJECT_ID_4dme_best.pth \
  --data_type 4DME_MOTION \
  --data_root /kaggle/working/4dme_ma3d_motion_3ROI_pixeldiff \
  --subject SUBJECT_ID
```

`--all_samples` dành cho một tập target độc lập; không dùng nó để báo cáo LOSO
trên cả các subject từng xuất hiện trong train. Cơ chế chọn best epoch theo UF1
held-out trong `train.py` được giữ nguyên, chưa chuyển sang nested validation.

Cross-dataset chỉ dùng **protocol 3 lớp** như code gốc; dữ liệu 5 lớp của notebook
hiện tại không phải input hợp lệ cho lệnh này. Ví dụ khi đã có hai tập 3 lớp:

```bash
python train_transfer.py \
  --source 4dme --source_root /path/to/4dme_3class \
  --target casme2=/path/to/casme2_3class \
  --n_roi 3 --motion_dim 128 --roi_pool_size 4 --roi_depth 2 \
  --geometry_conditioning none --appearance_fusion none \
  --output_dir transfer_runs --run_name roi_motion_seed42
```

Kết quả primary vẫn là final checkpoint; best-target là oracle như code gốc.
Transfer dùng mặc định LR/weight decay riêng của script, không giống LOSO.

## Giữ mô hình cũ để đối chiếu

`--motion_aggregation legacy_pyramid` giữ đường model cũ. Để tái tạo cấu hình
geometry + appearance trước đây, truyền rõ:

```text
--motion_aggregation legacy_pyramid --appearance_fusion cross_attention
--geometry_conditioning cross_attention --roi_pool_size 3 --num_film_blocks 5
```

Nếu dùng fall gate cũ, truyền thêm `--use_rise_fall --rise_fall_mode feature_gate`.
Đường legacy vẫn có ablation appearance=none/mean pooling như trước; không nên
coi đó là phép tách riêng tác dụng appearance. CLI mới mặc định compact nên các
lệnh cũ cần chọn legacy rõ ràng. `eval.py` hỗ trợ checkpoint cũ có `model_config`;
resume train cũ không được tự động chuyển đổi.

## Chưa triển khai trong vòng này

Distance-biased full-face attention; direction-aware rise–fall với signed flow;
learned-motion và pretraining theo MoExt. Không đổi preprocessing min–max và
không coi các feature sau ReLU là hướng optical flow vật lý. Đây là các bước
sau khi có kết quả baseline, đúng thứ tự ưu tiên trong context.

## Kiểm tra

```bash
python -m pytest -q tests
```

Bộ kiểm tra bao gồm gradient, 3/6 ROI, pooling weights, geometry locality và
residual identity, đóng băng backbone, lưu/nạp portable checkpoint, fall train-only,
augmentation, CLI, loader LOSO/transfer và train/resume trên dữ liệu giả lập.
Chưa chạy trên dataset thật hoặc GPU; các kiểm tra này không chứng minh tăng UF1/UAR.
