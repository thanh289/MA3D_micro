# import os
# from PIL import Image, ImageDraw, ImageFont
#
# dir1 = "attn_vis"
# dir2 = "attn_vis_ori"
# output_dir = "merged"
#
# os.makedirs(output_dir, exist_ok=True)
#
# files = sorted([f for f in os.listdir(dir1) if f.endswith(".png")])
#
# # label settings
# label_width = 200
# font_size = 20
#
# try:
#     font = ImageFont.truetype("arial.ttf", font_size)
# except:
#     font = ImageFont.load_default()
#
# for f in files:
#     path1 = os.path.join(dir1, f)
#     path2 = os.path.join(dir2, f)
#
#     if not os.path.exists(path2):
#         print(f"Missing {f} in second folder")
#         continue
#
#     img1 = Image.open(path1).convert("RGB")
#     img2 = Image.open(path2).convert("RGB")
#
#     # resize to same width
#     target_width = min(img1.width, img2.width)
#     img1 = img1.resize((target_width, int(img1.height * target_width / img1.width)))
#     img2 = img2.resize((target_width, int(img2.height * target_width / img2.width)))
#
#     # total size
#     total_width = label_width + target_width
#     total_height = img1.height + img2.height
#
#     canvas = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))
#     draw = ImageDraw.Draw(canvas)
#
#     # paste images
#     canvas.paste(img1, (label_width, 0))
#     canvas.paste(img2, (label_width, img1.height))
#
#     # draw labels
#     draw.text((10, img1.height // 2 - 10), "POSTER", fill=(0, 0, 0), font=font)
#     draw.text((10, img1.height + img2.height // 2 - 10), "MA3D-Net (ours)", fill=(0, 0, 0), font=font)
#
#     # optional: draw separator line
#     draw.line((0, img1.height, total_width, img1.height), fill=(0, 0, 0), width=2)
#
#     canvas.save(os.path.join(output_dir, f))
#
# print("Done!")