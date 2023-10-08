import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import SRCNN

# モデルの読み込み
model = SRCNN()
model.load_state_dict(torch.load('super_resolution_model.pth'))
model.eval()

# 推論対象の画像を読み込み、前処理を適用
input_image_path = './frames_360p/frame_100.png'
input_image = Image.open(input_image_path)
preprocess = transforms.Compose([
    transforms.Resize((360, 640), interpolation=Image.BICUBIC),  # 360pサイズにリサイズ
    transforms.ToTensor()
])
input_tensor = preprocess(input_image)
input_tensor = input_tensor.unsqueeze(0)  # バッチ次元を追加

# 推論
with torch.no_grad():
    output_tensor = model(input_tensor)

# 推論結果をPIL Imageに変換
output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())

# 推論結果を保存
output_image.save('output_image2.png')