import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import SRCNN, FSRCNN, DRRN

# モデルの読み込み

round = int(input())
model = DRRN()

model.load_state_dict(torch.load(f'model/{model.__class__.__name__}_model_{round}.pth', 'cpu')['model_state_dict'])

model.to('cpu')
model.eval()

# 推論対象の画像を読み込み、前処理を適用
input_image_path = './frames_360p/frame_0.png'
input_image = Image.open(input_image_path)
preprocess = transforms.Compose([
    transforms.Resize((720, 1280), interpolation=Image.BICUBIC),  # 360pサイズにリサイズ
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
output_image.save(f'{model.__class__.__name__}output{round}_image.png')