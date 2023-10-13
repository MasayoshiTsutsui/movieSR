import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from model import SRCNN, FSRCNN, DRRN
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from typing import Tuple




# ハイパーパラメータ
learning_rate: float = 0.001
batch_size: int = 1
num_epochs: int = 1000

# データセットのディレクトリ
data_dir_360p: str = './frames_360p'
data_dir_720p: str = './frames_720p'

model_savedir: str = './model'

# データセットの前処理（Bicubic補間とテンソル化）
preprocess: transforms.Compose = transforms.Compose([
    transforms.Resize((720, 1280), interpolation=Image.BICUBIC),
    transforms.ToTensor()
])

class SuperResolutionDataset(Dataset):
    def __init__(self, data_dir_360p: str, data_dir_720p: str, transform: transforms.Compose) -> None:
        self.data_dir_360p = data_dir_360p
        self.data_dir_720p = data_dir_720p
        self.transform = transform
        self.file_list = os.listdir(data_dir_360p)[:1]

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name = self.file_list[idx]
        img_360p = Image.open(os.path.join(self.data_dir_360p, img_name))
        img_720p = Image.open(os.path.join(self.data_dir_720p, img_name))
        
        if self.transform:
            img_360p = self.transform(img_360p)
            img_720p = self.transform(img_720p)

        return img_360p, img_720p

def train(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, num_epochs: int, device: torch.device) -> None:

    # log the train loss per epoch to csv file
    log_file = open('train_log.csv', 'w')
    log_file.write('epoch,train_loss\n')

    model.train()
    
    for epoch in range(num_epochs):
        for data in data_loader:
            inputs, targets = data
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            #print(f'Batch Loss: {loss.item()}')

        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {loss.item()}')
        log_file.write(f'{epoch + 1},{loss.item()}\n')

        if (epoch + 1) % 100 == 0:
            model_path = f'{model_savedir}/{model.__class__.__name__}_model_{epoch}.pth'

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, model_path)

# データローダーの作成
dataset = SuperResolutionDataset(data_dir_360p, data_dir_720p, transform=preprocess)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

model = DRRN()
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train(model, data_loader, criterion, optimizer, num_epochs, device)

