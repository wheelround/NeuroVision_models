import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Первый сверточный слой
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32,
                               kernel_size=3, stride=1, padding=1)
        # Второй сверточный слой
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64,
                               kernel_size=3, stride=1, padding=1)
        # Активация ReLU
        self.relu = nn.ReLU()
        # Пулинг
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # Полносвязные слои
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 классов (например, для CIFAR-10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)
        # Преобразование тензора в вектор для полносвязного слоя
        out = out.view(-1, 64 * 7 * 7)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out
