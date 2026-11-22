import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# -------------------------------
# 🚚 Data loading
# -------------------------------
def get_dataloaders(train_size=10000, test_size=1000, batch_size=256):
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    full_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    full_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_subset = torch.utils.data.Subset(full_train, range(train_size))
    test_subset = torch.utils.data.Subset(full_test, range(test_size))

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


# -------------------------------
# 🧠 CNN model
# -------------------------------
class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            # Block 1: 3 → 32 channels, 32x32 → 16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            # Block 2: 64 → 128 channels, 16x16 → 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x


# -------------------------------
# 🏋️ Training function
# -------------------------------
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    return running_loss / len(dataloader.dataset)


# -------------------------------
# 🧪 Evaluation function
# -------------------------------
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# -------------------------------
# 🧵 Training loop
# -------------------------------
def train_model(model, trainloader, testloader, device, epochs=20):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Cosine annealing: smoothly decays LR to near-zero
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, trainloader, optimizer, loss_fn, device)
        test_acc = evaluate(model, testloader, device)
        scheduler.step()

        if test_acc > best_acc:
            best_acc = test_acc

        print(f"Epoch {epoch+1:02d}/{epochs} | "
              f"LR: {scheduler.get_last_lr()[0]:.4f} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Acc: {test_acc*100:.2f}% | "
              f"Best: {best_acc*100:.2f}%")

    return best_acc


# -------------------------------
# 🚀 Main
# -------------------------------
def main():
    trainloader, testloader = get_dataloaders(train_size=10000, test_size=1000)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MyCNN().to(device)
    best_acc = train_model(model, trainloader, testloader, device, epochs=12)
    print(f"\nFinal best test accuracy: {best_acc*100:.2f}%")


if __name__ == '__main__':
    main()