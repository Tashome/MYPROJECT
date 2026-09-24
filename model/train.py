import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 0.001
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=transform
)
test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.network(x)
model = SimpleCNN()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(
        f"Epoch {epoch + 1}/{EPOCHS}, "
        f"Loss: {total_loss / len(train_loader):.4f}"
    )
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()
accuracy = 100 * correct / total
print(f"Test accuracy: {accuracy:.2f}%")
torch.save(
    model.state_dict(),
    "model/model.pt"
)
print("Model saved to model/model.pt")