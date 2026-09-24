import torch
import torch.nn as nn
from torchvision import datasets, transforms
from model.train import SimpleCNN
# Load trained model
model = SimpleCNN()
model.load_state_dict(
    torch.load(
        "model/model.pt",
        map_location="cpu"
    )
)
model.eval()
dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)
image, label = dataset[0]
image = image.unsqueeze(0)
image.requires_grad = True
loss_function = nn.CrossEntropyLoss()
output = model(image)
loss = loss_function(
    output,
    torch.tensor([label])
)
model.zero_grad()
loss.backward()
# FGSM
epsilon = 0.1
perturbation = epsilon * image.grad.sign()
adversarial_image = image + perturbation
adversarial_image = torch.clamp(
    adversarial_image,
    0,
    1
)
# Prediction after attack
with torch.no_grad():

    original_output = model(image)

    adversarial_output = model(
        adversarial_image
    )
    original_prediction = torch.argmax(
        original_output,
        dim=1
    ).item()
    adversarial_prediction = torch.argmax(
        adversarial_output,
        dim=1
    ).item()
print("True label:", label)
print(
    "Original prediction:",
    original_prediction
)
print(
    "Adversarial prediction:",
    adversarial_prediction
)
print(
    "Epsilon:",
    epsilon
)
from torchvision.utils import save_image
save_image(
    image,
    "samples/original/original.png"
)
save_image(
    adversarial_image,
    "samples/adversarial/adversarial.png"
)
print("Images saved.")