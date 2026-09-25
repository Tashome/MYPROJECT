import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ============================================================
# 1. Define the neural network
# ============================================================

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


# ============================================================
# 2. Load the trained model
# ============================================================

model = SimpleCNN()

model.load_state_dict(
    torch.load(
        "model/model.pt",
        map_location="cpu"
    )
)

model.eval()


# ============================================================
# 3. Load MNIST test dataset
# ============================================================

transform = transforms.ToTensor()

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)


# ============================================================
# 4. FGSM function
# ============================================================

def fgsm_attack(image, epsilon, gradient):

    perturbation = epsilon * gradient.sign()

    adversarial_image = image + perturbation

    adversarial_image = torch.clamp(
        adversarial_image,
        0,
        1
    )

    return adversarial_image


# ============================================================
# 5. Select an image
# ============================================================

image, label = next(iter(test_loader))

image.requires_grad = True


# ============================================================
# 6. Make the original prediction
# ============================================================

output = model(image)

original_prediction = output.argmax(
    dim=1
).item()


# ============================================================
# 7. Calculate the loss
# ============================================================

loss_function = nn.CrossEntropyLoss()

loss = loss_function(
    output,
    label
)


# ============================================================
# 8. Calculate gradient
# ============================================================

model.zero_grad()

loss.backward()

gradient = image.grad.data


# ============================================================
# 9. Create adversarial image
# ============================================================

epsilon = 0.1

adversarial_image = fgsm_attack(
    image,
    epsilon,
    gradient
)


# ============================================================
# 10. Make prediction on adversarial image
# ============================================================

with torch.no_grad():

    adversarial_output = model(
        adversarial_image
    )

    adversarial_prediction = (
        adversarial_output.argmax(
            dim=1
        ).item()
    )


# ============================================================
# 11. Print results
# ============================================================

print("--------------------------------")
print("FGSM ATTACK RESULTS")
print("--------------------------------")
print("True label:", label.item())
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
print("--------------------------------")
# ============================================================
# 12. Display images
# ============================================================

plt.figure(figsize=(8, 3))
plt.subplot(1, 2, 1)
plt.imshow(
    image.detach().squeeze(),
    cmap="gray"
)
plt.title(
    f"Original: {original_prediction}"
)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(
    adversarial_image.detach().squeeze(),
    cmap="gray"
)
plt.title(
    f"Adversarial: {adversarial_prediction}"
)
plt.axis("off")
plt.tight_layout()
plt.savefig(
    "attacks/results/fgsm_result.png"
)
plt.show()