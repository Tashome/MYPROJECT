import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

from torch.utils.data import DataLoader




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

model.load_state_dict(
    torch.load(
        "model/model.pt",
        map_location="cpu"
    )
)

model.eval()




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




def fgsm_attack(image, epsilon, gradient):

    perturbation = epsilon * gradient.sign()

    adversarial_image = image + perturbation

    adversarial_image = torch.clamp(
        adversarial_image,
        0,
        1
    )


    return adversarial_image
image, label = next(iter(test_loader))
image.requires_grad = True
output = model(image)

original_prediction = output.argmax(
    dim=1
).item()
loss_function = nn.CrossEntropyLoss()
loss = loss_function(
    output,
    label
)
model.zero_grad()

loss.backward()

gradient = image.grad.data
epsilon = 0.1
adversarial_image = fgsm_attack(
    image,
    epsilon,
    gradient
)
with torch.no_grad():
    adversarial_output = model(
        adversarial_image
    )
    adversarial_prediction = (
        adversarial_output.argmax(
            dim=1
        ).item()
    )
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

plt.savefig
(
    "attacks/results/fgsm_result.png"
)
plt.show()