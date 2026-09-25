from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms
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
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])
app = FastAPI(
    title="AI API"
)
@app.get("/")
def home():
    return {
        "message": "API is running"
    }
@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image_data = await file.read()
    image = Image.open(
        io.BytesIO(image_data)
    )
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.softmax(
            output,
            dim=1
        )
        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()
        confidence = probabilities[
            0,
            prediction
        ].item()
    return {
        "prediction": prediction,
        "confidence": confidence,
        "file": file.filename
    }