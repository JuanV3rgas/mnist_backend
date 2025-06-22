import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import base64
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

# Generator definition (same as training!)
class Generator(nn.Module):
    def __init__(self, noise_dim=100, num_classes=10, img_dim=28*28):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + num_classes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, img_dim),
            nn.Tanh()
        )
    def forward(self, noise, labels):
        c = self.label_emb(labels)
        x = torch.cat([noise, c], dim=1)
        img = self.model(x)
        return img

# Load the generator weights
generator = Generator()
generator.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), "generator.pth"), map_location="cpu"))
generator.eval()

def image_to_base64(img_array):
    img = Image.fromarray(img_array)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # O para más seguridad: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/generate")
def generate(digit: int = Query(..., ge=0, le=9), n: int = 5):
    noise_dim = 100
    z = torch.randn(n, noise_dim)
    labels = torch.full((n,), digit, dtype=torch.long)
    with torch.no_grad():
        imgs = generator(z, labels).view(-1, 28, 28).cpu().numpy()
        imgs = ((imgs + 1) / 2 * 255).astype(np.uint8)
        images_b64 = [image_to_base64(img) for img in imgs]
    return JSONResponse(content={"images": images_b64})
