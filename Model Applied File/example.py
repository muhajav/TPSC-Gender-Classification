import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# 1. Define model
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

state_dict = torch.load("models/gender_classifier_model.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

# 2. Preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# 3. Load image
img_path = "data/celeba/img_align_celeba/000001.jpg"
image = Image.open(img_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0)

# 4. Enable gradient for input
input_tensor.requires_grad_()

# 5. Forward pass
output = model(input_tensor)

# For binary output: take scalar value
score = output[0]  
score.backward()

# 6. Compute saliency
saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
saliency_map = saliency.squeeze().cpu().numpy()

# 7. Save saliency map
np.save("saliency_image01.npy", saliency_map)

print("✅ Saved saliency_image01.npy")