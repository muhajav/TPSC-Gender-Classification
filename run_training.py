# ===================================================================
# SECTION 1: IMPORTS AND SETUP
# ===================================================================
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import zipfile

# <<< CHANGED: Set device to Apple's MPS for GPU acceleration
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: Apple MPS (GPU)")
else:
    device = torch.device("cpu")
    print("Using device: CPU")


# ===================================================================
# SECTION 2: DATA PREPARATION (LOCAL PATHS)
# ===================================================================
# <<< CHANGED: Paths are now local to your project folder
DATA_ROOT = './data'
zip_path = os.path.join(DATA_ROOT, 'celeba/img_align_celeba.zip')
unzipped_path = os.path.join(DATA_ROOT, 'celeba/img_align_celeba/')

# Unzip the file if the folder doesn't already exist
if not os.path.exists(unzipped_path):
    print("Unzipping dataset... (This will take a few minutes)")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(DATA_ROOT, 'celeba/'))
    print("Unzipping complete.")
else:
    print("Dataset already unzipped.")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Load the CelebA dataset from your local data folder
train_dataset = torchvision.datasets.CelebA(
    root=DATA_ROOT,
    split='train',
    target_type='attr',
    download=False,
    transform=transform
)

# Create the DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
print("✅ Dataset is ready.")


# ===================================================================
# SECTION 3: MODEL PREPARATION
# ===================================================================
model = models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1) # Single output for binary classification
model = model.to(device) # Move model to MPS device
print("Model is ready.")


# ===================================================================
# SECTION 4: FINE-TUNING THE MODEL
# ===================================================================
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Starting model fine-tuning...")
model.train()
for i, (images, attributes) in enumerate(tqdm(train_loader)):
    # The 'Male' attribute is at index 20
    labels = attributes[:, 20].float().unsqueeze(1).to(device)
    images = images.to(device)

    outputs = model(images)
    loss = criterion(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i >= 200: # Train for ~200 batches for a quick but decent result
        break
print("Model fine-tuning complete!")


# ===================================================================
# SECTION 5: SAVE THE FINAL MODEL
# ===================================================================
# <<< CHANGED: Save the model to a local 'models' folder
MODEL_SAVE_PATH = './models/gender_classifier_model.pth'
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Model saved locally at: {MODEL_SAVE_PATH}")