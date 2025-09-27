# Gender Classification with CelebA

A project to classify gender from face images and explain the model's decisions using LIME and Saliency Maps. This repository contains the code for training the model and will be used for collaborative development of the explanation scripts.

---

## 📋 Project Overview

- **Dataset**: CelebA (Align&Cropped version)
- **Model**: A ResNet18 model, fine-tuned for binary gender classification.
- **Explainability Methods**: LIME vs. Saliency Maps

---

## ⚙️ Prerequisites

Before you begin, ensure you have the following installed on your machine:

- Python 3.8 or higher
- Git

---

## 🚀 Setup Instructions

Follow these steps to get a working copy of the project on your local machine.

### 1. Clone the Repository

First, clone the project code from GitHub. Open your terminal and run:

```bash
git clone <https://github.com/muhajav/TPSC-Gender-Classification>
```

### 2. Download the Data and Model

The dataset and the pre-trained model are too large for GitHub and are stored in a shared Google Drive folder.

- **[Click here to download the `data` and `models` folders](https://drive.google.com/drive/folders/1u_7_UV1suI2wwp864qCMfSdzwz--UDSb?usp=sharing)**
- Place both the `data` and `models` folders directly inside the project directory you just cloned.

### 3. Set Up the Python Environment

We use a virtual environment to keep project dependencies separate.

```bash
# Navigate into the project folder
cd TPSC-Gender-Classification

# Create a virtual environment
python3 -m venv venv

# Activate the environment
source venv/bin/activate
```

Your terminal prompt should now start with `(venv)`.

### 4. Install Dependencies

Install all the required Python libraries by running:

```bash
pip3 install torch torchvision tqdm
```

---

## ▶️ How to Run

Your environment is now set up.

### Training the Model

The model has already been trained by Javier, and the result is saved in the `models/` folder. You do not need to run the training script again. However, if you want to retrain it for any reason, you can run:

```bash
python3 run_training.py
```

### Applying Explainability Methods (Your Main Task)

Your main task is to load the pre-trained model and apply the LIME and Saliency methods. You should create a new script (e.g., `explain.py`) to:

1.  Load the trained model from `./models/gender_classifier_model.pth`.
2.  Load sample images from the `./data/celeba/` folder.
3.  Apply the explanation methods to the model's predictions on those images.
4.  Visualize and save the results (e.g., heatmaps).

---

## 📁 Final Project Structure

After completing all the setup steps, your project folder should look like this:

```
TPSC-Gender-Classification/
├── .gitignore
├── README.md
├── run_training.py
├── venv/
├── data/
│   └── celeba/
│       ├── img_align_celeba/
│       ├── img_align_celeba.zip
│       ├── identity_CelebA.txt
│       └── ... (and the other 4 txt files)
└── models/
    └── gender_classifier_model.pth
```
