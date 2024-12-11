import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Configuration
data_dir = r"C:\Users\service.si\OneDrive - MADININAIR\Documents\Torch_test\MNIST\data\mnist_png\training"  # Remplacez par le chemin de votre dossier
batch_size = 64
learning_rate = 0.001
num_epochs = 10

# Dataset personnalisé pour charger les images
class MNISTDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []

        # Charger les images et les labels
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):
                for img_file in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_file)
                    self.images.append(img_path)
                    self.labels.append(int(label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('L')  # Convertir en niveaux de gris
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Transformations à appliquer aux images
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Création du dataset et des DataLoaders
dataset = MNISTDataset(data_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Définition du modèle de réseau de neurones
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Initialisation du modèle, de la fonction de perte et de l'optimiseur
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Entraînement du modèle
for epoch in range(num_epochs):
    for images, labels in dataloader:
        optimizer.zero_grad()  # Remise à zéro des gradients
        outputs = model(images)  # Prédictions
        loss = criterion(outputs, labels)  # Calcul de la perte
        loss.backward()  # Rétropropagation
        optimizer.step()  # Mise à jour des poids

    print(f'Époque [{epoch+1}/{num_epochs}], Perte: {loss.item():.4f}')

# Sauvegarder le modèle
torch.save(model.state_dict(), 'mnist_model.pth')
print("Modèle sauvegardé sous mnist_model.pth")
