# 1. Importer les bibliothèques
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# 2. Charger et préparer les données
# Utiliser torchvision pour télécharger MNIST et créer des loaders
transform = transforms.Compose([
    transforms.ToTensor(),      # Convertit les images en tenseurs PyTorch
    transforms.Normalize((0.5,), (0.5,))  # Normaliser les données
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 3. Définir le modèle de réseau de neurones
# Un réseau de neurones simple avec une couche cachée

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)   # Première couche entièrement connectée
        self.fc2 = nn.Linear(128, 64)      # Deuxième couche cachée
        self.fc3 = nn.Linear(64, 10)       # Couche de sortie (10 classes, une pour chaque chiffre)

    def forward(self, x):
        x = x.view(-1, 28*28)  # Aplatir l'image en un vecteur 1D
        x = torch.relu(self.fc1(x))  # Fonction d'activation ReLU après chaque couche cachée
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # La couche finale n'a pas de fonction d'activation
        return x

# 4. Définir la fonction de perte et l'optimiseur
model = NeuralNet()
criterion = nn.CrossEntropyLoss()  # Fonction de perte pour la classification multi-classe
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Utiliser un optimiseur SGD

# 5. Entraîner le modèle
for epoch in range(epochs):  # epochs = nombre de passages sur les données
    for images, labels in train_loader:
        optimizer.zero_grad()  # Réinitialiser les gradients
        
        outputs = model(images)  # Passer les images dans le modèle
        loss = criterion(outputs, labels)  # Calculer la perte
        
        loss.backward()  # Calculer les gradients
        optimizer.step()  # Mettre à jour les poids avec les gradients

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 6. Tester le modèle
model.eval()  # Mettre le modèle en mode évaluation
with torch.no_grad():  # Pas besoin de calculer les gradients pendant l'évaluation
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # Obtenir la prédiction avec la probabilité la plus élevée
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Précision du modèle sur les images test : {100 * correct / total} %')
