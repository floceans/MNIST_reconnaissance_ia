import torch
from torchvision import transforms
from PIL import Image
from torch import nn


# Définition du modèle
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Fonction pour tester une image
def test_image(model, image_path):
    # Transformation à appliquer à l'image
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Charger l'image
    image = Image.open(image_path).convert('L')  # Convertir en niveaux de gris
    image = transform(image).unsqueeze(0)  # Ajouter une dimension pour le batch

    # Mettre le modèle en mode évaluation
    model.eval()

    with torch.no_grad():  # Désactiver le calcul des gradients
        output = model(image)  # Obtenir la sortie du modèle
        print("Sortie de la dernière couche :", output.numpy())  # Afficher la sortie

# Charger le modèle
model = SimpleNN()
model.load_state_dict(torch.load('mnist_model_1.pth'))  # Chargez votre modèle ici

# Chemin de l'image à tester
image_path = r"C:\Users\service.si\OneDrive - MADININAIR\Documents\Torch_test\MNIST\data\test_1par1\six.png"  # Remplacez par le chemin de votre image
test_image(model, image_path)
