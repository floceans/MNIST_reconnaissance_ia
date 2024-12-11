import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Définition du modèle (doit correspondre à la structure du modèle entraîné)
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

# Chargement du modèle
model = SimpleNN()
model.load_state_dict(torch.load('mnist_model.pth'))  # Charger les poids du modèle
model.eval()

# Fonction d'inversion du modèle
def generate_image(target_class, iterations=10000, learning_rate=0.005):
    # Initialiser une image aléatoire
    #image = torch.randint(0, 2, (1, 1, 28, 28), dtype=torch.float32, requires_grad=True)
    image = torch.randn(1, 1, 28, 28, requires_grad=True)  # Initialiser avec des valeurs aléatoires
    #afficher l'image
    plt.imshow(image.squeeze().detach().numpy(), cmap='gray')
    plt.axis('off')
    plt.title('Image initiale')
    plt.show()
    
    optimizer = torch.optim.Adam([image], lr=learning_rate)
    
    # Transformation pour dénormaliser
    transform = transforms.Compose([
        transforms.Normalize((-1,), (2,))  # Inverser la normalisation
    ])
    
    for _ in range(iterations):
        optimizer.zero_grad()
        output = model(image)  # Obtenir la prédiction du modèle
        loss = -output[0, target_class]  # Négatif de la sortie pour maximiser la classe cible
        loss.backward()  # Calculer les gradients
        optimizer.step()  # Mettre à jour l'image
        
        # Limiter les valeurs de l'image entre -1 et 1
        with torch.no_grad():
            image.data.clamp_(-1, 1)

    # Convertir l'image en format affichable
    generated_image = transform(image.detach()).squeeze().numpy()
    return generated_image

# Générer une image pour la classe cible (par exemple, le chiffre 3)
target_class = 2  # Chiffre que vous voulez générer
generated_image = generate_image(target_class)

# Affichage de l'image générée
plt.imshow(generated_image, cmap='gray')
plt.axis('off')
plt.title(f'Image générée pour la classe: {target_class}')
plt.show()
