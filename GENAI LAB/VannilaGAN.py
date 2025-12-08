import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
latent_dim = 100  # Size of noise vector for generator input
hidden_dim = 256  # Hidden layer size
image_dim = 784   # 28x28 flattened FashionMNIST images
num_epochs = 50   # Few epochs for quick training on small data
batch_size = 64
learning_rate = 0.0002
small_dataset_size = 1000  # Use a small subset

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load small FashionMNIST subset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])  # Normalize to [-1, 1]
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.Subset(train_dataset, range(small_dataset_size)),  # Small subset
    batch_size=batch_size, shuffle=True
)

# Generator Network: Maps noise to fake image
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, image_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self, z):
        return self.model(z)

# Discriminator Network: Classifies real (1) vs fake (0)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output probability [0, 1]
        )
    
    def forward(self, img):
        return self.model(img.view(img.size(0), -1))  # Flatten input

# Initialize models and optimizers
generator = Generator().to(device)
discriminator = Discriminator().to(device)

g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)

# Loss function
criterion = nn.BCELoss()

# Training loop
def train_gan():
    for epoch in range(num_epochs):
        for i, (real_imgs, _) in enumerate(train_loader):
            batch_size_current = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            
            # Labels for real and fake
            real_labels = torch.ones(batch_size_current, 1).to(device)
            fake_labels = torch.zeros(batch_size_current, 1).to(device)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            real_output = discriminator(real_imgs)
            d_real_loss = criterion(real_output, real_labels)
            
            z = torch.randn(batch_size_current, latent_dim).to(device)
            fake_imgs = generator(z)
            fake_output = discriminator(fake_imgs.detach())
            d_fake_loss = criterion(fake_output, fake_labels)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            fake_output = discriminator(fake_imgs)
            g_loss = criterion(fake_output, real_labels)  # Fool discriminator
            g_loss.backward()
            g_optimizer.step()
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}] D loss: {d_loss.item():.4f}, G loss: {g_loss.item():.4f}')

# Train the GAN
train_gan()

# Generate new set of images (e.g., 16 fakes)
def generate_images(num_images=16):
    z = torch.randn(num_images, latent_dim).to(device)
    fake_imgs = generator(z).detach().cpu().numpy()
    fake_imgs = (fake_imgs.reshape(num_images, 28, 28) + 1) / 2  # Denormalize to [0, 1]
    
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake_imgs[i], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('generated_images.png')  # Save for viewing
    plt.show()
    print("Generated images saved as 'generated_images.png'")

generate_images()

# Classify real/fake using trained Discriminator
def classify_images(real_imgs, fake_imgs):
    discriminator.eval()
    with torch.no_grad():
        real_probs = discriminator(real_imgs.to(device)).cpu().numpy()
        fake_probs = discriminator(fake_imgs.to(device)).cpu().numpy()
    
    # Sample a few for demo
    num_samples = 5
    print("\nClassification Results (Probability of being REAL):")
    print("Real Images:")
    for i in range(min(num_samples, len(real_probs))):
        prob = real_probs[i].item()
        pred = "Real" if prob > 0.5 else "Fake"
        print(f"Image {i+1}: Prob = {prob:.4f} -> {pred}")
    
    print("Fake Images:")
    for i in range(min(num_samples, len(fake_probs))):
        prob = fake_probs[i].item()
        pred = "Real" if prob > 0.5 else "Fake"
        print(f"Image {i+1}: Prob = {prob:.4f} -> {pred}")

# Demo classification: Use a batch of real and generated fake
real_batch, _ = next(iter(train_loader))
z_demo = torch.randn(batch_size, latent_dim).to(device)
fake_batch = generator(z_demo).detach()

classify_images(real_batch, fake_batch)  # Pass full batch