import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from itertools import combinations, product

class SiameseNetwork(nn.Module):
    """Siamese network for image verification."""
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        # Load pretrained ResNet18 as the backbone
        self.backbone = resnet18(pretrained=True)
        
        # Remove the final classification layer
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )
        
        # L2 normalize the output embeddings
        self.normalize = lambda x: nn.functional.normalize(x, p=2, dim=1)
    
    def forward_one(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for one branch."""
        embedding = self.backbone(x)
        return self.normalize(embedding)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for both branches."""
        return self.forward_one(x1), self.forward_one(x2)

class ContrastiveLoss(nn.Module):
    """Contrastive loss function."""
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, embedding1: torch.Tensor, embedding2: torch.Tensor, 
                label: torch.Tensor) -> torch.Tensor:
        """
        Calculate contrastive loss.
        label = 1 for similar pairs, 0 for dissimilar pairs
        """
        distance = nn.functional.pairwise_distance(embedding1, embedding2)
        loss = (label) * torch.pow(distance, 2) + \
               (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        return loss.mean()

class PairDataset(Dataset):
    """Dataset for creating pairs of images."""
    def __init__(self, dataset, known_classes: List[int], train: bool = True):
        self.dataset = dataset
        self.train = train
        self.known_classes = set(known_classes)
        
        # Separate data by class
        self.class_indices = self._organize_by_class()
        
        # Generate pairs
        self.pairs, self.labels = self._generate_pairs()
    
    def _organize_by_class(self) -> Dict[int, List[int]]:
        class_indices = {}
        for idx, (_, label) in enumerate(self.dataset):
            if label in self.known_classes:
                if label not in class_indices:
                    class_indices[label] = []
                class_indices[label].append(idx)
        return class_indices
    
    def _generate_pairs(self) -> Tuple[List[Tuple[int, int]], List[int]]:
        pairs = []
        labels = []
        
        # Generate similar pairs (same class)
        for class_idx in self.class_indices:
            indices = self.class_indices[class_idx]
            for idx1, idx2 in combinations(indices, 2):
                pairs.append((idx1, idx2))
                labels.append(1)
        
        # Generate dissimilar pairs (different classes)
        classes = list(self.class_indices.keys())
        for class1, class2 in combinations(classes, 2):
            indices1 = self.class_indices[class1]
            indices2 = self.class_indices[class2]
            for idx1, idx2 in product(indices1[:len(indices1)//2], 
                                    indices2[:len(indices2)//2]):
                pairs.append((idx1, idx2))
                labels.append(0)
        
        return pairs, labels
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        idx1, idx2 = self.pairs[index]
        img1, _ = self.dataset[idx1]
        img2, _ = self.dataset[idx2]
        label = self.labels[index]
        return img1, img2, torch.FloatTensor([label])
    
    def __len__(self) -> int:
        return len(self.pairs)

def prepare_data(known_classes: List[int] = [0, 1, 2, 3, 4]) -> Tuple[DataLoader, DataLoader]:
    """Prepare CIFAR-10 dataset with appropriate transforms."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    
    train_pairs = PairDataset(train_set, known_classes, train=True)
    test_pairs = PairDataset(test_set, known_classes, train=False)
    
    train_loader = DataLoader(train_pairs, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_pairs, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, test_loader

def train_siamese(model: nn.Module, 
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 num_epochs: int = 10,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
    """Train the Siamese network."""
    print(f"Training on device: {device}")
    
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    model = model.to(device)
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for img1, img2, labels in train_loader:
            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
            
            optimizer.zero_grad()
            embedding1, embedding2 = model(img1, img2)
            loss = criterion(embedding1, embedding2, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for img1, img2, labels in val_loader:
                img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)
                embedding1, embedding2 = model(img1, img2)
                loss = criterion(embedding1, embedding2, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        
        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_siamese_model.pth')

def evaluate_verification(model: nn.Module, 
                        test_loader: DataLoader,
                        device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict:
    """Evaluate the verification system and compute metrics."""
    model.eval()
    distances = []
    labels = []
    
    with torch.no_grad():
        for img1, img2, batch_labels in test_loader:
            img1, img2 = img1.to(device), img2.to(device)
            embedding1, embedding2 = model(img1, img2)
            distance = nn.functional.pairwise_distance(embedding1, embedding2)
            
            distances.extend(distance.cpu().numpy())
            labels.extend(batch_labels.numpy())
    
    distances = np.array(distances)
    labels = np.array(labels)
    
    # Calculate ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(labels, -distances)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    # Calculate metrics at optimal threshold
    predictions = (distances <= -optimal_threshold).astype(int)
    true_positive_rate = np.mean(predictions[labels == 1])
    false_positive_rate = np.mean(predictions[labels == 0])
    
    return {
        'auc': roc_auc,
        'optimal_threshold': -optimal_threshold,
        'tpr': true_positive_rate,
        'fpr': false_positive_rate,
        'roc_curve': (fpr, tpr)
    }

def plot_verification_results(metrics: Dict):
    """Plot ROC curve and verification results."""
    plt.figure(figsize=(10, 5))
    
    # Plot ROC curve
    fpr, tpr = metrics['roc_curve']
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {metrics["auc"]:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig('verification_results.png')
    plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define known classes (first 5 classes of CIFAR-10)
    known_classes = [0, 1, 2, 3, 4]
    
    # Prepare data
    train_loader, test_loader = prepare_data(known_classes)
    
    # Create and train Siamese network
    model = SiameseNetwork()
    train_siamese(model, train_loader, test_loader)
    
    # Evaluate verification system
    metrics = evaluate_verification(model, test_loader)
    
    # Plot results
    plot_verification_results(metrics)
    
    # Print final metrics
    print("\nVerification System Results:")
    print(f"AUC-ROC: {metrics['auc']:.4f}")
    print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
    print(f"True Positive Rate: {metrics['tpr']:.4f}")
    print(f"False Positive Rate: {metrics['fpr']:.4f}")

if __name__ == "__main__":
    main()