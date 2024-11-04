import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Tuple

class DigitClassifier(nn.Module):
    """Digit classifier based on pretrained ResNet18."""
    def __init__(self, num_classes: int = 10, freeze_backbone: bool = True):
        super().__init__()
        # Load pretrained ResNet18
        self.backbone = resnet18(pretrained=True)
        
        if freeze_backbone:
            # Freeze all backbone layers
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Modify final layer for digit classification
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

def prepare_data() -> Tuple[DataLoader, DataLoader]:
    """Prepare SVHN dataset with appropriate transforms."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    train_set = torchvision.datasets.SVHN(
        root='./data', split='train', download=True, transform=transform)
    test_set = torchvision.datasets.SVHN(
        root='./data', split='test', download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

def train_model(model: nn.Module, 
                train_loader: DataLoader,
                val_loader: DataLoader,
                num_epochs: int = 10,
                device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> None:
    """Train the digit classifier model."""
    print(f"Training on device: {device}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model = model.to(device)
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        print(f'Validation Accuracy: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_digit_classifier.pth')

def evaluate_model(model: nn.Module, 
                  test_loader: DataLoader,
                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict:
    """Evaluate the model and compute metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

def plot_comparative_results(metrics_frozen: Dict, metrics_finetuned: Dict):
    """Plot comparative results and confusion matrices using colorblind-friendly colors."""
    
    frozen_color = '#0077BB'  # Blue
    finetuned_color = '#EE7733'  # Orange
    
    # Plot comparative metrics
    plt.figure(figsize=(12, 5))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics))
    width = 0.35
    
    frozen_values = [metrics_frozen[m] for m in metrics]
    finetuned_values = [metrics_finetuned[m] for m in metrics]
    
    plt.bar(x - width/2, frozen_values, width, label='Frozen Backbone', color=frozen_color)
    plt.bar(x + width/2, finetuned_values, width, label='Fine-tuned', color=finetuned_color)
    
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparative_metrics.png')
    plt.close()
    
    # Plot confusion matrices side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.heatmap(metrics_frozen['confusion_matrix'], 
                annot=True, 
                fmt='d',
                cmap='Blues',
                ax=ax1)
    ax1.set_title('Confusion Matrix - Frozen Backbone')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    sns.heatmap(metrics_finetuned['confusion_matrix'], 
                annot=True, 
                fmt='d',
                cmap='Oranges',
                ax=ax2)
    ax2.set_title('Confusion Matrix - Fine-tuned')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Prepare data
    train_loader, test_loader = prepare_data()
    
    # Train with frozen backbone
    print("Training with frozen backbone...")
    model_frozen = DigitClassifier(freeze_backbone=True)
    train_model(model_frozen, train_loader, test_loader)
    metrics_frozen = evaluate_model(model_frozen, test_loader)
    
    # Train with fine-tuning
    print("\nTraining with fine-tuning...")
    model_finetuned = DigitClassifier(freeze_backbone=False)
    train_model(model_finetuned, train_loader, test_loader)
    metrics_finetuned = evaluate_model(model_finetuned, test_loader)
    
    # Plot comparative results
    plot_comparative_results(metrics_frozen, metrics_finetuned)
    
    # Print final metrics
    print("\nResults with frozen backbone:")
    print(f"Accuracy: {metrics_frozen['accuracy']:.4f}")
    print(f"Precision: {metrics_frozen['precision']:.4f}")
    print(f"Recall: {metrics_frozen['recall']:.4f}")
    print(f"F1 Score: {metrics_frozen['f1']:.4f}")
    
    print("\nResults with fine-tuning:")
    print(f"Accuracy: {metrics_finetuned['accuracy']:.4f}")
    print(f"Precision: {metrics_finetuned['precision']:.4f}")
    print(f"Recall: {metrics_finetuned['recall']:.4f}")
    print(f"F1 Score: {metrics_finetuned['f1']:.4f}")

if __name__ == "__main__":
    main()