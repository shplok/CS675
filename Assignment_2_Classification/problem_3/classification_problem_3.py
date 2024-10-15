import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

y_train = y_train.flatten()
y_test = y_test.flatten()

# Preprocess the data
def preprocess_data(X):
    # Flatten the images
    X_flat = X.reshape(X.shape[0], -1)
    # Normalize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_flat)
    return X_scaled

X_train_processed = preprocess_data(X_train)
X_test_processed = preprocess_data(X_test)

# Reduce dataset size for faster computation (for me to test if functional)
# n_samples = 10000
# X_train_reduced, _, y_train_reduced, _ = train_test_split(X_train_processed, y_train, train_size=n_samples, stratify=y_train, random_state=42)
# X_test_reduced, _, y_test_reduced, _ = train_test_split(X_test_processed, y_test, train_size=n_samples//5, stratify=y_test, random_state=42)

# Initialize classifiers
classifiers = {
    'Nearest Neighbors': KNeighborsClassifier(),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

# Perform cross-validation and hyperparameter tuning
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in classifiers.items():
    if name == 'Nearest Neighbors':
        param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
        best_score = 0
        best_k = 0
        for k in param_grid['n_neighbors']:
            clf.set_params(n_neighbors=k)
            scores = cross_val_score(clf, X_train_processed, y_train.ravel(), cv=cv, n_jobs=-1)
            if np.mean(scores) > best_score:
                best_score = np.mean(scores)
                best_k = k
        print(f"Best k for Nearest Neighbors: {best_k}")
        clf.set_params(n_neighbors=best_k)
    
    print(f"Training {name}...")
    clf.fit(X_train_processed, y_train.ravel())

# Evaluate classifiers
results = {}
for name, clf in classifiers.items():
    print(f"Evaluating {name}...")
    y_pred = clf.predict(X_test_processed)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    results[name] = {'accuracy': accuracy, 'confusion_matrix': conf_matrix}

# Print results
for name, result in results.items():
    print(f"{name}:")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print("Confusion Matrix:")
    print(result['confusion_matrix'])
    print()

# Plot confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

for ax, (name, result) in zip(axes, results.items()):
    cm = result['confusion_matrix']
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(name)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('cifar10_confusion_matrices.png')
plt.close()


# Discussion
print("\nDiscussion:")
print("1. Nearest Neighbors: This non-parametric method can capture complex decision boundaries but may be computationally expensive for large datasets.")
print("2. Linear Discriminant Analysis: Assumes classes are normally distributed with equal covariance matrices. May struggle with the complexity of image data.")
print("3. Quadratic Discriminant Analysis: Similar to LDA but allows for different covariance matrices for each class, potentially capturing more complex boundaries.")
print("4. Logistic Regression: A linear model that may struggle with the high-dimensionality and non-linearity of image data.")
print("\nThe performance of each classifier on CIFAR-10 will likely be lower than on simpler datasets due to the complexity of image classification.")
print("Analyze the confusion matrices to understand which classes are most often confused by each classifier.")