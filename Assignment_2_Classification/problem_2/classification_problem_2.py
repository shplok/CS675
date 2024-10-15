import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the digits dataset
digits = datasets.load_digits()
X, y = digits.data, digits.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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
            scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv)
            if np.mean(scores) > best_score:
                best_score = np.mean(scores)
                best_k = k
        print(f"Best k for Nearest Neighbors: {best_k}")
        clf.set_params(n_neighbors=best_k)
    
    clf.fit(X_train_scaled, y_train)

# Evaluate classifiers
results = {}
for name, clf in classifiers.items():
    y_pred = clf.predict(X_test_scaled)
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

for ax, (name, result) in zip(axes, results.items()):
    cm = result['confusion_matrix']
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(name)
    ax.set_xticks(np.arange(10))
    ax.set_yticks(np.arange(10))
    ax.set_xticklabels(np.arange(10))
    ax.set_yticklabels(np.arange(10))
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('digits_confusion_matrices.png')
plt.close()

# Discussion
print("\nDiscussion:")
print("1. Nearest Neighbors: This method can capture complex decision boundaries but may be sensitive to the choice of k and the curse of dimensionality.")
print("2. Linear Discriminant Analysis: Assumes classes are normally distributed with equal covariance matrices. It can perform well on this dataset due to its dimensionality reduction properties.")
print("3. Quadratic Discriminant Analysis: Similar to LDA but allows for different covariance matrices for each class, potentially capturing more complex boundaries.")
print("4. Logistic Regression: A linear model that can work well for multi-class problems like this one, especially when classes are roughly linearly separable.")
print("\nThe performance of each classifier depends on how well the underlying assumptions match the data distribution.")
print("Analyze the confusion matrices to gain insights into each classifier's strengths and weaknesses for different digits.")