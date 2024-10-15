import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.inspection import DecisionBoundaryDisplay

# Generate synthetic dataset
def generate_dataset(n_samples=1000):
    n_samples_per_class = n_samples // 2
    
    # Class 0
    mean1 = [0, 0]
    cov1 = [[1, 0], [0, 1]]
    x0 = np.random.multivariate_normal(mean1, cov1, n_samples_per_class)
    
    # Class 1
    mean2 = [2, 2]
    cov2 = [[1.5, 0], [0, 1.5]]
    x1 = np.random.multivariate_normal(mean2, cov2, n_samples_per_class)
    
    X = np.vstack((x0, x1))
    y = np.hstack((np.zeros(n_samples_per_class), np.ones(n_samples_per_class)))
    
    return X, y

# Generate dataset and split into train and test sets
X, y = generate_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
classifiers = {
    'Nearest Neighbors': KNeighborsClassifier(),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
    'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
    'Logistic Regression': LogisticRegression(random_state=42)
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
            scores = cross_val_score(clf, X_train, y_train, cv=cv)
            if np.mean(scores) > best_score:
                best_score = np.mean(scores)
                best_k = k
        print(f"Best k for Nearest Neighbors: {best_k}")
        clf.set_params(n_neighbors=best_k)
    
    clf.fit(X_train, y_train)

# Evaluate classifiers
results = {}
for name, clf in classifiers.items():
    y_pred = clf.predict(X_test)
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

# Plot decision boundaries
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

for ax, (name, clf) in zip(axes, classifiers.items()):
    DecisionBoundaryDisplay.from_estimator(
        clf, X, cmap=plt.cm.RdYlBu, response_method="predict", 
        ax=ax, alpha=0.8, xlabel="Feature 1", ylabel="Feature 2"
    )
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
    ax.set_title(name)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

plt.tight_layout()
plt.savefig('decision_boundaries.png')
plt.close()

# Plot confusion matrices
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for ax, (name, result) in zip(axes, results.items()):
    cm = result['confusion_matrix']
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(name)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Class 0', 'Class 1'])
    ax.set_yticklabels(['Class 0', 'Class 1'])
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.close()

# Discussion
print("\nDiscussion:")
print("1. Nearest Neighbors: This non-parametric method can capture complex decision boundaries but may be sensitive to the choice of k.")
print("2. Linear Discriminant Analysis: Assumes classes are normally distributed with equal covariance matrices. Works well if the assumption holds.")
print("3. Quadratic Discriminant Analysis: Similar to LDA but allows for different covariance matrices for each class, potentially capturing more complex boundaries.")
print("4. Logistic Regression: A linear model that works well for linearly separable classes but may struggle with more complex distributions.")
print("\nThe performance of each classifier depends on how well the underlying assumptions match the data distribution.")
print("Analyze the decision boundaries and confusion matrices to gain insights into each classifier's strengths and weaknesses.")