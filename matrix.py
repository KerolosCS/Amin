import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

# User-defined accuracy
accuracy = 0.8
num_classes = 10
num_samples = 1800

# Generate y_true and y_pred
y_true = np.random.randint(0, num_classes, num_samples)
y_pred = y_true.copy()

# Introduce errors based on the desired accuracy
num_correct = int(accuracy * num_samples)
num_incorrect = num_samples - num_correct

incorrect_indices = np.random.choice(num_samples, num_incorrect, replace=False)

for idx in incorrect_indices:
    y_pred[idx] = np.random.randint(0, num_classes)

# Create confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix ({num_classes} Classes)')

# Save as an image
plt.savefig(f'{accuracy*100}.png')
plt.close()
