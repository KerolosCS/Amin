# import matplotlib.pyplot as plt
# import numpy as np
#
# # Setting a seed for reproducibility
# np.random.seed(42)
#
# # Simulated data for different degrees of overfitting
# epochs = np.arange(0, 220, 1)
#
# # Define a smoothing function
# def smooth(y, box_pts):
#     box = np.ones(box_pts) / box_pts
#     y_smooth = np.convolve(y, box, mode='same')
#     return y_smooth
#
# # Near-ideal model with noise
# training_loss_ideal = smooth(np.exp(-0.03 * epochs) + 0.02 * np.random.randn(epochs.size), 5)
# validation_loss_ideal = smooth(np.exp(-0.03 * epochs) + 0.1 + 0.02 * np.random.randn(epochs.size), 5)
#
# # Moderate overfitting with noise
# training_loss_moderate = smooth(np.exp(-0.03 * epochs) + 0.02 * np.random.randn(epochs.size), 5)
# validation_loss_moderate = smooth(np.exp(-0.03 * epochs) + 0.2 * (epochs / max(epochs)) + 0.02 * np.random.randn(epochs.size), 5)
#
# # Severe overfitting with noise
# training_loss_severe = smooth(np.exp(-0.03 * epochs) + 0.02 * np.random.randn(epochs.size), 5)
# validation_loss_severe = smooth(np.exp(-0.03 * epochs) + 0.4 * (epochs / max(epochs)) ** 2 + 0.02 * np.random.randn(epochs.size), 5)
#
# plt.figure(figsize=(18, 6))
#
# # Plotting the near-ideal model
# plt.subplot(1, 3, 1)
# plt.plot(epochs, training_loss_ideal, label='Training Loss', color='blue')
# plt.plot(epochs, validation_loss_ideal, label='Validation Loss', color='orange')
# plt.title('Near-Ideal Model', color='black')
# plt.xlabel('Epoch', color='black')
# plt.ylabel('Loss', color='black')
# plt.legend()
# plt.grid(True)
#
# # Plotting the moderate overfitting model
# plt.subplot(1, 3, 2)
# plt.plot(epochs, training_loss_moderate, label='Training Loss', color='blue')
# plt.plot(epochs, validation_loss_moderate, label='Validation Loss', color='orange')
# plt.title('Moderate Overfitting', color='black')
# plt.xlabel('Epoch', color='black')
# plt.ylabel('Loss', color='black')
# plt.legend()
# plt.grid(True)
#
# # Plotting the severe overfitting model
# plt.subplot(1, 3, 3)
# plt.plot(epochs, training_loss_severe, label='Training Loss', color='blue')
# plt.plot(epochs, validation_loss_severe, label='Validation Loss', color='orange')
# plt.title('Severe Overfitting', color='black')
# plt.xlabel('Epoch', color='black')
# plt.ylabel('Loss', color='black')
# plt.legend()
# plt.grid(True)
#
# # Set white background for the figure and axes
# plt.gcf().patch.set_facecolor('white')
# for ax in plt.gcf().get_axes():
#     ax.set_facecolor('white')
#     ax.spines['top'].set_color('black')
#     ax.spines['right'].set_color('black')
#     ax.spines['left'].set_color('black')
#     ax.spines['bottom'].set_color('black')
#     ax.tick_params(axis='x', colors='black')
#     ax.tick_params(axis='y', colors='black')
#     ax.legend(frameon=False, loc='best')
#
# plt.tight_layout()
# plt.savefig('data/learning_rate_plots_white_background.png')
# plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Setting a seed for reproducibility
np.random.seed(42)

# Simulated data for different degrees of overfitting
epochs = np.arange(0, 220, 1)

# Define a smoothing function
def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# Generate smoother data
training_loss_ideal = smooth(np.exp(-0.03 * epochs) + 0.02 * np.random.randn(epochs.size), 10)
validation_loss_ideal = smooth(np.exp(-0.03 * epochs) + 0.1 + 0.02 * np.random.randn(epochs.size), 10)

training_loss_moderate = smooth(np.exp(-0.03 * epochs) + 0.02 * np.random.randn(epochs.size), 10)
validation_loss_moderate = smooth(np.exp(-0.03 * epochs) + 0.2 * (epochs / max(epochs)) + 0.02 * np.random.randn(epochs.size), 10)

training_loss_severe = smooth(np.exp(-0.03 * epochs) + 0.02 * np.random.randn(epochs.size), 10)
validation_loss_severe = smooth(np.exp(-0.03 * epochs) + 0.4 * (epochs / max(epochs)) ** 2 + 0.02 * np.random.randn(epochs.size), 10)

# Slicing data to start from epoch 20
epochs = epochs[10:]
training_loss_ideal = training_loss_ideal[10:]
validation_loss_ideal = validation_loss_ideal[10:]

training_loss_moderate = training_loss_moderate[10:]
validation_loss_moderate = validation_loss_moderate[10:]

training_loss_severe = training_loss_severe[10:]
validation_loss_severe = validation_loss_severe[10:]

plt.figure(figsize=(18, 6))

# Plotting the near-ideal model
plt.subplot(1, 3, 1)
plt.plot(epochs, training_loss_ideal, label='Training Loss', color='blue')
plt.plot(epochs, validation_loss_ideal, label='Validation Loss', color='orange')
plt.title('Last Model', color='black')
plt.xlabel('Epoch', color='black')
plt.ylabel('Loss', color='black')
plt.legend()
plt.grid(True)

# Plotting the moderate overfitting model
plt.subplot(1, 3, 2)
plt.plot(epochs, training_loss_moderate, label='Training Loss', color='blue')
plt.plot(epochs, validation_loss_moderate, label='Validation Loss', color='orange')
plt.title('Try solve Overfitting', color='black')
plt.xlabel('Epoch', color='black')
plt.ylabel('Loss', color='black')
plt.legend()
plt.grid(True)

# Plotting the severe overfitting model
plt.subplot(1, 3, 3)
plt.plot(epochs, training_loss_severe, label='Training Loss', color='blue')
plt.plot(epochs, validation_loss_severe, label='Validation Loss', color='orange')
plt.title('Overfitting', color='black')
plt.xlabel('Epoch', color='black')
plt.ylabel('Loss', color='black')
plt.legend()
plt.grid(True)

# Set white background for the figure and axes
plt.gcf().patch.set_facecolor('white')
for ax in plt.gcf().get_axes():
    ax.set_facecolor('white')
    ax.spines['top'].set_color('black')
    ax.spines['right'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.tick_params(axis='x', colors='black')
    ax.tick_params(axis='y', colors='black')
    ax.legend(frameon=False, loc='best')

plt.tight_layout()
plt.savefig('data/learning_rate_plots_white_background.png')
plt.show()

