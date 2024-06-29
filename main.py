# import matplotlib.pyplot as plt
#
# # Define the metrics
# metrics = {
#     'Accuracy': 0.87,
#     'Precision': 0.86,
#     'Recall': 0.87,
#     'F1 Score': 0.86
# }
#
# # Create a figure and axis
# fig, ax = plt.subplots()
#
# # Hide axes
# ax.xaxis.set_visible(False)
# ax.yaxis.set_visible(False)
#
# # Remove the frame
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
#
# # Add text
# text_str = '\n'.join([f'{key}: {value}' for key, value in metrics.items()])
# ax.text(0.5, 0.5, text_str, horizontalalignment='center', verticalalignment='center', fontsize=15, family='monospace')
#
# # Save the figure
# plt.savefig('metrics_plot.png')
#
# # Show the plot
# plt.show()
import matplotlib.pyplot as plt
import numpy as np
import random

# Function to generate random metrics
# def generate_random_metrics():
#     return {
#         'Accuracy': round(random.uniform(0.8, 0.9), 2),
#         'Precision': round(random.uniform(0.8, 0.9), 2),
#         'Recall': round(random.uniform(0.8, 0.9), 2),
#         'F1 Score': round(random.uniform(0.8, 0.9), 2)
#     }
#
#
# # Function to create and save the plot
# def create_metrics_plot(metrics, index):
#     fig, ax = plt.subplots()
#
#     # Set black background
#     fig.patch.set_facecolor('black')
#     ax.set_facecolor('black')
#
#     # Hide axes
#     ax.xaxis.set_visible(False)
#     ax.yaxis.set_visible(False)
#
#     # Remove the frame
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.spines['left'].set_visible(False)
#     ax.spines['bottom'].set_visible(False)
#
#     # Add text with white color
#     text_str = '\n'.join([f'{key}: {value}' for key, value in metrics.items()])
#     ax.text(0.5, 0.5, text_str, horizontalalignment='center', verticalalignment='center', fontsize=15,
#             family='monospace', color='white')
#
#     # Save the figure
#     plt.savefig(f'data/metrics_plot_{index}.png')
#     plt.close()
#
#
# # Generate and save 10 different plots
# for i in range(1, 11):
#     metrics = generate_random_metrics()
#     create_metrics_plot(metrics, i)
#
# print("10 metric plots have been generated and saved.")

import matplotlib.pyplot as plt

# Define the metrics for overfitting, underfitting, and well-fitted models
metrics_overfitting = {
    'Training Accuracy': 0.90,
    'Test Accuracy': 0.60,
    'Precision': 0.58,
    'Recall': 0.60,
    'F1 Score': 0.59
}

metrics_underfitting = {
    'Training Accuracy': 95.4,
    'Test Accuracy': 0.89,
    'Precision': 0.88,
    'Recall': 0.89,
    'F1 Score': 0.88
}

metrics_well_fitted = {
    'Training Accuracy': 0.75,
    'Test Accuracy': 0.70,
    'Precision': 0.69,
    'Recall': 0.70,
    'F1 Score': 0.69
}


# Function to create and save the plot
def create_metrics_plot(metrics, filename):
    fig, ax = plt.subplots()

    # Set black background
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Hide axes
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    # Remove the frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add text with white color
    text_str = '\n'.join([f'{key}: {value}' for key, value in metrics.items()])
    ax.text(0.5, 0.5, text_str, horizontalalignment='center', verticalalignment='center', fontsize=15,
            family='monospace', color='white')

    # Save the figure
    plt.savefig(f'data/{filename}.png')
    plt.close()


# Generate and save the plots
create_metrics_plot(metrics_overfitting, 'metrics_overfitting')
create_metrics_plot(metrics_underfitting, 'metrics_underfitting')
create_metrics_plot(metrics_well_fitted, 'metrics_well_fitted')

print("Metric plots for overfitting, underfitting, and well-fitted models have been generated and saved.")
