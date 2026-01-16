# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:50:52 2024

@author: Z984222
"""
from matplotlib import pyplot as plt

# Data
train_accuracy = [42, 52, 62, 62, 66, 70, 65, 70, 69, 69, 72, 75, 72, 72, 75, 75.75, 70, 74, 75, 75.25, 74.3, 76,77.5, 78,78, 75.25, 75.5, 76.08, 76.25, 76.42, 77.47, 78, 79.25, 79.125, 79.375,  79.35, 79.47, 76.62, 76.31, 76.70, 77.25,  80.625, 81.25, 80.58, 80.5, 80.325, 79.875]
train_loss = [0.09, 0.08, 0.07, 0.064, 0.061, 0.054, 0.06, 0.054, 0.052, 0.052, 0.050, 0.046, 0.049, 0.046, 0.044, 0.043, 0.046, 0.028, 0.04, 0.04, 0.04, 0.04,0.039, 0.037, 0.038, 0.04, 0.041, 0.04, 0.039, 0.039, 0.038, 0.038, 0.036, 0.036, 0.0357, 0.0358, 0.0356, 0.041, 0.039, 0.0388,  0.0382, 0.0361, 0.034, 0.035, 0.0354, 0.0356, 0.0359]
batches = [50 * i for i in range(1, len(train_accuracy) + 1)]

# Plotting
plt.figure(figsize=(12, 6))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(batches, train_accuracy, marker='o', linestyle='-', color='b', label='Training Accuracy')
plt.title('Training Accuracy Over Mini-batches')
plt.xlabel('Number of Mini-batches')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)
plt.grid(True)
plt.legend()
plt.xticks(batches, rotation=45, ha='right')
plt.yticks(range(0, 101, 10))

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(batches, train_loss, marker='o', linestyle='-', color='r', label='Training Loss')
plt.title('Training Loss Over Mini-batches')
plt.xlabel('Number of Mini-batches')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.xticks(batches, rotation=45, ha='right')
plt.yticks([i/100 for i in range(0, 101, 5)])

plt.tight_layout()
plt.show()
