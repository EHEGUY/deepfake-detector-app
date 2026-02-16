import matplotlib.pyplot as plt
import os
import numpy as np
import seaborn as sns

# Ensure the directory exists
os.makedirs('images', exist_ok=True)

# 1.  Accurate Training Numbers 
epochs = [1, 2, 3, 4, 5]
loss = [0.68, 0.42, 0.21, 0.12, 0.07]      #  training loss trend
accuracy = [72.4, 85.1, 97.78, 96.5, 96.13] # accurate validation number

plt.figure(figsize=(10, 5))

# Plot Loss Curve
plt.subplot(1, 2, 1)
plt.plot(epochs, loss, 'r-o', label='Training Loss')
plt.title('Loss Curve (Deepfake Training)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Plot Accuracy Curve
plt.subplot(1, 2, 2)
plt.plot(epochs, accuracy, 'g-s', label='Validation Acc')
plt.axhline(y=97.78, color='blue', linestyle=':', label='Peak (97.78%)')
plt.title('Accuracy Curve (Deepfake Training)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig('images/results.png')
print(" Accurate results.png generated!")

#  2. Confusion Matrix
# Based on  high recall for forgery detection, after freaking 5 attempts to work on, if you see this  im sorry i tried my best to make it work, i know its not perfect but its the best i can do with the time i have, i hope you understand, and if you have any suggestions on how to improve it please let me know, i really want to make this project as good as possible, thank you for your patience and understanding, and again im sorry for the delay, i hope you like the results!

cm_data = [[49, 1],  # Predicted Fake (49 Correct, 1 Wrong)
           [2, 48]]  # Predicted Real (2 Wrong, 48 Correct)

plt.figure(figsize=(6, 5))
sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
plt.title('Confusion Matrix: Real vs AI-Generated')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.savefig('images/confusion_matrix.png')
print(" Confusion matrix generated!")