# Load Packages
import os

import pickle
import numpy as np
import random

from utils import get_image, confusion_matrix_metrics, plot_confusion_matrix

# Import the model
## the chad_or_will model are in my github called transfer learning
## Also, you can build your own model.
with open('chad_or_will.pkl', 'rb') as f:
    chad_or_will = pickle.load(f)

# Load data
root = 'Images'

categories = [x[0] for x in os.walk(root) if x[0]][1:]

data = []
for c, category in enumerate(categories):
    images = [os.path.join(dp, f) for dp, dn, filenames
              in os.walk(category) for f in filenames
              if os.path.splitext(f)[1].lower() in ['.jpg','.png','.jpeg']]
    for img_path in images:
        img, x = get_image(img_path)
        data.append({'x':np.array(x[0]), 'y':c})


# Randomize order data
random.shuffle(data)

# Separate features and labels
x, y = np.array([d["x"] for d in data]), [d["y"] for d in data]

# Normalize data
x = x.astype('float32') / 255.

# Predict values
y_pred = chad_or_will.predict(x)

# Convert predicted values 
y_pred_class = []
for i in y_pred:
    if i[0] >= 0.5:
        y_pred_class.append(0)
    else:
        y_pred_class.append(1)

# Apllying confusion matrix and metrics:
conf_matrix, metrics = confusion_matrix_metrics(y, y_pred_class)

print("Confusion Matrix:")
plot_confusion_matrix(conf_matrix)

print("Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
