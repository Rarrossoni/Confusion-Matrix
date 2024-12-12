import numpy as np
import matplotlib.pyplot as plt

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

def get_image(path):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


# Create confusion matrix and metrics functions
def confusion_matrix_metrics(y_true, y_pred):
    # Initialize the confusion matrix components
    TP = 0  # True Positives
    TN = 0  # True Negatives
    FP = 0  # False Positives
    FN = 0  # False Negatives

    # Calculate confusion matrix components
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            TP += 1
        elif true == 0 and pred == 0:
            TN += 1
        elif true == 0 and pred == 1:
            FP += 1
        elif true == 1 and pred == 0:
            FN += 1

    # Sensitivity / Recall
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Precision
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    # F-Score
    f_score = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    # Output the metrics and the confusion matrix
    confusion_matrix = {
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN
    }

    metrics = {
        "Sensitivity/Recall": sensitivity,
        "Precision": precision,
        "Accuracy": accuracy,
        "F-Score": f_score
    }

    return confusion_matrix, metrics

def plot_confusion_matrix(conf_matrix):
    TP = conf_matrix["TP"]
    TN = conf_matrix["TN"]
    FP = conf_matrix["FP"]
    FN = conf_matrix["FN"]

    matrix = [
        [TN, FP],
        [FN, TP]
    ]

    labels = [["TN", "FP"], ["FN", "TP"]]

    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap="Blues")
    plt.colorbar(cax)

    for i in range(2):
        for j in range(2):
            ax.text(x=j, y=i, s=f"{labels[i][j]}: {matrix[i][j]}", va='center', ha='center')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.title("Confusion Matrix")
    plt.show()