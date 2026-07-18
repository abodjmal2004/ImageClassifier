import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import confusion_matrix, roc_curve, auc

def load_and_preprocess_data(data_dir, categories, target_size=(40, 40)):
    """
    Loads images from the given directory, resizes, normalizes, and flattens them.
    """
    flat_data_arr = [] 
    target_arr = []
    
    print("Starting data loading and preprocessing...")
    for category in categories:
        print(f"Processing category: {category}")
        class_num = categories.index(category)
        
        # Checking both train and test subdirectories
        for split in ['train', 'test']:
            path = os.path.join(data_dir, split, category)
            if not os.path.exists(path):
                continue
                
            for img_name in os.listdir(path):
                try:
                    img_array = imread(os.path.join(path, img_name))
                    img_resized = resize(img_array, (target_size[0], target_size[1], 3))
                    flat_data_arr.append(img_resized.flatten())
                    target_arr.append(class_num)
                except Exception as e:
                    print(f"Error loading image {img_name}: {e}")

    X = np.array(flat_data_arr)
    y = np.array(target_arr)
    print(f"Data loading complete. Shape of X: {X.shape}, Shape of y: {y.shape}")
    return X, y

def save_confusion_matrix(y_true, y_pred, classes, save_path="images/confusion_matrix.png"):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

def save_roc_curve(y_true, y_prob, save_path="images/roc_curve.png"):
    """Plots and saves the ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"ROC curve saved to {save_path}")
