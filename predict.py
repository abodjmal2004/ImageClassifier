import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.transform import resize
import os

CATEGORIES = ['cat', 'dog']

def predict_image(image_path, model_path='svm_model.pkl'):
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print("Error: Model not found. Please run train.py first.")
        return

    try:
        img_array = imread(image_path)
        img_resized = resize(img_array, (40, 40, 3))
        flat_image = np.array([img_resized.flatten()])
        
        prediction = model.predict(flat_image)
        predicted_class = CATEGORIES[prediction[0]]
        
        print(f"The predicted image is: {predicted_class.upper()}")
        
        os.makedirs('images', exist_ok=True)
        plt.imshow(img_array)
        plt.title(f"Prediction: {predicted_class.upper()}")
        plt.axis('off')
        plt.savefig('images/sample_prediction.png')
        print("Prediction image saved to images/sample_prediction.png")
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Cat or Dog from an image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file")
    args = parser.parse_args()
    
    predict_image(args.image)
