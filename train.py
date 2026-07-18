import os
import joblib
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from utils import load_and_preprocess_data, save_confusion_matrix, save_roc_curve

# --- 1. Configurations ---
DATA_DIR = 'dataset/catsAndDogs40/'
CATEGORIES = ['cat', 'dog']
IMAGE_SIZE = (40, 40)
MODEL_SAVE_PATH = 'svm_model.pkl'

os.makedirs('images', exist_ok=True)

def main():
    # --- 2. Data Preparation ---
    X, y = load_and_preprocess_data(DATA_DIR, CATEGORIES, IMAGE_SIZE)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    # --- 3. Model Building & Hyperparameter Tuning ---
    print("Setting up GridSearchCV for SVM...")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.0001, 0.001, 0.1, 1],
        'kernel': ['rbf', 'poly']
    }
    
    svc = svm.SVC(probability=True)
    model = GridSearchCV(svc, param_grid, cv=3, verbose=2, n_jobs=-1)
    
    # --- 4. Model Training ---
    print("Training the model (this may take a while)...")
    model.fit(X_train, y_train)
    print(f"Best parameters found: {model.best_params_}")

    # --- 5. Model Evaluation ---
    print("Evaluating the model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CATEGORIES))
    
    save_confusion_matrix(y_test, y_pred, CATEGORIES)
    save_roc_curve(y_test, y_prob)

    # --- 6. Save Model ---
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"Model successfully saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
