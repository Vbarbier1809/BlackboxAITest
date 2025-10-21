import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Evaluate the model performance on test data.

    Parameters:
    model: Trained model
    X_test (array-like): Test feature data
    y_test (array-like): Test target data
    class_names (list): List of class names for the confusion matrix

    Returns:
    dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1) if y_pred.ndim > 1 else y_pred

    # Calculate metrics
    report = classification_report(y_test, y_pred_classes, target_names=class_names, output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

    return {
        'classification_report': report,
        'confusion_matrix': cm,
        'accuracy': report['accuracy']
    }

def plot_training_history(history):
    """
    Plot the training history of the model.

    Parameters:
    history: Training history object from model.fit()
    """
    plt.figure(figsize=(12, 4))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
