import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from data_preprocessing import load_and_preprocess_data
from model_evaluation import evaluate_model, plot_training_history
import numpy as np

def build_cnn_model(input_shape, num_classes):
    """
    Build a Convolutional Neural Network model.

    Parameters:
    input_shape (tuple): Shape of the input data
    num_classes (int): Number of output classes

    Returns:
    model: Compiled CNN model
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def train_model(X_train, y_train, X_val, y_val, input_shape, num_classes, epochs=20, batch_size=32):
    """
    Train the CNN model with early stopping and model checkpointing.

    Parameters:
    X_train, y_train: Training data
    X_val, y_val: Validation data
    input_shape: Shape of input data
    num_classes: Number of classes
    epochs: Number of training epochs
    batch_size: Batch size for training

    Returns:
    model: Trained model
    history: Training history
    """
    model = build_cnn_model(input_shape, num_classes)

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_accuracy')

    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint]
    )

    return model, history

if __name__ == "__main__":
    # Example usage with MNIST dataset
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255

    # Split training data into train and validation
    X_train, X_val, y_train, y_val = load_and_preprocess_data(X_train.reshape(X_train.shape[0], -1), y_train, test_size=0.1)
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_val = X_val.reshape(-1, 28, 28, 1)

    input_shape = (28, 28, 1)
    num_classes = 10

    model, history = train_model(X_train, y_train, X_val, y_val, input_shape, num_classes)

    # Evaluate the model
    eval_results = evaluate_model(model, X_test, y_test, class_names=[str(i) for i in range(10)])
    print(f"Test Accuracy: {eval_results['accuracy']:.4f}")

    # Plot training history
    plot_training_history(history)
