import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(X, y, test_size=0.2, random_state=42):
    """
    Load and preprocess the data for training.

    Parameters:
    X (array-like): Feature data
    y (array-like): Target data
    test_size (float): Proportion of the dataset to include in the test split
    random_state (int): Random state for reproducibility

    Returns:
    X_train, X_test, y_train, y_test: Preprocessed training and testing data
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

def augment_data(X, y, augmentation_factor=2):
    """
    Augment the data by adding noise or other transformations.

    Parameters:
    X (array-like): Feature data
    y (array-like): Target data
    augmentation_factor (int): Factor by which to augment the data

    Returns:
    X_augmented, y_augmented: Augmented data
    """
    X_augmented = []
    y_augmented = []

    for i in range(len(X)):
        X_augmented.append(X[i])
        y_augmented.append(y[i])

        for _ in range(augmentation_factor - 1):
            noise = np.random.normal(0, 0.1, X[i].shape)
            X_augmented.append(X[i] + noise)
            y_augmented.append(y[i])

    return np.array(X_augmented), np.array(y_augmented)
