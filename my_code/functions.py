import pennylane as qml
from pennylane import numpy as np

# we have 19 possible angles
POSSIBLE_ANGLES = np.linspace(0, 2*np.pi, 19).tolist()
POSSIBLE_AMINOACIDS = ['A', 'G', 'I', 'L', 'V', 'F', 'W', 'Y', 'D', 'E', 'R', 'H', 'K', 'S', 'T', 'C', 'M', 'N', 'Q']

# dic of aminoacids and their corresponding angles
AMINOACIDS_ANGLES = {amino: angle for angle, amino in zip(POSSIBLE_ANGLES, POSSIBLE_AMINOACIDS)}

def string_to_angles(string):
    angles = []
    for amino in string:
        angles.append(AMINOACIDS_ANGLES[amino])
    return angles

def read_data_file(file_path):
    strings = []
    numbers = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                string, number = line.split()
                strings.append(string)
                numbers.append(float(number))

    return strings, numbers

def create_validating_set(X, Y, percentage=0.1):
    # we will use percentage% of the data as validation set
    n = len(X)
    n_validating = int(n*percentage)

    # we will use random indexes to select the validation set
    indexes = np.random.choice(n, n_validating, replace=False)
    X_validating = X[indexes]
    Y_validating = Y[indexes]

    # we will use the rest of the data as training set
    X_training = np.delete(X, indexes, axis=0)
    Y_training = np.delete(Y, indexes, axis=0)

    return X_training, Y_training, X_validating, Y_validating