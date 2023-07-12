import pennylane as qml
from pennylane import numpy as np

# we have 19 possible angles
possible_angles = np.linspace(0, 2*np.pi, 19).tolist()
possible_aminoacids = ['A', 'G', 'I', 'L', 'V', 'F', 'W', 'Y', 'D', 'E', 'R', 'H', 'K', 'S', 'T', 'C', 'M', 'N', 'Q']

# dic of aminoacids and their corresponding angles
aminoacids_angles = {amino: angle for angle, amino in zip(possible_angles, possible_aminoacids)}

def string_to_angles(string):
    angles = []
    for amino in string:
        angles.append(aminoacids_angles[amino])
    return angles