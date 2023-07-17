import pennylane as qml
from pennylane import numpy as np
import random


###--- AMINOACIDS DATA ---###

# amminoacids maps
type_map = {
    'GLY': ['G', 0],
    'ILE': ['I', 1],
    'LEU': ['L', 2],
    'MET': ['M', 3],
    'PHE': ['F', 4],
    'TRP': ['W', 5],
    'TYR': ['Y', 6],
    'VAL': ['V', 7],
    'ARG': ['R', 8],
    'LYS': ['K', 9],
    'SER': ['S', 10],
    'THR': ['T', 11],
    'ASN': ['N', 12],
    'GLN': ['Q', 13],
    'HIE': ['H', 14],
    'ALA': ['A', 15],
    'CYS': ['C', 16],
    'ASP': ['D', 17],
    'GLU': ['E', 18],
}
POSSIBLE_ANGLES = np.linspace(0, 2*np.pi, len(type_map)).tolist()
POSSIBLE_AMINOACIDS = [v[0] for v in type_map.values()]
code_map = {k: v[1] for k, v in type_map.items()}
letter_map = {v[0]: v[1] for v in type_map.values()}
angles_map = {amino: angle for angle, amino in zip(POSSIBLE_ANGLES, POSSIBLE_AMINOACIDS)}

def read_energies_file(file_single_path, file_pair_path):
    ### SINGLE ENERGIES ###

    # Read the data file
    with open(file_single_path, 'r') as file:
        lines = file.readlines()

    # Initialize an empty matrix
    h = np.zeros((12, 19), requires_grad=False)

    # Process each line in the file
    residue_idx = -1
    for line in lines:
        if not line.startswith('#########'):
            line = line.split()
            residue_idx = int(line[1])-1
            type_idx = code_map[line[3][-3:]]
            value = float(line[-1])
            h[residue_idx, type_idx] = value

    ### PAIR WISE ENERGIES ###

    # Read the data file
    with open(file_pair_path, 'r') as file:
        lines = file.readlines()

    # Initialize an empty matrix
    J = np.zeros((12, 12, 19, 19), requires_grad=False)

    # Process each line in the file
    residue_idx = -1
    for line in lines:
        if not line.startswith('#########'):

            line = line.split()

            residue_1_idx = int(line[1])-1
            residue_2_idx = int(line[5])-1
            type_1_idx = code_map[line[3][-3:]]
            type_2_idx = code_map[line[7][-3:]]
            value = float(line[-1])
            J[residue_1_idx, residue_2_idx, type_1_idx, type_2_idx] = value

    return h, J

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

def string_to_angles(string):
    return [letter_map[letter] for letter in string]

def string_to_energy(string, h, J):

    vector = string_to_angles(string)

    energy = np.sum(h[np.arange(len(vector)), vector]) + \
             np.sum([J[i, j, vector[i], vector[j]] for i in range(len(vector)) for j in range(len(vector))])

    return energy.item()


def generate_random_data_from_energies(file_single_path, file_pair_path, file_out_path, max_score=None, n_samples=1000):

    h, J = read_energies_file(file_single_path, file_pair_path)

    n_samples_to_do = n_samples
    n_samples = 0

    with open(file_out_path, 'w') as file:

        while n_samples < n_samples_to_do:
            string = ''.join(random.choice(POSSIBLE_AMINOACIDS) for _ in range(12))
            score = string_to_energy(string, h, J)

            if max_score is None or score <= max_score:
                file.write(f'{string}  {score}\n')
                n_samples += 1


###--- QML FUNCTIONS ---###

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

