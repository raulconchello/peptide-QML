import pennylane as qml
from pennylane import numpy as np
import random, os


###--- AMINOACIDS DATA ---###

# amminoacids maps  
''' 
 - code: is a three letter code for the aminoacid
 - letter: is the one letter code for the aminoacid
 - number: is the number associated to the aminoacid, also used as index in the matrices
 - angle: is the angle associated to the aminoacid
'''
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
n_aminoacids = len(type_map)
POSSIBLE_ANGLES = np.linspace(0, 2*np.pi, len(type_map)).tolist()
POSSIBLE_AMINOACIDS_LETTER = [v[0] for v in type_map.values()]
code_to_number = {k: v[1] for k, v in type_map.items()}
letter_to_number = {v[0]: v[1] for v in type_map.values()}
letter_to_angle = {letter: angle for angle, letter in zip(POSSIBLE_ANGLES, POSSIBLE_AMINOACIDS_LETTER)}
letter_to_vector = {v[0]: np.eye(n_aminoacids)[v[1]] for v in type_map.values()}

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
            type_idx = code_to_number[line[3][-3:]]
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
            type_1_idx = code_to_number[line[3][-3:]]
            type_2_idx = code_to_number[line[7][-3:]]
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
    return [letter_to_angle[letter] for letter in string]

def string_to_numbers(string):
    return [letter_to_number[letter] for letter in string]

def string_to_vector(string):
    return np.array([letter_to_vector[letter] for letter in string])

def string_to_energy(string, h, J):

    vector = string_to_numbers(string)

    energy = np.sum(h[np.arange(len(vector)), vector]) + \
             np.sum([J[i, j, vector[i], vector[j]] for i in range(len(vector)) for j in range(len(vector))])

    return energy.item()


def generate_random_data_from_energies(file_single_path, file_pair_path, file_out_path, max_score=None, n_samples=1000, n_amino_acids=12):

    h, J = read_energies_file(file_single_path, file_pair_path)

    n_samples_to_do = n_samples
    n_samples = 0

    with open(file_out_path, 'w') as file:

        while n_samples < n_samples_to_do:
            string = ''.join(random.choice(POSSIBLE_AMINOACIDS_LETTER) for _ in range(n_amino_acids))
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



###--- SAVE DATA ---###
def get_name_file_to_save(name_notebook, initial_path, extension, version, postfix=""):

    dict_extension_folder = {"png": "plots", "pth": "models", "pdf": "pdfs", "ipynb": "notebooks", "txt": "txts"}

    day = name_notebook[:4]
    folder = initial_path + "checkpoints/" + day
    filename = folder + "/" + dict_extension_folder[extension] + "/" + name_notebook[:-6] + postfix + "_" + str(version) + "." + extension

    # Check if the folder exists and if it doesn't exist, create it
    if not os.path.exists(folder):

        os.makedirs(folder)
        print(f"Folder '{folder}' created successfully.")

        for subfolder in ["models", "plots", "pdfs", "notebooks", "txts"]:
            os.makedirs(folder + "/" + subfolder)
            print(f"Folder '{folder}/{subfolder}' created successfully.")


    # check if the file exists and print a message
    if os.path.exists(filename):
        print("The file {} already exists, it will be replaced".format(filename))

    return filename