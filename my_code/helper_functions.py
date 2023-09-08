import pennylane as qml
from pennylane import numpy as np
import random, os, uuid


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

# def read_data_file(file_path):
#     strings = []
#     numbers = []

#     with open(file_path, 'r') as file:
#         for line in file:
#             line = line.strip()
#             if line:
#                 string, number = line.split()
#                 strings.append(string)
#                 numbers.append(float(number))

#     return strings, numbers

def read_data_file(
        file_path,  
        skip_first_line=False, 
        delimiter=' ', 
        columns_to_return=[0, 1],
        which_columns_are_numbers=[1]
    ):

    columns = []
    for _ in columns_to_return:
        columns.append([])

    with open(file_path, 'r') as file:
        for i, line in enumerate(file):

            if i == 0 and skip_first_line: continue #skip the first line

            line = line.strip()

            if line:
                items = line.split(delimiter)
                i_column = 0
                for n, item in enumerate(items):
                    if n in columns_to_return:
                        columns[i_column].append(item if n not in which_columns_are_numbers else float(item))
                        i_column = i_column + 1

    return columns

def string_to_angles(string):
    return [letter_to_angle[letter] for letter in string]

def string_to_numbers(string):
    return [letter_to_number[letter] for letter in string]

def string_to_vector(string):
    return np.array([letter_to_vector[letter] for letter in string]).flatten()

def string_to_energy(string, h, J):

    vector = string_to_numbers(string)

    energy = np.sum(h[np.arange(len(vector)), vector]) + \
             np.sum([J[i, j, vector[i], vector[j]] for i in range(len(vector)) for j in range(len(vector))])

    return energy.item()

def string_to_single_energy(string, h):    
    vector = string_to_numbers(string)
    energy = np.sum(h[np.arange(len(vector)), vector])
    return energy.item()

def string_to_pair_energy(string, J):
    vector = string_to_numbers(string)
    energy = np.sum([J[i, j, vector[i], vector[j]] for i in range(len(vector)) for j in range(len(vector))])
    return energy.item()

def generate_random_data_from_energies(file_single_path, file_pair_path, file_out_path, max_score=None, n_samples=1000, n_amino_acids=12, only_single_energies=False, only_pair_energies=False):

    if only_single_energies and only_pair_energies:
        raise Exception("You can't set both only_single_energies and only_pair_energies to True")

    h, J = read_energies_file(file_single_path, file_pair_path)

    n_samples_to_do = n_samples
    n_samples = 0

    with open(file_out_path, 'w') as file:

        while n_samples < n_samples_to_do:
            string = ''.join(random.choice(POSSIBLE_AMINOACIDS_LETTER) for _ in range(n_amino_acids))

            if only_single_energies: score = string_to_single_energy(string, h)
            elif only_pair_energies: score = string_to_pair_energy(string, J)
            else:                    score = string_to_energy(string, h, J)

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
def get_name_file_to_save(name_notebook, initial_path, extension, version=None, postfix=""):

    dict_extension_folder = {
        "png": "plots", 
        "pth": "models", 
        "pdf": "pdfs", 
        "ipynb": "notebooks", 
        "txt": "txts", 
        "version": "versions",
        "pickle": "pickled_objects"
    }

    day = name_notebook[:4]
    folder = initial_path + "checkpoints/" + day
    if version is None:
        filename = folder + "/" + dict_extension_folder[extension] + "/" + name_notebook[:-6] + postfix + "." + extension
    else:
        filename = folder + "/" + dict_extension_folder[extension] + "/" + name_notebook[:-6] + postfix + "_" + str(version) + "." + extension

    # Check if the folder exists and if it doesn't exist, create it
    if not os.path.exists(folder):

        os.makedirs(folder)
        print(f"Folder '{folder}' created successfully.")

        for subfolder in dict_extension_folder.values():
            os.makedirs(folder + "/" + subfolder)
            print(f"Folder '{folder}/{subfolder}' created successfully.")
            #create .gitkeep file
            with open(folder + "/" + subfolder + "/.gitkeep", 'w') as file:
                pass
            if subfolder == "versions":
                with open(folder + "/" + subfolder + "/.gitkeep", 'w') as file:
                    file.write("0")


    # check if the file exists and print a message
    if os.path.exists(filename) and not version is None:
        print("The file {} already exists, it will be replaced".format(filename))

    return filename

def read_version(name_notebook, initial_path, return_filename=False):

    filename = get_name_file_to_save(name_notebook, initial_path, "version")

    # if the file exists, read the version, otherwise create the file and set the version to 0
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            version = int(file.readline())
    else:
        with open(filename, 'w') as file:
            version = 0
            file.write(str(version))

    if return_filename:
        return version, filename
    return version

def update_version(name_notebook, initial_path):

    version, filename = read_version(name_notebook, initial_path, return_filename=True)

    # the version is increased by 1
    with open(filename, 'w') as file:
        file.write(str(version+1))

    return version+1

   


def should_stop_training(losses, lookback_epochs=5, threshold_slope=0.001, threshold_std_dev=0.1):
    """
    Determines if training should stop based on the standard deviation of the last `lookback_epochs` losses.
    
    Parameters:
    - losses (list): List of loss values, where the most recent loss is the last element.
    - lookback_epochs (int): Number of recent epochs to consider.
    - threshold_slope (float): Maximum slope of the linear regression of the losses.
    - threshold_std_dev (float): Maximum standard deviation of the losses.
    
    Returns:
    - bool: True if training should stop, False otherwise.
    """
    
    # If there aren't enough epochs yet, continue training
    if len(losses) < lookback_epochs:
        return False
    
    # Extract the last `lookback_epochs` losses
    recent_losses = losses[-lookback_epochs:]
    
    # Calculate the standard deviation of the recent losses
    std_dev = sum([(x - sum(recent_losses) / lookback_epochs) ** 2 for x in recent_losses]) ** 0.5 / lookback_epochs

    # If the standard deviation is above the threshold, continue training
    if std_dev > threshold_std_dev:
        return False
    
    # Calculate the slope of the linear regression of the recent losses
    slope = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]

    # If the negative slope is above the threshold, continue training
    if -slope > threshold_slope:
        return False

    # Otherwise, stop training
    return True