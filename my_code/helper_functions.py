import pennylane as qml
from pennylane import numpy as np
import random, os, csv, json
import matplotlib.pyplot as plt
import pickle
import datetime


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
number_to_letter = {v:k for k,v in letter_to_number.items()}
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

def numbers_to_string(numbers):
    return ''.join([number_to_letter[number] for number in numbers])

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


###--- PLOTS ---###
def plot_validation(results, fig_size=(6,6)):
    y_test = results.validation['y_test']
    y_pred = results.validation['y_prediction']
    mean = np.mean(results.validation['losses'])
    r_sq = results.validation['r_squared']

    plt.figure(figsize=fig_size)
    plt.scatter(y_test, y_pred, color='r', label='Actual vs. Predicted', alpha=0.1)
    plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'k--', lw=2, label='1:1 Line')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Predictions vs. True Values (avg: {:.4f}, R$^2$: {:.4f})'.format(mean, r_sq))
    plt.legend()
    plt.show()

def plot_losses_training(results, fig_size=(6,6)):
    for x, y, xlabel, title in [
        (None, 'loss_batch', 'Batch', 'Loss per batch'), 
        ('n_epoch', 'loss_epoch', 'Epoch', 'Loss per epoch'), 
        ('n_epoch', 'loss_validation_epoch', 'Epoch', 'Loss per epoch (validation)')
    ]:
        x = getattr(results, x).history if x else None
        y = getattr(results, y).history

        plt.figure(figsize=fig_size)
        if x is None: 
            plt.plot(y)
        else: 
            plt.plot(x, y)
        plt.xlabel(xlabel)
        plt.ylabel('Loss')
        plt.title(title)
        plt.show()

COLORS = [
    (0.121, 0.466, 0.705),  # Blue
    (1.0, 0.498, 0.054),    # Orange
    (0.172, 0.627, 0.172),  # Green
    (0.839, 0.153, 0.157),  # Red
    (0.580, 0.404, 0.741),  # Purple
    (0.549, 0.337, 0.294),  # Brown
    (0.890, 0.466, 0.760),  # Pink
    (0.498, 0.498, 0.498),  # Grey
    (0.737, 0.741, 0.133),  # Olive Green
    (0.090, 0.745, 0.812),  # Cyan
    (0.682, 0.780, 0.909),  # Lighter Blue
    (1.0, 0.733, 0.471),    # Lighter Orange
    (0.596, 0.875, 0.541),  # Lighter Green
    (1.0, 0.598, 0.6),      # Lighter Red
    (0.772, 0.690, 0.835),  # Lighter Purple
    (0.768, 0.611, 0.580),  # Lighter Brown
    (0.969, 0.714, 0.824),  # Light Pink
    (0.780, 0.780, 0.780),  # Light Grey
    (0.858, 0.859, 0.552),  # Light Olive Green
    (0.619, 0.854, 0.898)   # Light Cyan
]

def plot_w_poly_fit(x, y, degree=2, options_data={}, options_fit={}):
    plt.plot(x, y, **options_data)
    if not degree is None:
        z = np.polyfit(x, y, degree)
        p = np.poly1d(z)
        x_pred = np.linspace(min(x), max(x), 100)
        plt.plot(x_pred, p(x_pred), **options_fit)

###--- STRINGS ---###
def replace_string(string, replace):
    for old, new in replace:
        string = string.replace(old, new)
    return string


###--- SAVE DATA ---###
def save_csv(dict_to_save:dict, file_name, initial_path):

    file_path = initial_path + '/saved/' + file_name + '.csv'

    #add one row to the csv file
    with open(file_path, 'a', newline='') as file:
        csv_writer = csv.DictWriter(file, fieldnames=dict_to_save.keys())
        csv_writer.writerows([dict_to_save])

def save_json(dict_to_save:dict, file_name, folder, initial_path, day=None):

    folder = initial_path + '/saved/' + folder + '/'
    if not day is None:
        folder = folder + day + '/'
        if not os.path.exists(folder): #check if the folder day exists in folder
            os.makedirs(folder)
            print(f"Folder '{folder}' created successfully.")
    
    file_name = folder + file_name + '.json'

    #dump the dict in the json file
    with open(file_name, 'w') as file:
        json.dump(dict_to_save, file, indent=4)

def save_pickle(object, file_name, folder, initial_path, day=None):

    folder = initial_path + '/saved/' + folder + '/'
    if not day is None:
        folder = folder + day + '/'
        if not os.path.exists(folder): #check if the folder day exists in folder
            os.makedirs(folder)
            print(f"Folder '{folder}' created successfully.")
    
    file_name = folder + file_name + '.pickle'

    #dump
    with open(file_name, 'wb') as f:
        pickle.dump(object, f, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(file_name, folder, initial_path, day=None):

    folder = initial_path + '/saved/' + folder + '/'
    if not day is None:
        folder = folder + day + '/'
    
    file_name = folder + file_name + '.pickle'

    #load
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    

###--- LOAD DATA ---###
def get_from_csv(file_path, to_search, to_return):

    with open(file_path, mode='r') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if np.all([row[k] == v for k,v in to_search.items()]):
                return [row[k] for k in to_return]
            
    return [None for _ in to_return]




###--- OTHER FUNCTIONS ---###

def get_version(initial_path, notebook, file='model_uuids'): 
    ''' look for the last version of the notebook and return the next one '''

    version = 0

    with open(initial_path + '/saved/' + file + '.csv', 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            if row['notebook'] == notebook:
                version = int(row['version'])

    return version + 1

def get_day():
    return datetime.datetime.now().strftime("%m%d")
   


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

def find_intervals(intervals_list, values_list):
    """
    Finds the intervals in which the values of the 'values_list' are contained in the 'intervals_list'.
    Returns a list of tuples (interval_start, interval_end). One tuple for each value in the 'values_list'.    
    """
    results = []
    for num in values_list:
        for i in range(len(intervals_list) - 1):
            if intervals_list[i] <= num <= intervals_list[i + 1]:
                results.append((intervals_list[i], intervals_list[i + 1]))
                break
        else:
            # If no interval is found, add the first and last values of the first list.
            results.append((intervals_list[0], intervals_list[-1]))
    return results

def r_squared(x, y):
    """
    Calculates the R^2 value of the linear regression of y with respect to x.
    """
    x, y = np.array(x), np.array(y)
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    return 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)

