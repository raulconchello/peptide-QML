
# imports
import uuid
import datetime
import time as t
import numpy as np
from copy import deepcopy
from itertools import product

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# my imports
from . import helper_classes as c
from . import helper_functions as f

torch.set_default_dtype(torch.float64)

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, predictions, targets, mean_squared_targets=None):
        mean_squared_targets = torch.mean(targets**2) if mean_squared_targets is None else mean_squared_targets        
        return nn.MSELoss()(predictions, targets) / mean_squared_targets
    
class MSRELoss(nn.Module):
    def __init__(self):
        super(MSRELoss, self).__init__()

    def forward(self, predictions, targets):
        return torch.mean(torch.sum((predictions - targets)**2 / targets**2))
    
class Reshape(nn.Module):
    def __init__(self):
        super(Reshape, self).__init__()

    def forward(self, x):
        dims = len(x.shape)
        return torch.transpose(x, dims-2, dims-1).reshape(x.shape[-3] if dims==3 else 1, x.shape[-2]*x.shape[-1]).squeeze()


def computed_required(func):
    def wrapper(self, *args, **kwargs):
        if not self.computed:
            raise ValueError("You need to compute the validation first. Use validation.compute(pct=pct) to compute the validation for the test dataset.")
        return func(self, *args, **kwargs)
    return wrapper

class Model(nn.Module):

    def __str__(self):
        return super().__str__() + ";\nQuantum layer: " + str(getattr(self, 'quantum_layer', None))

    # loss properties
    @property
    def loss_train(self):
        return self.loss_function(self(self.data.x_train), self.data.y_train).item()
    @property
    def loss_test_train(self):
        return self.loss_function(self(self.data.x_test_ptc), self.data.y_test_ptc).item()
    @property
    def loss_test(self):
        return self.loss_function(self(self.data.x_test), self.data.x_test).item()
    
    # name properties
    @property
    def file_name(self):
        return self.name_notebook + "-" + str(self.version)
    
    # number of parameters
    @property
    def n_parameters(self):
        return sum([p.numel() for p in self.parameters()])
    
    def set_quantum_layer(self, quantum_layer):
        self.quantum_layer = quantum_layer

    def set_name_and_path(self, name_notebook, initial_path):
        self.name_notebook = name_notebook
        self.initial_path = initial_path

    def set_sweep_point(self, sweep_uuid, sweep_point, day):
        self.sweep_uuid = sweep_uuid
        self.sweep_point = sweep_point
        self.day = day

    def initialize_params(self, initialization_options= [{}]):

        self.initialization_options = initialization_options

        for io in initialization_options:
            getattr(nn.init, io['type'])(getattr(self[io['layer']], io['name']), **io['options'])

    def optimize_params(
        self,
        # save model info
        save_model_info = True,
        description = ' ',
        # data
        data = None,
        # train options
        loss_function = nn.MSELoss,
        optimizer = optim.SGD,
        optimizer_options = {'lr': 0.05},
        num_epochs = 25,
        batch_size = 32,
        print_batch = True,
        keep_track_params = False,
        # validation
        validation_pct = 0.1,
        validation_print_n = 3,
        # Stop training
        stop_training_options = {},
        # metadata
        metadata = {},
    ):
        self.eval()
        
        #check if data is given and if isinstance of c.data
        if data is None:
            raise ValueError("You need to give a Data object.")
        if not isinstance(data, c.Data):
            raise ValueError("The data object needs to be a Data object.")
        self.data = data

        # save metadata
        self.metadata = metadata

        # save training inputs
        self.training_inputs = {
            'loss_function': str(loss_function),
            'optimizer': str(optimizer),
            'optimizer_options': optimizer_options,
            'initialization_options': getattr(self, 'initialization_options', [{}]),
            'data': str(data.uuid),
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'print_batch': print_batch,
            'keep_track_params': keep_track_params,
            'validation_pct': validation_pct,
            'validation_print_n': validation_print_n,
            'stop_training_options': stop_training_options,
        }
        self.version, self.uuid, self.day = f.get_version(self.initial_path, self.name_notebook), uuid.uuid4(), getattr(self, 'day', f.get_day())

        # time
        start_time = t.time()

        # loss function and optimizer
        self.loss_function = loss_function()
        self.optimizer = optimizer(self.parameters(), **optimizer_options)

        #data
        self.data.set_test_ptc(validation_pct)

        #save model info
        if save_model_info:
            self.save_model_info(description=description)

        # keep track of results
        self.results = c.Results(
            model_uuid = self.uuid,
            file_name = self.file_name,
            day = self.day,
            initial_path = self.initial_path,
            metadata = metadata,
            loss_batch  = c.keep(last=False, best=False, history=True),
            time_batch  = c.keep(last=False, best=False, history=True),
            n_epoch     = c.keep(last=True, best=True, history=True),
            loss_epoch  = c.keep(last=True, best=True, history=True),
            loss_validation_epoch = c.keep(last=True, best=True, history=validation_pct>0),
            parameters_epoch  = c.keep(last=True, best=True, history=keep_track_params),
            time_epoch        = c.keep(last=True, best=True, history=True),
        )
        self.results.update(
            update_leader = 'loss_validation_epoch' if validation_pct else 'loss_epoch',
            n_epoch = 0,
            loss_epoch = self.loss_train,
            loss_validation_epoch = self.loss_test_train if validation_pct else None,
            parameters_epoch = deepcopy(self.state_dict()),
            time_epoch = t.time() - start_time,
        )
        print('Epoch [{}/{}], Loss epoch: {:.4f}, Loss validation: {:.4f}'.format(0, num_epochs, self.results.loss_epoch.last, self.results.loss_validation_epoch.last))

        # training loop
        data_loader = self.data.get_loader(batch_size=batch_size, shuffle=True)

        for epoch in range(1, num_epochs+1):
            self.train()

            loss_epoch = 0

            for i, (data, target) in enumerate(data_loader):
                
                # Forward pass
                outputs = self(data)

                # Compute the loss and optimize
                self.optimizer.zero_grad()
                loss = self.loss_function(outputs, target)
                loss.backward()
                self.optimizer.step()

                # Store the loss of the batch
                self.results.update(
                    loss_batch = loss.item(),
                    time_batch = int(t.time() - start_time),
                )

                # print the loss of the batch
                if print_batch:
                    print(' - Epoch [{}/{}], i: [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, (i+1)*batch_size, len(self.data.x_train), loss.item()), end='\r')

                # add to the epoch loss
                loss_epoch += loss.item() 

            self.eval()
            # update results
            self.results.update(
                update_leader = 'loss_validation_epoch' if validation_pct else 'loss_epoch',
                n_epoch = epoch,
                loss_epoch = loss_epoch/len(data_loader),
                loss_validation_epoch = self.loss_test_train if validation_pct else None,
                parameters_epoch = deepcopy(self.state_dict()),
                time_epoch = t.time() - start_time,
            )

            # print the loss of "validation_print_n" strings of the validation data
            for _ in range(validation_print_n):

                i = np.random.randint(len(self.data.x_test_ptc))
                prediction, target = self(self.data.x_test_ptc[i]), self.data.y_test_ptc[i]

                print('\t Validation string, \t i: {}; \t prediction: {:.4f}, \t target: {:.4f}, \t loss: {:.4f}'.format(
                    i, prediction.item(), target.item(), self.loss_function(prediction, target).item()))

            #time
            estimated_remaining_time = (num_epochs - epoch) * (t.time() - start_time) / epoch #remaining epochs * time per epoch
            hours, remainder = divmod(estimated_remaining_time, 3600) # Convert remaining time to hours, minutes, and seconds for better readability
            minutes, seconds = divmod(remainder, 60)

            # Print the loss and remaining time for this epoch
            print('Epoch [{}/{}], Loss epoch: {:.4f}, Loss validation: {:.4f}, Time remaining: ~{}h {}m {:.0f}s'.format(
                epoch, num_epochs, self.results.loss_epoch.last, self.results.loss_validation_epoch.last, hours, minutes, seconds))
           
            # Stop training if the loss is not changing
            if f.should_stop_training(self.results.loss_epoch.history, **stop_training_options):
                print('The loss is not changing anymore, so we stop training.')
                break

    def save_model_info(self, description):
        dict_to_save_csv = {
            'model_uuid': str(self.uuid),
            'day': self.day,   
            'notebook': self.name_notebook,         
            'version': self.version,
            'data_uuid': str(self.data.uuid),
            'n_aminoacids': self.data.input_params['n_aminoacids'],           
            'description': description,
            'sweep_uuid': str(self.sweep_uuid) if hasattr(self, 'sweep_uuid') else None,
            'sweep_point': {k:str(v) for k,v in self.sweep_point.items()} if hasattr(self, 'sweep_point') else None,
            'metadata': self.metadata,
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        dict_to_save_json = {
            **dict_to_save_csv,  
            'model': str(self), 
            'training_inputs': self.training_inputs,
        }
        f.save_csv(
            dict_to_save_csv, 
            file_name='model_uuids',
            initial_path=self.initial_path
        )
        f.save_json(
            dict_to_save_json, 
            folder='Models',
            initial_path=self.initial_path,
            file_name=self.file_name,
            day=self.day,
        )

    def validate(self, pct=1, add_to_results=True, plot=False, print_items=False): 

        # random order for test data
        random_order = np.random.permutation(len(self.data.x_test))
        x_test = self.data.x_test[random_order]
        y_test = self.data.y_test[random_order]

        # data
        x_test = x_test[:int(len(x_test)*pct)]
        y_test = y_test[:int(len(y_test)*pct)]
        y_prediction = []
        losses = []

        #time 
        len_data = len(x_test)
        start_time = t.time()

        self.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(zip(x_test, y_test)):
                prediction = self(x)
                y_prediction.append(prediction.item())
                losses.append(self.loss_function(prediction, y).item())

                if print_items:
                    print("i: {}/{}, y: {:.4f}, prediction: {:.4f}, loss: {:.6f}. \t\t\t Ending in {:.2f} minutes".format(
                        i+1, len_data,
                        y.item(), prediction.item(), self.loss_function(prediction, y),
                        (len_data-i+1)*(t.time()-start_time)/(i+1)/60)
                    )
                else:
                    # print progress
                    print("Progress: {:.2f}%. \t\t\t Ending in {:.2f} minutes".format(
                        100*(i+1)/len_data, (len_data-i+1)*(t.time()-start_time)/(i+1)/60), end="\r")

        r_squared = f.r_squared(y_test.flatten().tolist(), y_prediction)
        if add_to_results:
            self.results.add_plain_attributes(
                validation = {
                    'x_test': x_test.tolist(),
                    'y_test': y_test.tolist(),
                    'y_prediction': y_prediction,
                    'losses': losses,
                    'r_squared': r_squared,
                }
            )
            
        if plot:
            self.plot_validation()

        print("Mean loss: {}, std loss: {}, r^2: {}".format(np.mean(losses), np.std(losses), r_squared))
        
    @property
    def mean_loss_validation(self):
        return np.mean(self.results.validation['losses'])
    
    @property
    def std_loss_validation(self):
        return np.std(self.results.validation['losses'])
    
    def save_results(self):
        self.results.save()

    def plot_validation(self, fig_size=(6,6)):
        f.plot_validation(results=self.results, fig_size=fig_size)

    def plot_losses(self, fig_size=(6,6)):
        f.plot_losses_training(results=self.results, fig_size=fig_size)

    def load_results(self, uuid):

        file_path = self.initial_path + '/saved/results.csv'
        returned_list = f.get_from_csv(file_path, to_search={'model_uuid': str(uuid)}, to_return=['day', 'file_name'])
        self.day = returned_list[0]
        self.name_notebook, self.version = returned_list[1].split('-')
        self.version = int(self.version)
        self.uuid = uuid

        if self.day is None:
            raise ValueError("Returned None for day.")
        if self.file_name is None:
            raise ValueError("Returned None for file_name.")
        
        self.results = c.Results.load(
            initial_path=self.initial_path,
            day=self.day,
            file_name=self.file_name,
        )
        print("Loaded results with uuid {}.".format(uuid))

    def load_state_dict_from_results(self, state_dict_name, rule):
        self.load_state_dict(getattr(getattr(self.results, state_dict_name), rule))
        print("Loaded {} state_dict from results.".format(rule)) 

    
    def load_optimized_inputs(self):        
        try:
            self.optimized_inputs = f.load_pickle(self.file_name+"-optimized_inputs_list", 'Input_optimization', self.initial_path, self.day)
        except:
            self.optimized_inputs = []

    def get_input_with_low_score(
        self,
        layers_to_use:list,
        n_iter = 1000,
        optimizer = torch.optim.Adam,
        optimizer_options = {'lr': 0.1},
        stop_criterion = 1e-4,
        initialization_seed = 42,
        metadata = {},
    ):  
        
        self.input_optimization_uuid = uuid.uuid4()
        if not hasattr(self, 'optimized_inputs'): self.load_optimized_inputs()        
        
        optimization_inputs = {
            'layers_to_use': layers_to_use,
            'n_iter': n_iter,
            'optimizer': optimizer,
            'optimizer_options': optimizer_options,
            'stop_criterion': stop_criterion,
            'initialization_seed': initialization_seed,
            'metadata': metadata,
        }
        
        # OPTIMIZE THE QUANTUM INPUT
        self.input_optimization_results = c.Results(
            model_uuid = self.uuid,
            file_name = self.file_name,
            day = self.day,
            initial_path = self.initial_path,
            metadata = {
                'input_optimization_uuid': self.input_optimization_uuid,
                'optimization_inputs': optimization_inputs,
            },
            score  = c.keep(last=True, best=True, history=True),
            optimized_input = c.keep(last=True, best=True, history=False),
            n_iter = c.keep(last=True, best=True, history=True),
            time   = c.keep(last=True, best=True, history=True),
        )

        torch.manual_seed(initialization_seed)
        input_to_optimize = nn.Parameter(torch.randint(0, 18, self.quantum_layer.input_shape, dtype=torch.float64) / 18 * 2 * np.pi)
        optimizer = optimizer([input_to_optimize], **optimizer_options)

        def model_without_embedding(x):
            for layer in layers_to_use:
                x = getattr(self, layer)(x)
            return x

        self.eval()
        last_score = model_without_embedding(input_to_optimize).item()        
        self.input_optimization_results.update(
            update_leader = 'score',
            optimized_input = input_to_optimize.detach().numpy().tolist(),
            n_iter = 0,
            score = last_score,
            time = 0,
        )
        start_time = t.time()

        for i in range(n_iter):

            # optimize
            optimizer.zero_grad()
            score = model_without_embedding(input_to_optimize)
            score.backward()
            optimizer.step()

            # update results
            self.input_optimization_results.update(
                update_leader = 'score',
                optimized_input = input_to_optimize.detach().numpy().tolist(),
                n_iter = i+1,
                score = score.item(),
                time = t.time() - start_time,
            )

            # print the score and stop if the score is not changing anymore
            print(i, score.item(), end='\r')
            if abs(last_score - score.item()) < stop_criterion and i > 10:
                break

            # update last_score
            last_score = score.item()


        # FROM OPTIMIZED QUANTUM INPUT TO AMINOACID SEQUENCE
        def to_unique_angles(x):
            return x % (2 * np.pi)
        
        # get the optimized input between 0 and 2pi
        optimized_x = to_unique_angles(np.array(self.input_optimization_results.optimized_input.best))

        # get the embedding values between 0 and 2pi
        values_embedding = to_unique_angles(self.state_dict()['fc1.weight'].flatten().detach().numpy())
        dict_values = dict(zip(values_embedding, list(range(19))))
        values_embedding = sorted(values_embedding)

        # get between which embedding values the optimized input values are
        intervals = f.find_intervals(values_embedding, optimized_x) 
        # for each position we have two possible aminoacids
        possible_inputs = [(dict_values[i],dict_values[j]) for i,j in intervals] 
        # get all possible combinations
        combinations = list(product(*possible_inputs))
        

        best_combination_i = 0
        scores = []
        for i, comb in enumerate(combinations):
            score = self(torch.tensor(comb, dtype=torch.int)).item()
            scores.append(score)
            if score < scores[best_combination_i]:
                best_combination_i = i

            print('{}/{} combination, score: {:.4f}, best score: {:.4f}'.format(i+1, len(combinations), score, scores[best_combination_i]), end='\r')
        print('{}, score: {:.4f}'.format(combinations[best_combination_i], scores[best_combination_i]), end='                    ')


        self.optimized_inputs.append((combinations[best_combination_i], scores[best_combination_i]))
        self.input_optimization_results.add_plain_attributes(
            optimized_combinations = {
                'combinations': combinations,
                'scores': scores,
                'best_combination_i': best_combination_i,
            }
        )

    def get_input_with_low_score_2(
        self,
        layers_to_use:list,
        n_iter = 1000,
        optimizer = torch.optim.Adam,
        optimizer_options = {'lr': 0.1},
        stop_criterion = 1e-4,
        initialization_seed = 42,
        metadata = {},
        embedding_dim = 2,
    ):  
        
        self.input_optimization_uuid = uuid.uuid4()
        if not hasattr(self, 'optimized_inputs'): self.load_optimized_inputs()        
        
        optimization_inputs = {
            'layers_to_use': layers_to_use,
            'n_iter': n_iter,
            'optimizer': optimizer,
            'optimizer_options': optimizer_options,
            'stop_criterion': stop_criterion,
            'initialization_seed': initialization_seed,
            'metadata': metadata,
        }
        
        # OPTIMIZE THE QUANTUM INPUT
        self.input_optimization_results = c.Results(
            model_uuid = self.uuid,
            file_name = self.file_name,
            day = self.day,
            initial_path = self.initial_path,
            metadata = {
                'input_optimization_uuid': self.input_optimization_uuid,
                'optimization_inputs': optimization_inputs,
            },
            score  = c.keep(last=True, best=True, history=True),
            optimized_input = c.keep(last=True, best=True, history=False),
            n_iter = c.keep(last=True, best=True, history=True),
            time   = c.keep(last=True, best=True, history=True),
        )

        torch.manual_seed(initialization_seed)
        input_to_optimize = nn.Parameter(torch.randint(0, 18, self.quantum_layer.input_shape, dtype=torch.float64) / 18 * 2 * np.pi)
        optimizer = optimizer([input_to_optimize], **optimizer_options)

        def model_without_embedding(x):
            for layer in layers_to_use:
                x = getattr(self, layer)(x)
            return x

        self.eval()
        last_score = model_without_embedding(input_to_optimize).item()        
        self.input_optimization_results.update(
            update_leader = 'score',
            optimized_input = input_to_optimize.detach().numpy().tolist(),
            n_iter = 0,
            score = last_score,
            time = 0,
        )
        start_time = t.time()

        for i in range(n_iter):

            # optimize
            optimizer.zero_grad()
            score = model_without_embedding(input_to_optimize)
            score.backward()
            optimizer.step()

            # update results
            self.input_optimization_results.update(
                update_leader = 'score',
                optimized_input = input_to_optimize.detach().numpy().tolist(),
                n_iter = i+1,
                score = score.item(),
                time = t.time() - start_time,
            )

            # print the score and stop if the score is not changing anymore
            print(i, score.item(), end='\r')
            if abs(last_score - score.item()) < stop_criterion and i > 10:
                break

            # update last_score
            last_score = score.item()


        # FROM OPTIMIZED QUANTUM INPUT TO AMINOACID SEQUENCE
        def to_unique_angles(x):
            return x % (2 * np.pi)
        
        # get the optimized input between 0 and 2pi
        optimized_x = to_unique_angles(np.array(self.input_optimization_results.optimized_input.best))
        embedding_dim = self.fc1.embedding_dim
        optimized_x = torch.tensor(optimized_x).reshape(embedding_dim, self.quantum_layer.input_shape[0]//embedding_dim).T

        # get the embedding values between 0 and 2pi
        values_embedding = to_unique_angles(self.fc1.weight.data)

        seq = []
        for i in optimized_x:
            distance = torch.norm(values_embedding - i, dim=1)
            nearest = torch.argmin(distance)
            seq.append(nearest.item())
        score = self(torch.tensor(seq, dtype=torch.int)).item()
        seq = tuple(seq)

        print('Score: {:.4f}, sequence: {}'.format(score, seq), end='                    ')


        self.optimized_inputs.append((seq, score))
        self.input_optimization_results.add_plain_attributes(
            optimized_input = {
                'sequence': seq,
                'score': score,
            }
        )

    def save_input_optimization_results(self):
        self.input_optimization_results.save(
            csv_file='input_optimization',
            folder='Input_optimization',
            additional_csv={
                'optimization_uuid': self.input_optimization_uuid,
                'optimization_version': len(self.optimized_inputs)-1,
            },
            file_name_prefix='-'+str(len(self.optimized_inputs)-1),
        )
        f.save_pickle(self.optimized_inputs, self.file_name+"-optimized_inputs_list", 'Input_optimization', self.initial_path, self.day)

    def load_input_optimization_results(self, i=None):
        self.load_optimized_inputs()
        file_name = self.file_name+'-'+str(len(self.optimized_inputs)-1) if i is None else self.file_name+'-'+str(i)
        self.input_optimization_results = f.load_pickle(file_name, 'Input_optimization', self.initial_path, day=self.day)
        self.input_optimization_uuid = self.input_optimization_results.metadata['input_optimization_uuid']
        print("Loaded input optimization results with uuid {}.".format(self.input_optimization_uuid))

    @property
    def optimized_inputs_unique(self):
        return list(set(self.optimized_inputs))
    
    @property
    def best_optimized_input(self):
        return min(self.optimized_inputs_unique, key=lambda x: x[1])
        





    # def plot_parameter(self, layer, index=None, save=False):

        # parameter_evolution = []

        # if self.parameters is None:
        #     print("No parameters saved, so no plot can be made. Please set keep_track_params=True when initializing the model or do model.keep_track_params = True.")
        # else:
        #     if index is None:
        #         for i in range(len(self.parameters)):
        #             parameter_evolution.append(torch.mean(self.parameters[i][layer]).item())
        #     else:
        #         for i in range(len(self.parameters)):
        #             parameter_evolution.append(self.parameters[i][layer][index].item())

        #     plt.figure()
        #     plt.plot(parameter_evolution)
        #     plt.xlabel('Batch')
        #     plt.ylabel('Parameter value')
        #     plt.title('Parameter ({}, {})'.format(layer, index))

        #     if save:
        #         file = f.get_name_file_to_save(self.name_notebook, self.initial_path, extension="png", version=self.version, postfix= "_parameter_{}_{}".format(layer, index))
        #         plt.savefig(file)
        #         print("Saved in: ", file)

        #     plt.show()

