
# imports
import os
import uuid
import datetime
import time as t
import numpy as np
from copy import deepcopy

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
    



def computed_required(func):
    def wrapper(self, *args, **kwargs):
        if not self.computed:
            raise ValueError("You need to compute the validation first. Use validation.compute(pct=pct) to compute the validation for the test dataset.")
        return func(self, *args, **kwargs)
    return wrapper

class Model(nn.Module):

    def set_name_and_path(self, name_notebook, initial_path):
        self.name_notebook = name_notebook
        self.initial_path = initial_path

    # def set_data(
    #     self, 
    #     x, 
    #     y, 
    #     split_options = dict(
    #         test_size= 0.2, 
    #         random_state = 42
    #     ),
    #     tensors_type_x = torch.float64,
    #     tensors_type_y = torch.float64,
    # ):
        
    #     self.data = c.data(x, y, split_options, tensors_type_x, tensors_type_y)

    def initialize_params(self, initialization_options= [{}]):

        self.initialization_options = initialization_options

        for io in initialization_options:
            getattr(nn.init, io['type'])(getattr(self[io['layer']], io['name']), **io['options'])

    @property
    def loss_train(self):
        return self.loss_function(self(self.data.x_train), self.data.y_train).item()
    @property
    def loss_test_train(self):
        return self.loss_function(self(self.data.x_test_train), self.data.x_test_train).item()
    @property
    def loss_test(self):
        return self.loss_function(self(self.data.x_test), self.data.x_test).item()

    def optimize_params(
        self,
        # loss function and optimizer
        loss_function = nn.MSELoss,
        optimizer = optim.SGD,
        optimizer_options = {'lr': 0.05},
        # data
        data = None,
        # training loop
        num_epochs = 25,
        batch_size = 32,
        print_batch = True,
        keep_track_params = False,
        # validation
        validation_pct = 0.2,
        validation_print_n = 3,
        # Stop training
        stop_training_options = {},
    ):
        
        #check if data is given and if isinstance of c.data
        if data is None:
            raise ValueError("You need to give a data object.")
        if not isinstance(data, c.data):
            raise ValueError("The data object needs to be a data object.")
        self.data = data

        self.training_inputs = {
            'loss_function': loss_function,
            'optimizer': optimizer,
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
        self.version, self.uuid, self.day = f.update_version(self.name_notebook, self.initial_path), uuid.uuid4(), datetime.datetime.now().strftime("%m%d")

        # time
        start_time = t.time()

        # loss function and optimizer
        self.loss_function = loss_function()
        self.optimizer = optimizer(self.model.parameters(), **optimizer_options)

        #data
        self.data.set_test_train(validation_pct)

        # keep track of results
        self.results = c.Results(
            model_uuid = self.uuid,
            loss_batch  = c.keep(last=False, best=False, history=True),
            time_batch  = c.keep(last=False, best=False, history=True),
            n_epoch     = c.keep(last=True, best=True, history=True),
            loss_epoch  = c.keep(last=True, best=True, history=True),
            loss_validation_epoch = c.keep(last=True, best=True, history=True),
            parameters_epoch  = c.keep(last=True, best=True, history=keep_track_params),
            time_epoch        = c.keep(last=True, best=True, history=True),
        )
        self.results.update(
            update_leader = 'loss_validation_epoch' if validation_pct else 'loss_epoch',
            n_epoch = 0,
            loss_epoch = self.loss_train,
            loss_validation_epoch = self.loss_test_train if validation_pct else None,
            parameters = deepcopy(self.state_dict()) if keep_track_params else None,
            time = t.time() - start_time,
        )
        print('Epoch [{}/{}], Loss epoch: {:.4f}, Loss validation: {:.4f}'.format(0, num_epochs, self.results.loss_epoch.last, self.results.loss_epoch_validation.last))

        # training loop
        data_loader = self.data.get_loader(batch_size=batch_size, shuffle=True)

        for epoch in range(1, num_epochs+1):

            loss_epoch = 0

            for i, (data, target) in enumerate(data_loader):
                
                # Forward pass
                outputs = self(data)

                # Compute the loss and optimize
                optimizer.zero_grad()
                loss = self.loss_function(outputs, target)
                loss.backward()
                optimizer.step()

                # Store the loss of the batch
                self.results.update(
                    loss_batch = loss.item(),
                    time_batch = t.time() - start_time,
                )

                # print the loss of the batch
                if print_batch:
                    print(' - Epoch [{}/{}], i: [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, (i+1)*batch_size, len(self.data.x_train), loss.item()), end='\r')

                # add to the epoch loss
                loss_epoch += loss.item() 

            # update results
            self.results.update(
                update_leader = 'loss_validation_epoch' if validation_pct else 'loss_epoch',
                n_epoch = epoch,
                loss_epoch = loss_epoch/len(data_loader),
                loss_validation_epoch = self.loss_test_train if validation_pct else None,
                parameters = deepcopy(self.state_dict()) if keep_track_params else None,
                time = t.time() - start_time,
            )

            # print the loss of "validation_print_n" strings of the validation data
            for _ in range(validation_print_n):

                i = np.random.randint(len(self.data.x_test_train))
                prediction, target = self.model(self.data.set_test_train[i]), self.data.y_test_train[i]

                print('\t Validation string, \t i: {}; \t prediction: {:.4f}, \t target: {:.4f}, \t loss: {:.4f}'.format(
                    i, prediction.item(), target.item(), self.loss_function(prediction, target).item()))

            #time
            estimated_remaining_time = (num_epochs - epoch) * (t.time() - start_time) / epoch #remaining epochs * time per epoch
            hours, remainder = divmod(estimated_remaining_time, 3600) # Convert remaining time to hours, minutes, and seconds for better readability
            minutes, seconds = divmod(remainder, 60)

            # Print the loss and remaining time for this epoch
            print('Epoch [{}/{}], Loss epoch: {:.4f}, Loss validation: {:.4f}, Time remaining: ~{}h {}m {:.0f}s'.format(
                epoch, num_epochs, self.results.loss_epoch.last, self.results.loss_epoch_validation.last, hours, minutes, seconds))
           
            # Stop training if the loss is not changing
            if f.should_stop_training(self.losses_epochs, **stop_training_options):
                print('The loss is not changing anymore, so we stop training.')
                break

    def save_model_info(self, description, metadata={}):
        dict_to_save_csv = {
            'uuid': str(self.uuid),
            'day': self.day,            
            'version': self.version,
            'data_uuid': str(self.data.uuid),
            'n_aminoacids': self.data.input_params['n_aminoacids'],
            'name_notebook': self.name_notebook,
            'description': description,
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        dict_to_save_json = {
            'model_uuid': str(self.uuid),
            'name_notebook': self.name_notebook,
            'initial_path': self.initial_path,
            'model': str(self),
            'version': self.version,
            'training_inputs': self.training_inputs,
            'metadata': metadata,
        }


    def validate(self, pct=1, plot=True, save=True, metadata={}):
        self.validation = c.validation(self, model_uuid=self.uuid)
        self.validation.compute(pct=pct)
        if plot: self.validation.plot()
        if save: 
            self.save_model_info()
            self.validation.dump(
                path=f.get_name_file_to_save(self.name_notebook, self.initial_path, extension="json", version=self.version, postfix="_validation"),
                metadata=metadata
            ) 
        return self.validation.mean_loss
    
    def save_results(self, metadata = {}):
        self.save_model_info()
        self.results.dump(
            f.get_name_file_to_save(self.name_notebook, self.initial_path, extension="json", version=self.version, postfix="_results"), 
            metadata=metadata
        )


    def plot_losses(self):

        for x, y, title in [(None, 'loss_batch', 'Loss per batch'), ('n_epoch', 'loss_epoch', 'Loss per epoch'), ('n_epoch', 'loss_validation_epoch','Loss per epoch (validation)')]:
            self.results.plot(x=x, y=y, title=title, xlabel=x, ylabel=y)


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









# class pytorch_model:

#     def __init__(   
#         self, 
#         circuit_layers, 
#         save_options,
#         keep_track_params = False,
#     ):

#         # model
#         self.model = nn.Sequential(*circuit_layers)

#         # save options
#         self.name_notebook = save_options['name_notebook']
#         self.initial_path = save_options['initial_path']
#         self.version = None

#         # keep track of the losses (and parameters)
#         self.losses_batches = None
#         self.losses_epochs = None
#         self.losses_epochs_validation = None

#         self.keep_track_params = keep_track_params
#         self.parameters = None
#         self.training_inputs = None


#     def __call__(self, *args, **kwds):
#         return self.model(*args, **kwds)

#     def __repr__(self):
#         return self.model.__repr__()

#     def set_data(self, data_X, data_Y, data_X_validation, data_Y_validation):
#         self.data_X = data_X
#         self.data_Y = data_Y
#         self.data_X_validation = data_X_validation
#         self.data_Y_validation = data_Y_validation

#     def train(  self, 
#                 # loss function and optimizer
#                 loss_function = ScaledSquaredRelativeMSELoss,  
#                 optimizer = optim.SGD,
#                 optimizer_options = {'lr': 0.05},
#                 # initial parameters
#                 initialization_options = [{}],
#                 # training loop
#                 num_epochs = 25,
#                 batch_size = 32,
#                 print_batch = True,
#                 # validation
#                 validation = True,
#                 n_validation = 10,
#                 n_print_validation = 3,
#                 # Time
#                 time = True,
#                 # Stop training
#                 stop_training_options = {},
#     ):
        
#         self.training_inputs = {
#             'loss_function': loss_function,
#             'optimizer': optimizer,
#             'optimizer_options': optimizer_options,
#             'initialization_options': initialization_options,
#             'num_epochs': num_epochs,
#             'batch_size': batch_size,
#             'print_batch': print_batch,
#             'validation': validation,
#             'n_validation': n_validation,
#             'n_print_validation': n_print_validation,
#             'time': time,
#             'stop_training_options': stop_training_options,
#         }
#         self.version = f.update_version(self.name_notebook, self.initial_path)

#         # initialization
#         for io in initialization_options:
#             if io: self._initialize(**io) # if the dict 'io' is not empty, we initialize the parameters chosen with the options in 'io'
        
#         # data
#         input_data = self.data_X
#         target_data = self.data_Y
#         input_data_validation = self.data_X_validation
#         target_data_validation = self.data_Y_validation
                
#         mean_squared_targets = torch.mean(target_data**2)

#         # time
#         if time:
#             start_time = t.time()

#         # loss function and optimizer
#         self.loss_function = loss_function()
#         self.optimizer = optimizer(self.model.parameters(), **optimizer_options)
#         loss_function = self.loss_function
#         optimizer = self.optimizer

#         # keep track of the losses (and parameters)
#         self.losses_batches = []
#         self.losses_epochs = [loss_function(self.model(input_data), target_data, mean_squared_targets).item()]
#         if self.keep_track_params:  self.parameters = [deepcopy(self.model.state_dict())]

#         # validation
#         if validation and (input_data_validation is None or target_data_validation is None): #if no validation data is given, we don't do validation
#             validation = False
#             print('No validation data given, so no validation will be done.')
#         if validation:
#             #validation data
#             i_validation = input_data_validation[::n_validation] #we take only every n_validation-th data point
#             t_validation = target_data_validation[::n_validation] 

#             # keep track of the losses
#             self.losses_epochs_validation = [loss_function(self.model(i_validation), t_validation, mean_squared_targets).item()]

#             # print the loss before training
#             print('Epoch [{}/{}], Loss: {:.4f}, Loss validation: {:.4f}'.format('0', num_epochs, self.losses_epochs[-1], self.losses_epochs_validation[-1]))            
#         else:
#             # print the loss before training
#             print('Epoch [{}/{}], Loss: {:.4f}'.format('0', num_epochs, self.losses_epochs[-1]))

#         # training loop
#         for epoch in range(num_epochs):

#             # Shuffle the dataset
#             indices = torch.randperm(input_data.size(0))
#             input_data = input_data[indices]
#             target_data = target_data[indices]

#             # add a new entry to the epoch losses
#             self.losses_epochs.append(0)

#             # batch training
#             for i in range(0, input_data.size(0), batch_size):

#                 inputs = input_data[i:i+batch_size]
#                 targets = target_data[i:i+batch_size]

#                 # Forward pass
#                 outputs = self.model(inputs)

#                 # Compute the loss
#                 loss = loss_function(outputs, targets, mean_squared_targets)

#                 # Backward pass and optimization
#                 optimizer.zero_grad()
#                 loss.backward()
#                 optimizer.step()

#                 # Store the loss
#                 self.losses_batches.append(loss.item())

#                 # print the loss of the batch
#                 if print_batch:
#                     print('- Epoch [{}/{}], i: [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i, input_data.size(0), loss.item()), end='\r')

#                 # add to the epoch loss
#                 self.losses_epochs[-1] += loss.item() 

#                 # keep track of the parameters
#                 if self.keep_track_params:  self.parameters.append(deepcopy(self.model.state_dict()))

#             # divide the epoch loss by the number of batches, to get the average loss
#             self.losses_epochs[-1] /= (input_data.size(0)/batch_size)

#             # Validation
#             if validation:
#                 self.losses_epochs_validation.append(loss_function(self.model(i_validation), t_validation, mean_squared_targets).item()) 

#             # print the loss of "n_print_validation" strings of the validation data
#             if validation:
#                 n_print_validation = min(n_print_validation, len(i_validation)) #if n_print_validation is bigger than the number of validation data points, we print all of them
#                 for i in range(n_print_validation):
#                     prediction = self.model(i_validation[i])
#                     target = t_validation[i]
#                     print('\t Validation string, \t i: {}; \t prediction: {:.4f}, \t target: {:.4f}, \t loss: {:.4f}'.format(i, prediction.item(), target.item(), loss_function(prediction, target, mean_squared_targets).item()))


#             if time:
#                 # time
#                 # Compute elapsed time and remaining time
#                 elapsed_time = t.time() - start_time
#                 avg_time_per_epoch = elapsed_time / (epoch + 1)
#                 remaining_epochs = num_epochs - (epoch + 1)
#                 estimated_remaining_time = avg_time_per_epoch * remaining_epochs

#                 # Convert remaining time to hours, minutes, and seconds for better readability
#                 hours, remainder = divmod(estimated_remaining_time, 3600)
#                 minutes, seconds = divmod(remainder, 60)

#                 # Print the loss and remaining time for this epoch
#                 print('Epoch [{}/{}], Loss: {:.4f}, Loss validation: {:.4f}, Time remaining: ~{}h {}m {:.0f}s'.format(
#                     epoch+1, num_epochs, self.losses_epochs[-1], self.losses_epochs_validation[-1], hours, minutes, seconds))
#             else:
#                 # Print the loss for this epoch
#                 print('Epoch [{}/{}], Loss: {:.4f}, Loss validation: {:.4f}'.format(epoch+1, num_epochs, self.losses_epochs[-1], self.losses_epochs_validation[-1]))


#             # Stop training if the loss is not changing
#             if self.should_stop_training(self.losses_epochs, **stop_training_options):
#                 print('The loss is not changing anymore, so we stop training.')
#                 break

#     def should_stop_training(self, losses, lookback_epochs=5, threshold_slope=0.001, threshold_std_dev=0.1):
#         """
#         Determines if training should stop based on the standard deviation of the last `lookback_epochs` losses.
        
#         Parameters:
#         - losses (list): List of loss values, where the most recent loss is the last element.
#         - lookback_epochs (int): Number of recent epochs to consider.
#         - threshold_slope (float): Maximum slope of the linear regression of the losses.
#         - threshold_std_dev (float): Maximum standard deviation of the losses.
        
#         Returns:
#         - bool: True if training should stop, False otherwise.
#         """
        
#         # If there aren't enough epochs yet, continue training
#         if len(losses) < lookback_epochs:
#             return False
        
#         # Extract the last `lookback_epochs` losses
#         recent_losses = losses[-lookback_epochs:]
        
#         # Calculate the standard deviation of the recent losses
#         std_dev = sum([(x - sum(recent_losses) / lookback_epochs) ** 2 for x in recent_losses]) ** 0.5 / lookback_epochs

#         # If the standard deviation is above the threshold, continue training
#         if std_dev > threshold_std_dev:
#             return False
        
#         # Calculate the slope of the linear regression of the recent losses
#         slope = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]

#         # If the negative slope is above the threshold, continue training
#         if -slope > threshold_slope:
#             return False

#         # Otherwise, stop training
#         return True
    
#     def _initialize(self, type, layer, name, options):
#         getattr(nn.init, type)(getattr(self.model[layer], name), **options)

#     def plot_parameter(self, layer, index=None, save=False):

#         parameter_evolution = []

#         if self.parameters is None:
#             print("No parameters saved, so no plot can be made. Please set keep_track_params=True when initializing the model or do model.keep_track_params = True.")
#         else:
#             if index is None:
#                 for i in range(len(self.parameters)):
#                     parameter_evolution.append(torch.mean(self.parameters[i][layer]).item())
#             else:
#                 for i in range(len(self.parameters)):
#                     parameter_evolution.append(self.parameters[i][layer][index].item())

#             plt.figure()
#             plt.plot(parameter_evolution)
#             plt.xlabel('Batch')
#             plt.ylabel('Parameter value')
#             plt.title('Parameter ({}, {})'.format(layer, index))

#             if save:
#                 file = f.get_name_file_to_save(self.name_notebook, self.initial_path, extension="png", version=self.version, postfix= "_parameter_{}_{}".format(layer, index))
#                 plt.savefig(file)
#                 print("Saved in: ", file)

#             plt.show()

#     def plot_losses(self, batches=True, epochs=True, epochs_validation=True, save=False, save_txt=False):
        
#         if batches:
#             plt.figure()
#             plt.plot(self.losses_batches)
#             plt.title('Loss per batch')
#             if save: 
#                 file = f.get_name_file_to_save(self.name_notebook, self.initial_path, extension="png", version=self.version, postfix="_losses_batches")
#                 plt.savefig(file)
#                 print("Saved in: ", file)
#             plt.show()

#         if epochs:
#             plt.figure()
#             plt.plot(self.losses_epochs)
#             plt.title('Loss per epoch')
#             if save: 
#                 file = f.get_name_file_to_save(self.name_notebook, self.initial_path, extension="png", version=self.version, postfix="_losses_epochs")
#                 plt.savefig(file)
#                 print("Saved in: ", file)
#             plt.show()

#         if epochs_validation:
#             plt.figure()
#             plt.plot(self.losses_epochs_validation)
#             plt.title('Loss per epoch (validation)')
#             if save: 
#                 file = f.get_name_file_to_save(self.name_notebook, self.initial_path, extension="png", version=self.version, postfix="_losses_epochs_validation")
#                 plt.savefig(file)
#                 print("Saved in: ", file)
#             plt.show()

#         #save data
#         if save_txt:
#             self.save_losses(batches=batches, epochs=epochs, epochs_validation=epochs_validation)
    
#     def save_losses(self, batches=True, epochs=True, epochs_validation=True):        
#         for b, name in zip([batches, epochs, epochs_validation], ["losses_batches", "losses_epochs", "losses_epochs_validation"]):
#             if b:
#                 file_name = f.get_name_file_to_save(self.name_notebook, self.initial_path, extension="txt", version=self.version, postfix="_"+name)
#                 with open(file_name, 'w') as file:
#                     for loss in getattr(self, name, []):
#                         file.write(str(loss)+'\n')
#                 print("Saved in: ", file_name)


#     def _compute_validation(self, percentatge=1):

#         # if we have a new version or a new percentatge compute the validation
#         if getattr(self, 'last_validation_computed', 0) != self.version or getattr(self, 'last_validation_percentatge', 0) != percentatge:

#             # save the version and the percentatge
#             self.last_validation_computed = self.version
#             self.last_validation_percentatge = percentatge

#             # data cut with percentatge
#             if percentatge < 1:
#                 data_X_validation = self.data_X_validation[:int(len(self.data_X_validation)*percentatge)]
#                 data_Y_validation = self.data_Y_validation[:int(len(self.data_Y_validation)*percentatge)]
#             else:
#                 data_X_validation = self.data_X_validation
#                 data_Y_validation = self.data_Y_validation

#             # varaibles
#             self.avg_loss_validation = 0
#             self.targets_validation = []
#             self.predictions_validation = []
#             self.losses_validation = []
            
#             # variables for printing progress
#             len_data = len(data_X_validation)
#             start_time = t.time()

#             # compute validation and save
#             mean_squared_targets = torch.mean(self.data_Y**2)
#             for i, (x, y) in enumerate(zip(data_X_validation, data_Y_validation)):

#                 # print progress
#                 print("Progress: {:.2f}%. \t\t\t Ending in {:.2f} minutes".format(
#                     100*(i+1)/len_data, (len_data-i+1)*(t.time()-start_time)/(i+1)/60), end="\r")

#                 # compute
#                 prediction = self.model(x)
#                 loss = self.loss_function(prediction, y, mean_squared_targets)

#                 # save
#                 self.avg_loss_validation += loss/len_data
#                 self.targets_validation.append(y.item())
#                 self.predictions_validation.append(prediction.item())
#                 self.losses_validation.append(loss.item())

#         return np.array(self.targets_validation), np.array(self.predictions_validation), np.array(self.losses_validation), self.avg_loss_validation

#     def plot_validation(self, save=False, percentatge=1, fig_size=(6, 6)):

#         y_test, y_pred, losses, avg_loss = self._compute_validation(percentatge=percentatge)

#         plt.figure(figsize=fig_size)
#         plt.scatter(y_test, y_pred, color='r', label='Actual vs. Predicted', alpha=0.1)
#         plt.plot([np.min(y_test), np.max(y_test)], [np.min(y_test), np.max(y_test)], 'k--', lw=2, label='1:1 Line')
#         plt.xlabel('True Values')
#         plt.ylabel('Predictions')
#         plt.title('Predictions vs. True Values (average loss: {:.4f})'.format(avg_loss))
#         plt.legend()

#         if save: 
#             file = f.get_name_file_to_save(self.name_notebook, self.initial_path, extension="png", version=self.version, postfix="_validation")
#             plt.savefig(file)
#             print("Saved in: ", file)

#         plt.show()

#     def str_validation(self, save=True, precision=3, percentatge=1, printing=False):

#         y_test, y_pred, losses, avg_loss = self._compute_validation(percentatge=percentatge)

#         # varaibles
#         output_lines = []
#         format_string = 'i: {}, \t\t target: {:.%df}, \t prediction: {:.%df}, \t loss: {:.%df}' % (precision, precision, precision)

#         # function to save and print
#         def output(string):
#             output_lines.append(string)
#             if printing: print(string)

#         # print and save in variables
#         for x, (t, p, l) in enumerate(zip((y_test), y_pred, losses)):
#             output(format_string.format(x, t, p, l))

#         output('Average loss: {:.{}f}'.format(avg_loss, precision))

#         # save
#         if save:
#             save_filename = f.get_name_file_to_save(self.name_notebook, self.initial_path, extension="txt", version=self.version, postfix="_validation")
#             with open(save_filename, 'w') as file:
#                 file.write("Validation Results:\n")
#                 file.write("\n".join(output_lines))
#                 print("Saved in: ", save_filename)

#     def save_str(self, metadata={}):

#         filename = f.get_name_file_to_save(self.name_notebook, self.initial_path, extension="txt", version=self.version, postfix="_model_str")

#         # we save the string of the model
#         with open(filename, 'w') as file:

#             # metadata
#             file.write("metadata:\n")
#             for k, v in metadata.items():
#                 file.write("\t{}: {}\n".format(k, v))

#             # model layers
#             file.write("\n\n\nModel (notebook {}, version {}):\n".format(self.name_notebook, self.version))

#             # model training inputs
#             file.write("\nTraining inputs:\n")
#             for k, v in self.training_inputs.items():
#                 file.write("\t{}: {}\n".format(k, v))

#         print("Saved in: ", filename)

#     def save_state_dict(self, intermediate=False):

#         output_filename = f.get_name_file_to_save(self.name_notebook, self.initial_path, extension="pth", version=self.version)

#         torch.save(self.model.state_dict(), output_filename)

#         if intermediate and self.parameters is not None:
#             # create a folder with the name of the notebook and save all the self.parameters
#             folder = output_filename[:-4]
#             try: #try to create it (if it already exists, it will fail)
#                 os.mkdir(folder)
#             except: #if it fails, it means that the folder already exists
#                 #remove everything inside the folder
#                 for file in os.listdir(folder):
#                     os.remove(folder+'/'+file)

#             def numbers_with_zeros(i, x):
#                 num_digits = len(str(x))
#                 format_string = "{:0" + str(num_digits) + "d}"

#                 return format_string.format(i)
            
#             #save all the parameters
#             n_params = len(self.parameters)
#             for i, state_dict in enumerate(self.parameters):
                
#                 torch.save(state_dict, folder+'/state_dict_'+numbers_with_zeros(i, n_params)+'.pth')

#             print("Model saved as {} and intermediate parameters saved in {}".format(output_filename, folder))

#         else:
#             print("Model saved as {}".format(output_filename))


#     def load_state_dict(self, version=None, initial_path=None, name_notebook=None):

#         version       = self.version       if version is None       else version
#         initial_path  = self.initial_path  if initial_path is None  else initial_path
#         name_notebook = self.name_notebook if name_notebook is None else name_notebook

#         input_filename = initial_path + "checkpoints/"+ name_notebook[:4] +"/models/" + name_notebook[:-6] + "_" + str(version) + ".pth"

#         #check if the input file exists
#         if not os.path.exists(input_filename):
#             print("The file {} does not exist".format(input_filename))
#             return

#         self.model.load_state_dict(torch.load(input_filename))

#         self.version = version

#         print("Model loaded from {}".format(input_filename))

#     def dump(self, filename=None):
#         #dump the object in a pickle file without the model attribute
#         if filename is None:
#             filename = f.get_name_file_to_save(self.name_notebook, self.initial_path, extension="pickle", version=self.version)
#         with open(filename, "wb") as file:
#             x, self.model = self.model, str(self.model)
#             pickle.dump(self, file)
#             self.model = x

#     @classmethod
#     def load(cls, filename=None, name_notebook=None, version=None, initial_path=None):
#         if filename is None:
#             filename = initial_path + "checkpoints/"+ name_notebook[:4] +"/pickled_objects/" + name_notebook + "_" + str(version) + ".pickle"
#         with open(filename, "rb") as file:
#             loaded_obj = pickle.load(file)
#         return loaded_obj


    
