import torch
import torch.nn as nn
import torch.optim as optim
import time as t
import matplotlib.pyplot as plt
import os
from copy import deepcopy

from . import functions as f

torch.set_default_dtype(torch.float64)

class RelativeMSELoss(nn.Module):
    def __init__(self):
        super(RelativeMSELoss, self).__init__()

    def forward(self, predictions, targets):
        mse_loss = nn.MSELoss()(predictions, targets)
        relative_mse_loss = mse_loss / torch.mean(targets**2)
        rmse_loss = torch.sqrt(relative_mse_loss)
        return rmse_loss

class pytorch_model:

    def __init__(   self, 
                    circuit_layers, 
                    save_options,
                    keep_track_params = False,
    ):

        # model
        self.model = nn.Sequential(*circuit_layers)

        # save options
        self.name_notebook = save_options['name_notebook']
        self.initial_path = save_options['initial_path']
        self.version = None

        # keep track of the losses (and parameters)
        self.losses_batches = None
        self.losses_epochs = None
        self.losses_epochs_validation = None

        self.keep_track_params = keep_track_params
        self.parameters = None


    def __call__(self, *args, **kwds):
        return self.model(*args, **kwds)

    def __repr__(self):
        return self.model.__repr__()

    def set_data(self, data_X, data_Y, data_X_validation, data_Y_validation):
        self.data_X = data_X
        self.data_Y = data_Y
        self.data_X_validation = data_X_validation
        self.data_Y_validation = data_Y_validation

    def train(  self, 
                # loss function and optimizer
                loss_function = RelativeMSELoss,  
                optimizer = optim.SGD,
                optimizer_options = {'lr': 0.05},
                # Training loop
                num_epochs = 25,
                batch_size = 32,
                print_batch = True,
                # validation
                validation = True,
                n_validation = 10,
                n_print_validation = 3,
                # Time
                time=True
    ):
        
        self.version = f.update_version(self.name_notebook, self.initial_path)
        
        # data
        input_data = self.data_X
        target_data = self.data_Y
        input_data_validation = self.data_X_validation
        target_data_validation = self.data_Y_validation

        # time
        if time:
            start_time = t.time()

        # loss function and optimizer
        self.loss_function = loss_function()
        self.optimizer = optimizer(self.model.parameters(), **optimizer_options)
        loss_function = self.loss_function
        optimizer = self.optimizer

        # keep track of the losses (and parameters)
        self.losses_batches = []
        self.losses_epochs = [loss_function(self.model(input_data), target_data).item()]
        if self.keep_track_params:  self.parameters = [deepcopy(dict(self.model.state_dict().items()))]

        # validation
        if validation and (input_data_validation is None or target_data_validation is None): #if no validation data is given, we don't do validation
            validation = False
            print('No validation data given, so no validation will be done.')
        if validation:
            #validation data
            i_validation = input_data_validation[::n_validation] #we take only every n_validation-th data point
            t_validation = target_data_validation[::n_validation] 

            # keep track of the losses
            self.losses_epochs_validation = [loss_function(self.model(i_validation), t_validation).item()]

            # print the loss before training
            print('Epoch [{}/{}], Loss: {:.4f}, Loss validation: {:.4f}'.format('0', num_epochs, self.losses_epochs[-1], self.losses_epochs_validation[-1]))            
        else:
            # print the loss before training
            print('Epoch [{}/{}], Loss: {:.4f}'.format('0', num_epochs, self.losses_epochs[-1]))

        # training loop
        for epoch in range(num_epochs):

            # Shuffle the dataset
            indices = torch.randperm(input_data.size(0))
            input_data = input_data[indices]
            target_data = target_data[indices]

            # add a new entry to the epoch losses
            self.losses_epochs.append(0)

            # batch training
            for i in range(0, input_data.size(0), batch_size):

                inputs = input_data[i:i+batch_size]
                targets = target_data[i:i+batch_size]

                # Forward pass
                outputs = self.model(inputs)

                # Compute the loss
                loss = loss_function(outputs, targets)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Store the loss
                self.losses_batches.append(loss.item())

                # print the loss of the batch
                if print_batch:
                    print('- Epoch [{}/{}], i: [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i, input_data.size(0), loss.item()), end='\r')

                # add to the epoch loss
                self.losses_epochs[-1] += loss.item() 

                # keep track of the parameters
                if self.keep_track_params:  self.parameters.append(deepcopy(dict(self.model.state_dict().items())))

            # divide the epoch loss by the number of batches, to get the average loss
            self.losses_epochs[-1] /= (input_data.size(0)/batch_size)

            # Validation
            if validation:
                self.losses_epochs_validation.append(loss_function(self.model(i_validation), t_validation).item()) 

            # print the loss of "n_print_validation" strings of the validation data
            if validation:
                n_print_validation = min(n_print_validation, len(i_validation)) #if n_print_validation is bigger than the number of validation data points, we print all of them
                for i in range(n_print_validation):
                    prediction = self.model(i_validation[i])
                    target = t_validation[i]
                    print('\t Validation string, \t i: {}; \t prediction: {:.4f}, \t target: {:.4f}, \t loss: {:.4f}'.format(i, prediction.item(), target.item(), loss_function(prediction, target).item()))


            if time:
                # time
                # Compute elapsed time and remaining time
                elapsed_time = t.time() - start_time
                avg_time_per_epoch = elapsed_time / (epoch + 1)
                remaining_epochs = num_epochs - (epoch + 1)
                estimated_remaining_time = avg_time_per_epoch * remaining_epochs

                # Convert remaining time to hours, minutes, and seconds for better readability
                hours, remainder = divmod(estimated_remaining_time, 3600)
                minutes, seconds = divmod(remainder, 60)

                # Print the loss and remaining time for this epoch
                print('Epoch [{}/{}], Loss: {:.4f}, Loss validation: {:.4f}, Time remaining: ~{}h {}m {:.0f}s'.format(
                    epoch+1, num_epochs, self.losses_epochs[-1], self.losses_epochs_validation[-1], hours, minutes, seconds))
            else:
                # Print the loss for this epoch
                print('Epoch [{}/{}], Loss: {:.4f}, Loss validation: {:.4f}'.format(epoch+1, num_epochs, self.losses_epochs[-1], self.losses_epochs_validation[-1]))

    def plot_parameter(self, layer, index=None, save=False):

        parameter_evolution = []

        if index is None:
            for i in range(len(self.parameters)):
                parameter_evolution.append(torch.mean(self.parameters[i][layer]).item())
        else:
            for i in range(len(self.parameters)):
                parameter_evolution.append(self.parameters[i][layer][index].item())

        plt.figure()
        plt.plot(parameter_evolution)
        plt.xlabel('Epoch')
        plt.ylabel('Parameter value')
        plt.title('Parameter ({}, {})'.format(layer, index))

        if save:
            file = f.get_name_file_to_save(self.name_notebook, self.initial_path, extension="png", version=self.version, postfix= "_parameter_{}_{}".format(layer, index))
            plt.savefig(file)
            print("Saved in: ", file)

        plt.show()


    def print_validation(self, save=False, precision=3, percentatge=1):

        # data cut with percentatge
        data_X_validation = self.data_X_validation[:int(len(self.data_X_validation)*percentatge)]
        data_Y_validation = self.data_Y_validation[:int(len(self.data_Y_validation)*percentatge)]

        # varaibles
        avg_loss = 0
        output_lines = []
        format_string = 'i: {}, \t\t target: {:.%df}, \t output: {:.%df}, \t loss: {:.%df}' % (precision, precision, precision)

        # function to print and save
        def output(string):
            output_lines.append(string)
            print(string)

        # print and save in variables
        for x, (i, t) in enumerate(zip((data_X_validation), data_Y_validation)):
            outputs = self.model(i)
            loss = self.loss_function(outputs, t)
            avg_loss += loss/len(self.data_Y_validation)
            output(format_string.format(x, t.item(), outputs.item(), loss))

        output('Average loss: {:.{}f}'.format(avg_loss, precision))

        # save
        if save:
            save_filename = f.get_name_file_to_save(self.name_notebook, self.initial_path, extension="txt", version=self.version, postfix="_validation")
            with open(save_filename, 'w') as file:
                file.write("Validation Results:\n")
                file.write("\n".join(output_lines))
                print("Saved in: ", save_filename)

    def plot_losses(self, batches=True, epochs=True, epochs_validation=True, save=False):
        
        if batches:
            plt.figure()
            plt.plot(self.losses_batches)
            plt.title('Loss per batch')
            if save: 
                file = f.get_name_file_to_save(self.name_notebook, self.initial_path, extension="png", version=self.version, postfix="_losses_batches")
                plt.savefig(file)
                print("Saved in: ", file)
            plt.show()

        if epochs:
            plt.figure()
            plt.plot(self.losses_epochs)
            plt.title('Loss per epoch')
            if save: 
                file = f.get_name_file_to_save(self.name_notebook, self.initial_path, extension="png", version=self.version, postfix="_losses_epoch")
                plt.savefig(file)
                print("Saved in: ", file)
            plt.show()

        if epochs_validation:
            plt.figure()
            plt.plot(self.losses_epochs_validation)
            plt.title('Loss per epoch (validation)')
            if save: 
                file = f.get_name_file_to_save(self.name_notebook, self.initial_path, extension="png", version=self.version, postfix="_losses_epoch_validation")
                plt.savefig(file)
                print("Saved in: ", file)
            plt.show()


    def save_state_dict(self):

        output_filename = f.get_name_file_to_save(self.name_notebook, self.initial_path, extension="pth", version=self.version)

        torch.save(self.model.state_dict(), output_filename)

        print("Model saved as {}".format(output_filename))


    def load_state_dict(self, version=None, initial_path=None, name_notebook=None):

        version       = self.version       if version is None       else version
        initial_path  = self.initial_path  if initial_path is None  else initial_path
        name_notebook = self.name_notebook if name_notebook is None else name_notebook

        input_filename = initial_path + "checkpoints/"+ name_notebook[:4] +"/models/" + name_notebook[:-6] + "_" + str(version) + ".pth"

        #check if the input file exists
        if not os.path.exists(input_filename):
            print("The file {} does not exist".format(input_filename))
            return

        self.model.load_state_dict(torch.load(input_filename))

        print("Model loaded from {}".format(input_filename))