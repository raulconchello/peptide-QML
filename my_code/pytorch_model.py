from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
import time as t
import matplotlib.pyplot as plt
import os

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

    def __init__(self, circuit_layers):

        # model
        self.model = nn.Sequential(*circuit_layers)

        # keep track of the losses
        self.losses_batches = None
        self.losses_epochs = None
        self.losses_epochs_validation = None

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

        # keep track of the losses
        self.losses_batches = []
        self.losses_epochs = [loss_function(self.model(input_data), target_data).item()]

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


    def print_validation(self):
        avg_loss = 0
        for x, (i, t) in enumerate(zip((self.data_X_validation), self.data_Y_validation)):
            outputs = self.model(i)
            loss = self.loss_function(outputs, t)
            avg_loss += loss/len(self.data_Y_validation)
            print('i: {}, \t target: {:.3f}, \t output: {:.3f}, \t loss: {:.3f}'.format(x, t.item(), outputs.item(), loss))

        print('Average loss: {:.3f}'.format(avg_loss))

    def plot_losses(self, batches=True, epochs=True, epochs_validation=True):
        
        if batches:
            plt.figure()
            plt.plot(self.losses_batches)
            plt.title('Loss per batch')
            plt.show()

        if epochs:
            plt.figure()
            plt.plot(self.losses_epochs)
            plt.title('Loss per epoch')
            plt.show()

        if epochs_validation:
            plt.figure()
            plt.plot(self.losses_epochs_validation)
            plt.title('Loss per epoch (validation)')
            plt.show()

    def save_state_dict(self, name_notebook, initial_path=""):

        output_filename = initial_path + "Notebooks/models/"+ name_notebook[:4] +"/" + name_notebook[:-6] + "_0.pth"

        #check if the output file already exists
        while os.path.exists(output_filename):
            print("The file {} already exists".format(output_filename))
            output_filename = output_filename[:-5] + str(int(output_filename[-5]) + 1) + ".pth"
            print("Trying to save the file as {}".format(output_filename))


        torch.save(self.model.state_dict(), output_filename)

        print("Model saved as {}".format(output_filename))

        return output_filename[-5]

    def load_state_dict(self, name_notebook, version=0, initial_path=""):

        input_filename = initial_path + "Notebooks/models/"+ name_notebook[:4] +"/" + name_notebook[:-6] + "_" + version + ".pth"

        #check if the input file exists
        if not os.path.exists(input_filename):
            print("The file {} does not exist".format(input_filename))
            return

        self.model.load_state_dict(torch.load(input_filename))

        print("Model loaded from {}".format(input_filename))