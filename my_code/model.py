import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

def get_value(x):
    try:
        return x._value
    except:
        return x

class model:

    def __init__(   
            self, 
            n_qubits_data,
            circuit_layers,
            device = "default.qubit",
            bias = True,
            output_to_prediction = get_value
        ):

        # circuit_layers
        self.circuit_layers = circuit_layers

        # set the number of qubits for each layer
        for layer in self.circuit_layers: layer.n_qubits_circuit = n_qubits_data

        # set the number of ancillas for each layer
        n_ancillas = 0
        for layer in self.circuit_layers: 
            if getattr(layer, 'ancilla_layer', False) == True:
                n_ancillas = layer.n_ancillas if layer.n_ancillas > n_ancillas else n_ancillas

        for layer in self.circuit_layers: layer.n_ancillas_circuit = n_ancillas 

        # check if the first and last layers are state preparation and mesurament layers
        if self.circuit_layers[0].statepreparation_layer == False:
            raise Exception('The first circuit layer must be a state preparation layer.')
        if self.circuit_layers[-1].mesurament_layer == False:
            raise Exception('The last circuit layer must be a mesurament layer.')
        
        # number of qubits and device
        self.n_qubits = n_qubits_data
        self.n_ancillas = n_ancillas
        self.dev = qml.device(device, wires=range(-1*n_ancillas,n_qubits_data))

        # parameters and bias, and training records
        self.bias = bias
        self.initialize_params()
        self.last_cost = None

        # output to prediction function
        self.output_to_prediction = output_to_prediction


        self.times = 0

    def set_data(self, data_X, data_Y, data_X_validation=None, data_Y_validation=None):

        # none of the data requires gradient
        for i in [data_X, data_Y, data_X_validation, data_Y_validation]: 
            if i is not None:
                i.requires_grad = False

        # data
        self.data_X = data_X
        self.data_Y = data_Y
        self.data_validation_X = data_X_validation
        self.data_validation_Y = data_Y_validation

    def initialize_params(self):

        # parameters and bias
        shape_params = [layer.shape_params for layer in self.circuit_layers if layer.shape_params is not None]
        self.params = [np.random.randn(*shape, requires_grad=True) for shape in shape_params] 
        if self.bias: self.params.append(np.array(0.0, requires_grad=True))

        # training records
        self.costs = []
        self.accuracies_batch = []
        self.accuracies_validation = []

    def parametric_model(self, input, params, draw=False, draw_options={}):

        self.times += 1

        @qml.qnode(self.dev, interface="autograd")
        def circuit(input, params):

            # sate preparation
            self.circuit_layers[0].gates(input)

            # variational circuit
            i = 0
            for layer in self.circuit_layers[1:-1]:
                if layer.shape_params is not None:
                    layer.gates(params[i])
                    i += 1
                else:
                    layer.gates()

            # measurement
            return self.circuit_layers[-1].gates()
        
        if draw:
            qml.drawer.use_style("black_white")
            fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(input, self.params)
            fig.set_size_inches(draw_options['size'])

        bias = params[-1]  if self.bias else 0
        
        output = circuit(input, params) + bias
        return output
    
    def draw_circuit(self, size=(6, 3)):
        draw_options = {'size': size}
        self.parametric_model(self.circuit_layers[0].mock_input, self.params, draw=True, draw_options=draw_options)
    
    def loss(self, labels, predictions): # squared loss
        loss = 0
        for l, p in zip(labels, predictions):
            loss = loss + (l - p) ** 2
        loss = loss / len(labels)
        return loss
    
    # def accuracy(self, objectives, predictions):
    #     loss = self.loss(objectives, predictions)
    #     return 1 - loss
    
    def cost(self, X, Y, *params):
        output = [self.parametric_model(x, params) for x in X]
        cost = self.loss(Y, output)
        self.last_cost = { #TODO history of costs and parameters
            'X': X,
            'Y': Y,
            'output': output
        }
        return cost
    
    def train(
                self,  
                epochs = 10, 
                optimizer = qml.GradientDescentOptimizer(),
                cost_every = 1,
                batch_size = 10,
                randomize_batches = False,
                initialize_params = False, 
                plot=True, 
                plot_options={}
            ):
        
        self.times = 0
        
        # set optimizer and batch size
        self.optimizer = optimizer
        self.batch_size = batch_size

        if getattr(self, 'data_X', None) is None or getattr(self, 'data_Y', None) is None:
            raise Exception('Data not set. Use set_data() method.')

        if initialize_params: 
            self.initialize_params()

        iterations = (len(self.data_X) // self.batch_size)

        # train
        for epoch in range(epochs):

            # batches
            if randomize_batches or (epoch == 0):
                # put random order in data
                random_order = np.random.permutation(len(self.data_X))
                data_X = self.data_X[random_order]
                data_Y = self.data_Y[random_order]

                #create batches
                data_X_batches = np.array_split(data_X, iterations)
                data_Y_batches = np.array_split(data_Y, iterations)

            for it in range(iterations):

                # Get batch
                X_batch = data_X_batches[it]
                Y_batch = data_Y_batches[it]                

                # Update parameters and append cost (if optimizer is SPSA, we compute cost every 'x' iterations, where 'x' is cost_every)
                if not isinstance(self.optimizer, qml.SPSAOptimizer) or it % cost_every == 0:
                    params, cost = self.optimizer.step_and_cost(self.cost, X_batch, Y_batch, *self.params)
                    self.params = params[2:]
                    print("Epoch: {} | Iter: {:5d} | Cost: {:0.7f}".format(epoch, it + 1, cost))
                else:
                    self.params, cost = self.optimizer.step(self.cost, X_batch, Y_batch, *self.params)[2:], None
                    print("Epoch: {} | Iter: {:5d}".format(epoch, it + 1))
                self.costs.append(cost)

                # plot progress
                if plot: self.plot(iteration=it, last_iteration=it==(iterations-1), **plot_options)



                # # Compute accuracy batch
                # predictions = [self.output_to_prediction(x) for x in self.last_cost['output']]
                # acc = self.accuracy(Y_batch, predictions)
                # self.accuracies_batch.append(acc)

                # 
                #     if self.data_validation_X is not None:
                #         # Compute accuracy validation
                #         predictions = [self.output_to_prediction(self.parametric_model(x, self.params)) for x in self.data_validation_X]
                #         acc_val = self.accuracy(self.data_validation_Y, predictions)
                #         self.accuracies_validation.append(acc_val)

                #         # Print progress (cost, accuracy batch and accuracy validation for this batch)
                #         print("Epoch: {} | Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.7f} | Accuracy validation: {:0.7f}".format(epoch, it + 1, cost, acc, acc_val))

                #     else:
                #         # Print progress (cost and accuracy for this batch)
                #         cost = self.cost(X_batch, Y_batch, *self.params)

                #         print("Epoch: {} | Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.7f} ".format(epoch, it + 1, cost, acc))

                

            

    def predict(self, input):
        return self.output_to_prediction(self.parametric_model(input, self.params))

    def plot(self, iteration, last_iteration=False, cost=True, accuracy=True, accuracy_validation=True, plot_every=1):

        if (iteration+1) % plot_every == 0 or last_iteration:
            
            # Clear previous iteration outputs
            # clear_output(wait=True)

            # Create a new figure
            fig, ax1 = plt.subplots()

            # Create a second y-axis that shares the same x-axis
            ax2 = ax1.twinx()

            costs = [c for c in self.costs if c is not None]

            if cost:
                # Plot cost on the first y-axis
                ax1.plot(costs, 'g-')
                ax1.set_xlabel('Iteration')
                ax1.set_ylabel('Cost', color='g')
                ax1.tick_params('y', colors='g')

            if accuracy:
                # Plot accuracy on the second y-axis
                ax2.plot(self.accuracies_batch, 'b-')
                ax2.set_ylabel('Accuracy', color='b')
                ax2.tick_params('y', colors='b')

            if self.data_validation_X is not None and accuracy_validation:
                # Plot accuracy (validation) on the second y-axis
                ax2.plot(self.accuracies_validation, 'r-')

            plt.title('Cost and Accuracy Over Iterations')
            plt.grid(True)
            plt.show()



