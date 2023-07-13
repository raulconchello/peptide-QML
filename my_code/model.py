import pennylane as qml
import numpy
from pennylane import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output



class score_predictor:

    def __init__(   
            self, 
            n_qubits,
            data, 
            circuit_layers,
            device = "default.qubit",
            optimizer = qml.NesterovMomentumOptimizer(0.5),
            batch_size = 5,
            bias = True,
            output_to_prediction = lambda x: x,
        ):

        # number of qubits and device
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits)

        # data
        self.data_X = data[0]
        self.data_Y = data[1]

        # circuit_layers
        self.circuit_layers = circuit_layers
        for layer in self.circuit_layers: layer.n_qubits = self.n_qubits # set the number of qubits for each layer
        if self.circuit_layers[0].statepreparation_layer == False:
            raise Exception('The first circuit layer must be a state preparation layer.')
        if self.circuit_layers[-1].mesuarment_layer == False:
            raise Exception('The last circuit layer must be a mesuarment layer.')

        # optimizer and batch size
        self.opt = optimizer
        self.batch_size = batch_size

        # parameters and bias
        shape_params = [layer.shape_params for layer in self.circuit_layers if layer.shape_params is not None]
        self.params = [np.random.randn(*shape, requires_grad=True) for shape in shape_params] 
        if bias: self.params.append(np.array(0.0, requires_grad=True))
        self.bias = bias

        # output to prediction function
        self.output_to_prediction = output_to_prediction

        # training records    
        self.costs = []
        self.accuracies = []
        self.last_cost = None

    def variational_classifier(self, input, params, draw=False, draw_options={}):

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
            fig, ax = qml.draw_mpl(circuit, expansion_strategy="device")(self.data_X[0], self.params)
            fig.set_size_inches((6, 3))

        bias = params[-1]  if self.bias else 0
        return circuit(input, params) + bias
    
    def draw_circuit(self):
        draw_options = {}
        self.variational_classifier(self.data_X[0], self.params, draw=True, draw_options=draw_options)
    
    def square_loss(self, labels, predictions):
        loss = 0
        for l, p in zip(labels, predictions):
            loss = loss + (l - p) ** 2
        loss = loss / len(labels)
        return loss
    
    def accuracy(self, labels, predictions):
        loss = 0
        for l, p in zip(labels, predictions):
            if abs(l - p) < 1e-5:
                loss = loss + 1
        loss = loss / len(labels)
        return loss
    
    def cost(self, X, Y, *params):
        output = [self.variational_classifier(x, params) for x in X]
        cost = self.square_loss(Y, output)
        self.last_cost = { #TODO history of costs and parameters
            'X': X,
            'Y': Y,
            'output': output
        }
        return cost
    
    def train(self, iterations, plot=True):
        for it in range(iterations):

            # Clear previous iteration outputs
            clear_output(wait=True)

            # Update the weights by one optimizer step
            batch_index = np.random.randint(0, len(self.data_X), (self.batch_size,))
            X_batch = self.data_X[batch_index]
            Y_batch = self.data_Y[batch_index]
            params, cost = self.opt.step_and_cost(self.cost, X_batch, Y_batch, *self.params)

            # Update parameters and append cost
            self.params = params[2:]
            self.costs.append(cost)

            # Compute accuracy
            predictions = [self.output_to_prediction(x) for x in self.last_cost['output']]
            acc = self.accuracy(Y_batch, predictions)
            self.accuracies.append(acc)

            # Print progress (cost and accuracy for this batch)
            print("Iter: {:5d} | Cost: {:0.7f} | Accuracy: {:0.7f} ".format(it + 1, cost, acc))

            if plot: self.plot()


    def plot(self):

        # Create a new figure
        fig, ax1 = plt.subplots()

        # Create a second y-axis that shares the same x-axis
        ax2 = ax1.twinx()

        # Plot cost on the first y-axis
        ax1.plot(self.costs, 'g-')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost', color='g')
        ax1.tick_params('y', colors='g')

        # Plot accuracy on the second y-axis
        ax2.plot(self.accuracies, 'b-')
        ax2.set_ylabel('Accuracy', color='b')
        ax2.tick_params('y', colors='b')

        plt.title('Cost and Accuracy Over Iterations')
        plt.grid(True)
        plt.show()



