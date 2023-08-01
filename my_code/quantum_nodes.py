import pennylane as qml
import numpy as np

from .pytorch_wrappers import QLayer

class parts:

    # Embeddings
    class Embedding:
        def __init__(self, n_qubits):
            self.n_qubits = n_qubits

    class AngleEmbedding(Embedding):
        
        # when called it returns the circuit
        def __call__(self, input):        
            qml.AngleEmbedding(input, wires=range(self.n_qubits))


    # Ansatzes
    class Ansatz:
        def __init__(self, n_qubits):
            self.n_qubits = n_qubits
            self.weights_shapes = None

    class Ansatz_11(Ansatz):

        def __init__(self, n_qubits):
            super().__init__(n_qubits)
            self.weights_shapes = [(n_qubits, 2), (n_qubits//2, 2)]
    
        def __call__(self, weights_1, weights_2):

            n_qubits = self.n_qubits
            
            # rotations for each qubit
            for j in range(n_qubits):
                qml.RY(weights_1[j,0], wires=j)
                qml.RZ(weights_1[j,1], wires=j)

            # ZZ rotation for neighboring qubits         
            for j in range(0,n_qubits,2): 
                qml.CNOT(wires=[j, (j+1)])

            # rotations for some qubits
            for j, w in enumerate(range(1, n_qubits, 4)): 
                qml.RY(weights_2[j,0], wires=w)
                qml.RZ(weights_2[j,1], wires=w)
                qml.RY(weights_2[j+1,0], wires=w+1)
                qml.RZ(weights_2[j+1,1], wires=w+1)
                qml.CNOT(wires=[w, (w+1)])

    # Measurements
    class Measurement:
        def __init__(self, n_qubits):
            self.n_qubits = n_qubits

    class exp_Z(Measurement):
        def __init__(self, which_qubit):
            self.which_qubit = which_qubit
            self.gate = qml.PauliZ

        def __call__(self, n_qubits):
            self.n_qubits = n_qubits

            if self.which_qubit == 'all':
                qubits = range(n_qubits)
            elif type(self.which_qubit) == int:
                qubits = [self.which_qubit]
            else:
                qubits = self.which_qubit

            return lambda: [qml.expval(self.gate(i)) for i in qubits]
            


class circuit:

    def __init__(
            self, 
            n_qubits, 
            device="default.qubit",
            device_options = {'shots': None},
            qnode_options = {'interface': 'torch'},
            embedding = parts.AngleEmbedding, 
            ansatz = parts.Ansatz_11,
            measurement = parts.exp_Z(1),
            n_layers = 10,
            wrapper_qlayer = QLayer,
        ):

        # variables
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits, **device_options),
        self.n_layers = n_layers

        # parts of the circuit
        self.embedding = embedding(n_qubits)
        self.ansatz = ansatz(n_qubits)
        self.measurement = measurement(n_qubits)

        # circuit, qnode, and torch_qlayer
        def circuit(input, *weights): 
            #embedding           
            self.embedding(input)

            #block
            for i in range(n_layers):
                self.ansatz( *(w[i] for w in weights) )

            #measurement
            return self.measurement(n_qubits)
        
        self.weights_shapes = {k:(n_layers, *v) for k,v in self.ansatz.weights_shapes.items()}
        
        self.qnode = qml.QNode(circuit, self.dev, **qnode_options)
        self.torch_qlayer = wrapper_qlayer(qml.qnn.TorchLayer(self.qnode, self.ansatz.weights_shapes))

    def __call__(self):
        self.torch_qlayer

    def draw(self, size=(50,3)):

        input = [0 for i in range(self.n_qubits)]
        weights = {k:np.zeros(*v) for k,v in self.weights_shapes.items()}

        qml.drawer.use_style("black_white")
        fig, ax = qml.draw_mpl(self.qnode, expansion_strategy="device")(input, **weights)
        fig.set_size_inches(size)

        
        

        

    

    