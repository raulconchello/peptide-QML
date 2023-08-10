import pennylane as qml
import numpy as np
import torch
class parts:

    class part:
        def __str__(self):
            return self.__class__.__name__

    # Embeddings
    class Embedding(part):
        def __init__(self, n_qubits):
            self.n_qubits = n_qubits

    class AngleEmbedding(Embedding):
        
        # when called it returns the circuit
        def __call__(self, input):        
            qml.AngleEmbedding(input, wires=range(self.n_qubits))


    # Ansatzes
    class Ansatz(part):
        def __init__(self, n_qubits):
            self.n_qubits = n_qubits
            self.weights_shapes = None
            self.weights_sizes = None
            self.weights_size = None

    class Ansatz_11(Ansatz):

        def __init__(self, n_qubits):
            super().__init__(n_qubits)
            self.weights_shapes = [(n_qubits, 2), (n_qubits//2, 2)]
            self.weights_sizes = [np.product(shape) for shape in self.weights_shapes]
            self.weights_size = np.sum(self.weights_sizes)
    
        def __call__(self, weights):

            # split weights
            weights_1 = weights[:self.weights_sizes[0]].reshape(self.weights_shapes[0])
            weights_2 = weights[self.weights_sizes[0]:].reshape(self.weights_shapes[1])

            # number of qubits
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
    class Measurement(part):

        dict_gate = {
            'X': qml.PauliX,
            'Y': qml.PauliY,
            'Z': qml.PauliZ
        }

        def __init__(self, gate, which_qubit):
            self.which_qubit = which_qubit
            self.str_gate = gate

        def __call__(self, n_qubits):       
            return self.__exp_gate(self.str_gate, self.dict_gate, n_qubits, self.which_qubit)

        class __exp_gate:
            def __init__(self, str_gate, dict_gate, n_qubits, which_qubit):
                
                self.str_gate = str_gate
                self.gate = dict_gate[str_gate]
                self.n_qubits = n_qubits
                self.which_qubit = which_qubit

                if self.which_qubit == 'all':
                    self.qubits = range(n_qubits)
                elif type(self.which_qubit) == int:
                    self.qubits = [self.which_qubit]
                else:
                    self.qubits = self.which_qubit
            
            def __call__(self):
                return [qml.expval(self.gate(i)) for i in self.qubits]

            def __str__(self):
                return f"Measurement('{self.str_gate}', {self.which_qubit})"
  

        
            


class circuit:

    def __init__(
            self, 
            n_qubits, 
            device="default.qubit",
            device_options = {'shots': None},
            qnode_options = {'interface': 'torch'},
            embedding = parts.AngleEmbedding, 
            embedding_ansatz = None,
            block_ansatz = parts.Ansatz_11,
            measurement = parts.Measurement('Z', 1),
            embedding_n_layers = 0,
            different_inputs_per_layer = False,
            block_n_layers = 10,
            wrapper_qlayer = None,
        ):

        # variables
        self.n_qubits = n_qubits
        self.dev = qml.device(device, wires=n_qubits, **device_options)
        self.different_inputs_per_layer = different_inputs_per_layer

        embedding_n_layers = embedding_n_layers if not embedding_ansatz is None else 0
        self.n_layers = (embedding_n_layers, block_n_layers)

        # parts of the circuit
        self.embedding = embedding(n_qubits)
        self.embedding_ansatz = embedding_ansatz(n_qubits) if not embedding_ansatz is None else None
        self.block_ansatz = block_ansatz(n_qubits)
        self.measurement = measurement(n_qubits)

        self.wrapper_qlayer = wrapper_qlayer

        # circuit
        self.input_shape = (n_qubits*(1+embedding_n_layers*different_inputs_per_layer),)
        def _circuit(inputs, block_weights, embedding_weights=None):
            
            #embedding   
            if embedding_weights is None:                
                self.embedding(inputs)

            else:            
                embedding_weights = np.split(embedding_weights, embedding_n_layers)

                if different_inputs_per_layer:
                    
                    inputs = np.split(inputs, embedding_n_layers+1, axis=len(inputs.shape)-1)
                    
                    self.embedding(inputs[0])     
                    for i in range(embedding_n_layers):  
                        self.embedding_ansatz( embedding_weights[i] )
                        self.embedding(inputs[i+1])
                else:
                    self.embedding(inputs)   
                    for i in range(embedding_n_layers):  
                        self.embedding_ansatz( embedding_weights[i] )
                        self.embedding(inputs)
            
            #block
            block_weights = np.split(block_weights, block_n_layers)
            for i in range(block_n_layers):
                self.block_ansatz( block_weights[i] )

            #measurement
            return self.measurement()

        if embedding_n_layers==0:
            def circuit(inputs, block_weights): 
                return _circuit(inputs, block_weights)
            
            self.weights_shape = {
                'block_weights':     (block_n_layers    *self.block_ansatz.weights_size,)
            }

        else:
            def circuit(inputs, embedding_weights, block_weights):
                return _circuit(inputs, block_weights, embedding_weights)
            
            self.weights_shape = {
                'embedding_weights': (embedding_n_layers*self.embedding_ansatz.weights_size,),
                'block_weights':     (block_n_layers    *self.block_ansatz.weights_size,)
            }
        
        # qnode, and torch_qlayer
        self.qnode = qml.QNode(func=circuit, device=self.dev, **qnode_options)
        self.torch_qlayer = qml.qnn.TorchLayer(self.qnode, self.weights_shape)

        if wrapper_qlayer is not None:
            self.torch_qlayer = wrapper_qlayer(self.torch_qlayer)

        self.torch_qlayer.__str__ = self.__str__

    def __call__(self):
        return self.torch_qlayer

    def __str__(self):
        string = "QLayer(\n"
        string += f'\tn_qubits: {self.n_qubits}\n'
        string += f'\tembedding: {self.embedding}\n'
        string += f'\tembedding_ansatz: {self.embedding_ansatz}\n'
        string += f'\tblock_ansatz: {self.block_ansatz}\n'
        string += f'\tmeasurement: {self.measurement}\n'
        string += f'\tembedding_n_layers: {self.n_layers[0]}\n'
        string += f'\tdifferent_inputs_per_layer: {self.different_inputs_per_layer}\n'
        string += f'\tblock_n_layers: {self.n_layers[1]}\n'
        string += f'\twrapper_qlayer: {self.wrapper_qlayer.__name__}\n' if self.wrapper_qlayer is not None else 'wrapper_qlayer: None\n'   
        string += f'\tdevice: \n{self.dev}'.replace('\n', '\n\t\t\t')
        string += '\n)'

        return string

    def draw(self, size=(50,3)):

        inputs = np.array([0 for _ in range(self.input_shape[0])])

        if self.n_layers[0] == 0:        
            weights = np.zeros(self.weights_shape['block_weights']), 
        else:
            weights = np.zeros(self.weights_shape['embedding_weights']), np.zeros(self.weights_shape['block_weights'])

        qml.drawer.use_style("black_white")
        fig, ax = qml.draw_mpl(self.qnode, expansion_strategy="device")(inputs, *weights)
        fig.set_size_inches(size)

        
        

        

    

    