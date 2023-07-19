from typing import Any
import pennylane as qml
from pennylane import numpy as np

###--- GATES ---###
def qml_RZZ(params, wires):
    """
    RZZ gate.
    """
    qml.CNOT(wires=wires)
    qml.RZ(params, wires=wires[1])
    qml.CNOT(wires=wires)

### --- LAYERS --- ###
# layer classes
class layer:

    POSSIBLE_QUBITS = ['all', 'first_half', 'second_half', 'odd', 'even', 'ancillas', 'all_but_first']
    statepreparation_layer = False
    mesuarment_layer = False

    def __init__(self, qubits='all'):        
        """
        Layer class. It is the base class for all other layer classes.

        Parameters:
            - qubits: ... TODO
        """

        self._qubits = qubits
        self.n_qubits_circuit = None
        self.n_ancillas_circuit = None

    #we check if self._qubits is a string or a list of integers if it is a string we check that is a valid string: 'all', 'first_half', 'second_half', 'odd', 'even'
    def __setattr__(self, __name: str, __value: Any):
        if __name == '_qubits' and __value is not None:
            if not isinstance(__value, str) and not isinstance(__value, list):
                raise Exception('Invalid type for qubits attribute. It must be a string or a list of integers.')
            if isinstance(__value, str):
                if __value not in self.POSSIBLE_QUBITS:
                    raise Exception('Invalid string for qubits attribute. It must be one of the following: {}'.format(self.POSSIBLE_QUBITS))
        super().__setattr__(__name, __value)

    @property
    def shape_params(self):
        return None

    @property
    def qubits(self):

        if self.n_qubits_circuit is None:
            raise Exception('n_qubits_circuit is None. It must be set before calling qubits property.')

        if self._qubits == self.POSSIBLE_QUBITS[0]:
            return [i for i in range(self.n_qubits_circuit)]
        if self._qubits == self.POSSIBLE_QUBITS[1]:
            return [i for i in range(self.n_qubits_circuit//2)]
        if self._qubits == self.POSSIBLE_QUBITS[2]:
            return [i for i in range(self.n_qubits_circuit//2, self.n_qubits_circuit)]
        if self._qubits == self.POSSIBLE_QUBITS[3]:
            return [i for i in range(self.n_qubits_circuit) if i%2==1]
        if self._qubits == self.POSSIBLE_QUBITS[4]:
            return [i for i in range(self.n_qubits_circuit) if i%2==0]
        if self._qubits == self.POSSIBLE_QUBITS[5]:
            if self.n_ancillas_circuit is None:
                raise Exception('n_ancillas_circuit is None. You must set some ancillas to use "ancilla".')
            return [i for i in range(-1*self.n_ancillas_circuit, 0)]
        if self._qubits == self.POSSIBLE_QUBITS[6]:
            return [i for i in range(1, self.n_qubits_circuit)]
        else:
            return self._qubits
    
    def gates(self):
        raise NotImplementedError

# state preparation layer classes
class basis_preparation(layer):
    statepreparation_layer = True

    @property
    def len_input(self):
        return len(self.qubits)
    
    @property
    def mock_input(self):
        return np.random.rand(self.len_input)
    
    def _check_input(self, input):
        if len(input) != len(self.qubits):
            raise Exception('State preparation layer: Invalid input size. Data must have the correct input size. In this case, qubits="{}", which means that len(input) should be {}.'.format(self._qubits, len(self.qubits)))
        
    def gates(self, input):
        self._check_input(input)
        qml.BasisState(input, wires=self.qubits)

class angle_preparation(basis_preparation):

    def gates(self, input):
        self._check_input(input)
        qml.AngleEmbedding(input, wires=self.qubits)

# fixed layer classes
class CNOTs_layer(layer):

    def gates(self):
        n_qubits = len(self.qubits)
        qubits = self.qubits

        for i in range(n_qubits): # entangle qubits: qubits 0 and 1 are entangled with CNOT, qubits 2 and 3 are entangled with CNOT, ..., qubits n_wires-1 and 0 are entangled with CNOT
            qml.CNOT(wires=[qubits[i], qubits[(i+1)%n_qubits]])

# parametric layer classes
class rotation_layer(layer):
    
    @property
    def shape_params(self):
        return (len(self.qubits), 3)

    def gates(self, params):
        for i, q in enumerate(self.qubits): # rotate differently each qubit (each rotation needs 3 parameters)
            qml.Rot(params[i, 0], params[i, 1], params[i, 2], wires=q)

class rotationX_layer(layer):

    @property
    def shape_params(self):
        return (len(self.qubits), 1)

    def gates(self, params):
        for i, q in enumerate(self.qubits): # X rotation for each qubit
            qml.RX(params[i,0], wires=q)

class rotationZ_layer(rotationX_layer):

    def gates(self, params):
        for i, q in enumerate(self.qubits): # Z rotation for each qubit
            qml.RZ(params[i,0], wires=q)

class rotationY_layer(rotationX_layer):
    
    def gates(self, params):
        for i, q in enumerate(self.qubits): # Y rotation for each qubit
            qml.RY(params[i,0], wires=q)

class rotationZZ_layer(layer):
    
        @property
        def shape_params(self):
            return (len(self.qubits), 1)
    
        def gates(self, params):
            qubits = self.qubits
            n_qubits = len(qubits)

            for x in range(2):
                for i in range(x,len(qubits),2): # ZZ rotation for neighboring qubits 
                    qml_RZZ(params[i,0], wires=[qubits[i], qubits[(i+1)%n_qubits]]) 



# ancilla layer classes
class ancillas(layer):
    ancilla_layer = True
    
    def __init__(self, n_ancillas, n_qubits='all'):
        super().__init__(n_qubits)
        self.n_ancillas = n_ancillas

    @property
    def shape_params(self):
        return (self.n_ancillas, len(self.qubits)*3 + 2)

    def gates(self, params):
        for i in range(self.n_ancillas):
            i_ancilla = -1*(i+1)
            qml.RX(params[i,0], wires=[i_ancilla])
            qml.RZ(params[i,1], wires=[i_ancilla])

            j = 2
            for q in self.qubits: 
                qml_RZZ(params[i,j], wires=[q,i_ancilla] )
                qml.RX (params[i,j+1], wires=[i_ancilla])
                qml.RZ (params[i,j+2], wires=[i_ancilla])
                j += 3
        

        

# mesurament layer classes
class mesurament(layer):
    mesurament_layer = True

    def gates(self):

        expectations = []
        for i in self.qubits:
            expectations.append(qml.expval(qml.PauliZ(i)))

        return tuple(expectations)

