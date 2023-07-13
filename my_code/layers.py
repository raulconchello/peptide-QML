from typing import Any
import pennylane as qml
from pennylane import numpy as np

# layer classes
class layer:

    POSSIBLE_QUBITS = ['all', 'first_half', 'second_half', 'odd', 'even']
    statepreparation_layer = False
    mesuarment_layer = False

    def __init__(self, qubits='all'):        
        """
        Layer class. It is the base class for all other layer classes.

        Parameters:
            - qubits: ... TODO
        """

        self._qubits = qubits
        self.n_qubits = None

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

        if self.n_qubits is None:
            raise Exception('n_qubits is None. It must be set before calling qubits property.')

        if self._qubits == self.POSSIBLE_QUBITS[0]:
            return [i for i in range(self.n_qubits)]
        if self._qubits == self.POSSIBLE_QUBITS[1]:
            return [i for i in range(self.n_qubits//2)]
        if self._qubits == self.POSSIBLE_QUBITS[2]:
            return [i for i in range(self.n_qubits//2, self.n_qubits)]
        if self._qubits == self.POSSIBLE_QUBITS[3]:
            return [i for i in range(self.n_qubits) if i%2==1]
        if self._qubits == self.POSSIBLE_QUBITS[4]:
            return [i for i in range(self.n_qubits) if i%2==0]
        else:
            return self._qubits
    
    def gates(self):
        raise NotImplementedError

# state preparation layer classes
class basis_preparation(layer):
    statepreparation_layer = True

    def gates(self, input):
        if len(input) != len(self.qubits):
            raise Exception('basis_preparation layer: Invalid input size. Data must have the correct input size. In this case, qubits="{}", which means that len(input) should be {}.'.format(self._qubits, len(self.qubits)))
        
        qml.BasisState(input, wires=self.qubits)

# fixed layer classes
class CNOTs_layer(layer):

    def gates(self):
        n_qubits = self.n_qubits
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

        # #CNOTs between all qubits
        # for i in range(n_wires):
        #     for j in range(i+1,n_wires):
        #         qml.CNOT(wires=[i,j])

# measurement layer classes
class mesurament(layer):
    mesuarment_layer = True

    def gates(self):

        expectations = []
        for i in self.qubits:
            expectations.append(qml.expval(qml.PauliZ(i)))

        return tuple(expectations)

