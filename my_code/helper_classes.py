import csv
import json
import uuid
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from . import helper_functions as f


## RESULTS CLASSES ##

class keep():
    def __init__(
        self, 
        last = True, 
        best = False,
        history = False, 
    ):
        if not any([last, best, history]):
            raise ValueError("At least one of 'last', 'best', 'history' must be True")

        self.last = last
        self.best = best
        self.history = history

    def __repr__(self):
        return str(self.__dict__)

    
class results_item():
    def __init__(self, what_to_keep = keep()):  

        self.what_to_keep = what_to_keep      

        if what_to_keep.last:
            self.last = None
        if what_to_keep.best:
            self.best = None
        if what_to_keep.history:
            self.history = []

    def __call__(self):
        if self.what_to_keep.last:
            return self.last
        if self.what_to_keep.best:
            return self.best
        if self.what_to_keep.history:
            return self.history

    def __repr__(self):
        return str({k: v for k, v in self.__dict__.items() if k != 'what_to_keep'})
    
    def __iter__(self):
        return iter({k: v for k, v in self.__dict__.items()  if k != 'what_to_keep'}.items())

    def update(self, value, update_best=None):
        best_changed = False
        if self.what_to_keep.last:
            self.last = value

        if self.what_to_keep.best:
            if update_best is None:
                if self.best is None or self.best > value:         
                    best_changed = True
                    self.best = value
            elif update_best:
                best_changed = True
                self.best = value

        if self.what_to_keep.history:
            self.history.append(value)    
        return best_changed

    def __getitem__(self, key):
        return getattr(self, key)

class Results():
    def __init__(
        self, 
        model_uuid,
        file_name,
        day,
        initial_path,
        metadata,
        **attributes
    ):
        """
            Example:
            results = Results(
                model_uuid = model_uuid,
                file_name = file_name,
                day = day,
                initial_path = initial_path,
                metadata = metadata,
                loss = keep(last=True, best=True, history=True),
                accuracy = keep(last=True, best=True, history=True),
            )
        """
        self.model_uuid = model_uuid
        self.file_name = file_name
        self.day = day
        self.initial_path = initial_path
        self.metadata = metadata
        for attribute, what_to_keep in attributes.items():
            # check if options is a keep object
            if not isinstance(what_to_keep, keep):
                raise ValueError(f"Value for '{attribute}' must be an object of class 'keep'")

            # create attribute
            setattr(self, attribute, results_item(what_to_keep))

        self.attributes = {
            'items': list(attributes.keys()),
            'plain': [],
        }

        self.models_id = []

    def __call__(self):
        return self.__dict__

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)
        
    def __iter__(self):
        items = {k: dict(v) for k, v in self.__dict__.items() if k in self.attributes['items']}
        plain = {k: v for k, v in self.__dict__.items() if k in self.attributes['plain']}
        return iter({**items, **plain}.items())

    def update(self, update_leader:str=None, **attributes):

        # check if all attributes are available
        for attribute in attributes.keys():
            if not hasattr(self, attribute):
                raise ValueError(f"Attribute '{attribute}' not found. Attributes available: {list(self.__dict__.keys())}")
            attribute = getattr(self, attribute)

        # update attributes
        if update_leader is None: # if there is no leader, update all attributes
            for attribute, value in attributes.items():
                attribute = getattr(self, attribute, False)
                attribute.update(value)
                
        else: # if there is a leader, update first the leader and then the rest (passing if the leader updated the best)
            if update_leader not in attributes.keys():
                raise ValueError(f"Attribute '{update_leader}' not found. It should be one of {list(attributes.keys())}")

            best_changed = getattr(self, update_leader).update(attributes.pop(update_leader))
            for attribute, value in attributes.items():
                getattr(self, attribute).update(value, update_best=best_changed)

    def add_plain_attributes(self, **attributes):
        
        for k, v in attributes.items():                    
            setattr(self, k, v)
            self.attributes['plain'].append(k)

    def save(self):
        file_name, initial_path = self.file_name, self.initial_path
        dict_to_save_csv = {
            "model_uuid": str(self.model_uuid),
            "day": self.day,
            "file_name": file_name, 
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        f.save_csv(
            dict_to_save_csv, 
            file_name='results', 
            initial_path=initial_path,
        )
        f.save_pickle(
            self,
            file_name=file_name, 
            folder='Results',
            initial_path=initial_path,
            day=self.day,
        )

    @classmethod
    def load(cls, day, file_name, initial_path):   
        return f.load_pickle(
            file_name=file_name,
            folder='Results',
            initial_path=initial_path,
            day=day,
        )
                

## DATA CLASSES ##
MAP_TENSOR_TYPE = {
    'float': torch.float64,
    'float32': torch.float32,
    'float64': torch.float64,
    'int': torch.int32,
    'int32': torch.int32,
    'int64': torch.int64,
}
class Data:

    def __init__(
        self, 
        x, 
        y, 
        split_options,
        tensors_type_x,
        tensors_type_y,
        description,
        original_data_file,
        n_aminoacids,
    ):
        
        self.uuid = uuid.uuid4()

        # split data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, **split_options)

        # to tensor
        if tensors_type_x not in MAP_TENSOR_TYPE or tensors_type_y not in MAP_TENSOR_TYPE:
            raise ValueError(f"tensors_type_x and tensors_type_x must be one of {list(MAP_TENSOR_TYPE.keys())}")
        self.x_train = torch.tensor(self.x_train, dtype=MAP_TENSOR_TYPE[tensors_type_x])
        self.x_test = torch.tensor(self.x_test, dtype=MAP_TENSOR_TYPE[tensors_type_x])
        self.y_train = torch.tensor(self.y_train, dtype=MAP_TENSOR_TYPE[tensors_type_y]).view(-1, 1)
        self.y_test = torch.tensor(self.y_test, dtype=MAP_TENSOR_TYPE[tensors_type_y]).view(-1, 1)

        #inputs
        self.input_params = {
            'description': description,
            'original_data_file': original_data_file,
            'n_aminoacids': n_aminoacids,
            'split_options': split_options,
            'tensors_type_x': tensors_type_x,
            'tensors_type_y': tensors_type_y,
        }

    def set_test_ptc(self, ptc):
        self.x_test_ptc = self.x_test[:int(len(self.x_test)*ptc)]
        self.y_test_ptc = self.y_test[:int(len(self.y_test)*ptc)]

    def get_loader(self, batch_size=32, **dataloader_options):
        return DataLoader(TensorDataset(self.x_train, self.y_train), batch_size=batch_size, **dataloader_options)
    
    def save(self, file_name, initial_path):
        dict_to_save_csv = {
            "data_uuid": str(self.uuid), 
            "file_name": file_name, 
            "n_aminoacids": self.input_params['n_aminoacids'],
            "description": self.input_params['description'],
            "original_data_file": self.input_params['original_data_file'],
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        f.save_csv(
            dict_to_save_csv, 
            file_name='data_uuids', 
            initial_path=initial_path,
        )
        f.save_pickle(
            self,
            file_name=file_name,
            folder='Data',
            initial_path=initial_path,
        )

    @classmethod
    def load(cls, initial_path, file_name):
        return f.load_pickle(
            file_name=file_name,
            folder='Data',
            initial_path=initial_path,
        )