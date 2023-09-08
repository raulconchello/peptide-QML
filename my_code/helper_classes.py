import csv
import json
import uuid
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


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
        **attributes
    ):
        """
            Example:
            results = Results(
                model_uuid = model_uuid,
                loss = keep(last=True, best=True, history=True),
                accuracy = keep(last=True, best=True, history=True),
            )
        """
        self.model_uuid = model_uuid
        for attribute, what_to_keep in attributes.items():
            # check if options is a keep object
            if not isinstance(what_to_keep, keep):
                raise ValueError(f"Value for '{attribute}' must be an object of class 'keep'")

            # create attribute
            setattr(self, attribute, results_item(what_to_keep))

    def update(self, update_leader:str=None, **attributes):

        # check if all attributes are available
        for attribute in attributes.keys():
            attribute = getattr(self, attribute, False)
            if not attribute:
                raise ValueError(f"Attribute '{attribute}' not found. Attributes available: {list(self.__dict__.keys())}")

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

    def __call__(self):
        return self.__dict__

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)
        
    def __iter__(self):
        return iter({k: dict(v) for k, v in self.__dict__.items()}.items())

    def dump(self, path, metadata = {}):
        # union of metadata and results
        dict_to_dump = {
            'model_uuid': str(self.model_uuid),
            'metadata': metadata, 
            'results': dict(self)
        }
        with open(path, 'w') as f:
            json.dump(dict_to_dump, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            dict_to_load = json.load(f)['results']
            model_uuid = uuid.UUID(json.load(f)['model_uuid'])

        results_obj = cls(
            model_uuid = model_uuid,
            **{k: keep('last' in v, 'best' in v, 'history' in v) for k, v in dict_to_load.items()}
        )
        for att, dic in dict_to_load.items():
            for k, v in dic.items():
                setattr(getattr(results_obj, att), k, v)
        
        return results_obj

    def plot(self, y_attribute, x_attribute=None, title=None, xlabel=None, ylabel=None):

        
        data_y = getattr(getattr(self, y_attribute), 'history', False)
        if data_y is False: raise ValueError(f"Attribute '{y_attribute}' does not have history")

        if x_attribute is None:
            plt.plot(data_y)
        else:
            data_x = getattr(getattr(self, x_attribute), 'history', False)
            if data_x is False: raise ValueError(f"Attribute '{x_attribute}' does not have history")

            plt.plot(data_x, data_y)
            
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()


## VALIDATION CLASSES ##

def computed_required(func):
    def wrapper(self, *args, **kwargs):
        if not self.computed:
            raise ValueError("You need to compute the validation first. Use validation.compute(pct=pct) to compute the validation for the test dataset.")
        return func(self, *args, **kwargs)
    return wrapper

class validation:
    
    def __init__(self, model, model_uuid):
        self.model = model  
        self.model_uuid = model_uuid
        self.computed = False   

    def compute(self, pct=1):  
        self.pct = pct

        if getattr(self, 'last_pct', 0) != self.pct:
            self.last_pct = self.pct

            # random order for test data
            random_order = np.random.permutation(len(self.model.data.x_test))
            x_test = self.model.data.x_test[random_order]
            y_test = self.model.data.y_test[random_order]
            
            # data
            self.x_test = x_test[:int(len(x_test)*self.pct)]
            self.y_test = y_test[:int(len(y_test)*self.pct)]
            self.y_prediction = []
            self.losses = []

            self.model.eval()
            with torch.no_grad():
                for x, y in zip(self.x_test, self.y_test):
                    prediction = self.model(x)
                    self.y_prediction.append(prediction.item())
                    self.losses.append(self.model.loss_function(prediction, y).item())

        self.computed = True

    @computed_required
    @property
    def mean_loss(self):       
        return np.mean(self.losses)
    
    @computed_required
    def plot(self, fig_size=(6, 6)):
        plt.figure(figsize=fig_size)
        plt.scatter(self.y_test, self.y_prediction, color='r', label='Actual vs. Predicted', alpha=0.1)
        plt.plot([np.min(self.y_test), np.max(self.y_test)], [np.min(self.y_test), np.max(self.y_test)], 'k--', lw=2, label='1:1 Line')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Predictions vs. True Values (mean loss: {:.4f})'.format(self.mean_loss))
        plt.legend()
        plt.show()

    @computed_required
    def dump(self, path):  
        dict_to_dump = {
            'model_uuid': str(self.model_uuid),
            'pct': self.pct,
            'x_test': self.x_test.tolist(),
            'y_test': self.y_test.tolist(),
            'y_prediction': self.y_prediction.tolist(),
            'losses': self.losses,
        }
        with open(path, 'w') as f:
            json.dump(dict_to_dump, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            dict_to_load = json.load(f)
            model_uuid = uuid.UUID(dict_to_load.pop('model_uuid'))

        validation_obj = cls(
            model = None,
            model_uuid = model_uuid,
        )
        for k, v in dict_to_load.items():
            setattr(validation_obj, k, v)
        
        validation_obj.computed = True
        return validation_obj
                

## DATA CLASSES ##
MAP_TENSOR_TYPE = {
    'float': torch.float64,
    'float32': torch.float32,
    'float64': torch.float64,
    'int': torch.int32,
    'int32': torch.int32,
    'int64': torch.int64,
}
class data:

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
        self.y_train = torch.tensor(self.y_train, dtype=MAP_TENSOR_TYPE[tensors_type_y])
        self.y_test = torch.tensor(self.y_test, dtype=MAP_TENSOR_TYPE[tensors_type_y])

        #inputs
        self.input_params = {
            'description': description,
            'original_data_file': original_data_file,
            'n_aminoacids': n_aminoacids,
            'split_options': split_options,
            'tensors_type_x': tensors_type_x,
            'tensors_type_y': tensors_type_y,
        }

    def set_test_train(self, ptc):
        self.x_test_train = self.x_train[:int(len(self.x_train)*ptc)]
        self.y_test_train = self.y_train[:int(len(self.y_train)*ptc)]

    def get_loader(self, batch_size=32, **dataloader_options):
        return DataLoader(TensorDataset(self.x_train, self.y_train), batch_size=batch_size, **dataloader_options)
    
    def dump(self, initial_path, file_name):
        path = initial_path + "/data/objects/jsons/" + file_name + ".json"
        dict_to_dump = {
            'uuid': str(self.uuid),
            'x_train': self.x_train,
            'x_test': self.x_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'input_params': self.input_params,
        }
        with open(path, 'w') as f:
            json.dump(dict_to_dump, f)

        # add the uuid, file name and description to the list of uuids
        new_data = [{
            "uuid": str(self.uuid), 
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_name": file_name, 
            "n_aminoacids": self.input_params['n_aminoacids'],
            "description": self.input_params['description'],
            "original_data_file": self.input_params['original_data_file'],
        },]
        with open(initial_path + "/data/objects/uuids.json", 'a', newline='') as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=new_data[0].keys())
            csv_writer.writerows(new_data)

    @classmethod
    def load(cls, initial_path, file_name):
        path = initial_path + "/data/objects/jsons/" + file_name + ".json"
        with open(path, 'r') as f:
            dict_to_load = json.load(f)
        data_obj = cls()
        data_obj.uuid = uuid.UUID(dict_to_load['uuid'])
        data_obj.x_train = torch.tensor(dict_to_load['x_train'], dtype=MAP_TENSOR_TYPE[dict_to_load['input_params']['tensors_type_x']])
        data_obj.x_test = torch.tensor(dict_to_load['x_test'], dtype=MAP_TENSOR_TYPE[dict_to_load['input_params']['tensors_type_x']])
        data_obj.y_train = torch.tensor(dict_to_load['y_train'], dtype=MAP_TENSOR_TYPE[dict_to_load['input_params']['tensors_type_y']])
        data_obj.y_test = torch.tensor(dict_to_load['y_test'], dtype=MAP_TENSOR_TYPE[dict_to_load['input_params']['tensors_type_y']])
        data_obj.input_params = dict_to_load['input_params']
        return data_obj