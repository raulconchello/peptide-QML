import uuid
import time
import torch
import datetime
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
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

class Results:
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
            if k not in self.attributes['plain']:
                self.attributes['plain'].append(k)

    def save(self, csv_file='results', folder='Results', additional_csv={}, file_name_prefix=''):
        file_name, initial_path = self.file_name, self.initial_path

        dict_to_save_csv = {
            "model_uuid": str(self.model_uuid),
            "day": self.day,
            "file_name": file_name, 
            **additional_csv,
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        f.save_csv(
            dict_to_save_csv, 
            file_name=csv_file, 
            initial_path=initial_path,
        )
        f.save_pickle(
            self,
            file_name=file_name + file_name_prefix, 
            folder=folder,
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

    def set_test_ptc(self, ptc):
        self.x_test_ptc = self.x_test[:int(len(self.x_test)*ptc)]
        self.y_test_ptc = self.y_test[:int(len(self.y_test)*ptc)]

    def to(self, device):
        self.x_train = self.x_train.to(device)
        self.x_test = self.x_test.to(device)
        self.y_train = self.y_train.to(device)
        self.y_test = self.y_test.to(device)
        self.x_test_ptc = self.x_test_ptc.to(device)
        self.y_test_ptc = self.y_test_ptc.to(device)
        return self

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
    
## OPTIMIZER CLASSES ##
class Optimizer:

    def __init__(self, model, optimizer_class, optimizer_options:dict):
        self.model = model
        self.optimizer = optimizer_class(model.parameters(), **optimizer_options)

    @staticmethod
    def time_left(time_start, n_epochs_total, n_batches_total, n_epochs_done, current_batch):
        time_left = (time.time() - time_start) * (n_epochs_total*n_batches_total / (n_epochs_done*n_batches_total + current_batch) - 1)
        total_hours = int(time_left // 3600)
        total_minutes = int((time_left - total_hours * 3600) // 60)
        total_seconds = int(time_left - total_hours * 3600 - total_minutes * 60)

        # remaining time for the current epoch
        time_left_epoch = (time.time() - time_start) / (n_epochs_done*n_batches_total + current_batch) * (n_batches_total - current_batch)
        epoch_hours = int(time_left_epoch // 3600)
        epoch_minutes = int((time_left_epoch - epoch_hours * 3600) // 60)
        epoch_seconds = int(time_left_epoch - epoch_hours * 3600 - epoch_minutes * 60)

        return epoch_hours, epoch_minutes, epoch_seconds, total_hours, total_minutes, total_seconds

    @staticmethod
    def print_optimizer_status(epoch, n_epochs, batch_idx, n_batches, loss, time_start):
        if batch_idx == None:
            h, m, s, _, _, _ = Optimizer.time_left(time_start, n_epochs, n_batches, epoch, n_batches)
            l_epoch, l_test, l_accuracy = loss['epoch'][-1], loss['test'][-1], loss['accuracy'][-1]                    
            print(f"Epoch {epoch+1}/{n_epochs}, \t loss={l_epoch:.4f}, \t loss test={l_test:.4f}, \t accuracy test={l_accuracy:.4f}, \t\t time left = {h}h {m}m {s}s, \t\t                                     ", end='\n')
        else:
            h, m, s, th, tm, ts = Optimizer.time_left(time_start, n_epochs, n_batches, epoch, batch_idx+1)
            print(f'\t Epoch {epoch+1}/{n_epochs}, \t batch {batch_idx+1}/{n_batches}, \t loss={loss:.4f}, \t total time left = {th}h {tm}m {ts}s, \t epoch time left = {h}h {m}m {s}s                         ', end='\r')

    def optimize_parameters(
            self, 
            data:Data, 
            n_epochs, 
            batch_size, 
            validation=True, 
            save=True,
            save_path=None,
            test_ptc=0.1, 
            loss_fn_options={},
            early_stopping_options={
                'patience': 10,
                'min_delta': 0.001,
            },
        ):

        # checks
        if not isinstance(data, Data):
            raise ValueError("data must be an object of class Data")
        if save and save_path is None:
            raise ValueError("save_path must be specified if save=True")

        # data
        data.set_test_ptc(test_ptc)
        data_loader = data.get_loader(batch_size=batch_size, shuffle=True)

        # dict to save losses
        self.model.losses = { k: [] for k in ['batch', 'epoch', 'test', 'accuracy'] }

        # optimization loop
        time_start, n_batches = time.time(), len(data_loader)
        for epoch in range(n_epochs):
            self.model.train()
            for batch_idx, batch in enumerate(data_loader):

                # optimization step
                loss = self.model.optimization_step(batch, self.optimizer, loss_fn_options)
                self.model.losses['batch'].append(loss)

                # print status
                Optimizer.print_optimizer_status(epoch, n_epochs, batch_idx, n_batches, loss, time_start)

            # epoch loss
            self.model.losses['epoch'].append(sum(self.model.losses['batch'][-n_batches:])/n_batches)

            # validation
            if validation:
                self.model.eval()
                loss, accuracy = self.model.validation((data.x_test_ptc, data.y_test_ptc), loss_fn_options)
                self.model.losses['test'].append(loss)
                self.model.losses['accuracy'].append(accuracy)

            # save
            if save:
                self.model.save(save_path)

            # early stopping
            if early_stopping_options:
                patience, min_delta = early_stopping_options['patience'], early_stopping_options['min_delta']
                if epoch > patience:
                    loss_difference = sum(self.model.losses['batch'][-patience:])/patience - self.model.losses['batch'][-1]
                    if loss_difference < min_delta:
                        print('Early stopping - epoch')
                        break

            # print status
            Optimizer.print_optimizer_status(epoch, n_epochs, None, n_batches, loss, time_start)

    
## SWEEP CLASSES ##

class Sweep:
    def __init__(self, name_notebook, initial_path, description=None, **params):

        # trace attributes
        self.name_notebook = name_notebook
        self.uuid = uuid.uuid4()
        self.description = description
        self.day = f.get_day()
        self.initial_path = initial_path
        self.version = f.get_version(initial_path, name_notebook, file='sweep_uuids')

        # create points
        self.params = params        
        self.list_points = list(product(*params.values()))
        self.n_points = len(self.list_points)

        # more attributes
        self.added_data = {k: {} for k in range(0, self.n_points)}
        self.start_time = time.time()

    @property
    def points(self):
        """
        Returns a generator with a dict for each point.
        Attributes of each dict: idx, param1, param2, ..., paramN
        Values of each dict: index, value1, value2, ..., valueN
        """
        for index, point in enumerate(self.list_points):
            yield {'idx': index, **dict(zip(self.params.keys(), point))}

    @property
    def points_w_data(self):
        """
        Returns a generator with a dict for each point. With the added data. 
        Attributes of each dict: idx, param1, param2, ..., paramN, key_data1, key_data2, ..., key_dataN
        Values of each dict: index, value1, value2, ..., valueN, value_data1, value_data2, ..., value_dataN

        Some point may not have added data.
        """
        for index, point in enumerate(self.list_points):
            yield {'idx': index, **dict(zip(self.params.keys(), point)), **self.added_data[index]}

    @property
    def points_only_w_data(self):
        """
        Returns a generator with a dict for each point that has data added. With the added data. 
        Attributes of each dict: key_data1, key_data2, ..., key_dataN
        Values of each dict: value_data1, value_data2, ..., value_dataN
        """
        for index, point in enumerate(self.list_points):
            if self.added_data[index]:
                yield {'idx': index, **dict(zip(self.params.keys(), point)), **self.added_data[index]}

    # def points_constrained(self, constraints):  
    #     """
    #     Returns a generator with a dict for each point, constrained by the constraints.
    #     Example of constraints: [('param1', 'operator1', value1), ('param2', 'operator2', value2), ...]
    #     'operator' can be: '==', '!=', '>', '<', '>=', '<='
    #     """
    #     for index, point in enumerate(self.list_points):
    #         if all([eval(f"{point[k]} {operator} {v}") for k, operator, v in constraints]):
    #             yield {'idx': index, **dict(zip(self.params.keys(), point))}

    @property
    def points_left(self):
        """
        Returns a generator with a dict for each point that has not been added data.
        Attributes of each dict: idx, param1, param2, ..., paramN
        Values of each dict: index, value1, value2, ..., valueN

        Useful to continue a sweep that was stopped.
        """
        for index, point in enumerate(self.list_points):
            if not self.added_data[index]:
                yield {'idx': index, **dict(zip(self.params.keys(), point))}

    @property
    def lists(self):
        """
        Returns a dict with a list for each parameter.
        """
        lists = {k: [] for k in tuple(self.points_only_w_data)[0].keys()}
        for point in self.points_only_w_data:
            for key, value in point.items():
                lists[key].append(value)
        return lists
    
    @property
    def arrays(self):
        """
        Returns a dict with a array for each parameter.
        """
        return {k: np.array(v) for k, v in self.lists.items()}

    def arrays_constrained(self, constraints):  
        """
        Returns a dict with a array for each parameter, constrained by the constraints.
        Example of constraints: [('param1', 'operator1', value1), ('param2', 'operator2', value2), ...]
        'operator' can be: '==', '!=', '>', '<', '>=', '<='
        """
        arrays = self.arrays
        for k, operator, v in constraints:
            if isinstance(v, str):
                arrays[k] = np.array([str(i) for i in arrays[k]])
                
            constraint = eval(f"arrays[k] {operator} v")
            for a in arrays.keys():
                arrays[a] = arrays[a][constraint]
        return arrays
    
    @property
    def file_name(self):
        return f"{self.name_notebook}-{self.version}"

    def __iter__(self):
        return iter(self.points_left)

    def add_data(self, idx, **info):
        self.added_data[idx].update(info)

    def get_data(self, idx):
        return self.added_data[idx]
    
    def reset(self):
        self.added_data = {k: {} for k in range(0, self.n_points)}
    
    def get_point(self, idx):
        return {**list(self.points)[idx], **self.added_data[idx]}
    
    def get_point_min(self, key, fix=[], to_string=[]):
        arrays = self.arrays
        for k in to_string:
            arrays[k] = np.array([str(i) for i in arrays[k]])
            
        if fix:
            idx = arrays['idx'][np.all([arrays[k] == v for k, v in fix], axis=0)][arrays[key][np.all([arrays[k] == v for k, v in fix], axis=0)].argmin()]
            return self.get_point(idx)
        else:
            idx = arrays['idx'][arrays[key].argmin()]
            return self.get_point(idx)
    
    def save(self, csv=True, pickle=True):
        dict_to_save_csv = {
            "sweep_uuid": str(self.uuid),
            "day": self.day,
            "notebook": self.name_notebook,
            'version': self.version,
            "params": str(list(self.params.keys())),
            "ranges": str(list(self.params.values())),
            "description": self.description,
            'date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        if csv:
            f.save_csv(
                dict_to_save_csv, 
                file_name='sweep_uuids', 
                initial_path=self.initial_path,
            )
        if pickle:
            f.save_pickle(
                self,
                file_name=self.file_name,
                folder='Sweeps',
                initial_path=self.initial_path,
                day=self.day,
            )

    @classmethod
    def load(cls, initial_path, day, file_name):
        return f.load_pickle(
            file_name=file_name,
            folder='Sweeps',
            initial_path=initial_path,
            day=day,
        )
    
    def print_sweep_status(self, idx):

        print("\n\n --- SWEEP POINT {}/{}: {} ---".format(idx+1, self.n_points, self.list_points[idx]))

        #time
        if idx > 0:
            estimated_remaining_time = (time.time() - self.start_time) / idx * (self.n_points - idx) 
            hours, remainder = divmod(estimated_remaining_time, 3600) 
            minutes, seconds = divmod(remainder, 60)

            # Print remaining time 
            print(' --- time reamining: {:0>2}:{:0>2}:{:05.2f} \n'.format(int(hours),int(minutes),seconds)) 
        
        else:
            print(' --- parameters sweeping: {} \n'.format(list(self.params.keys())))

    def plot(self, x_key, y_key, legend_keys=[], fit_degree=2, replace=[], constraints=[], to_string=[], figsize=(10,6), colors=f.COLORS):

        arrays = self.arrays_constrained(constraints)
        for k in to_string:
            arrays[k] = np.array([str(i) for i in arrays[k]])

        plt.figure(figsize=figsize)
        
        for color, dict_values_legend in zip(colors, [{legend_keys[i]: value for i, value in enumerate(x)} for x in product(*(np.unique(arrays[k]) for k in legend_keys))]):

            points_to_plot = np.all([arrays[k] == v for k, v in dict_values_legend.items()], axis=0) if dict_values_legend else np.ones(len(arrays[x_key]), dtype=bool)

            if any(points_to_plot):                
                x, y = arrays[x_key][points_to_plot], arrays[y_key][points_to_plot]

                f.plot_w_poly_fit(
                    x, y, degree=fit_degree, 
                    options_data={
                        'marker': 'x', 
                        'linestyle': '', 
                        'color': color, 
                        'label': f.replace_string(str(dict_values_legend), replace + [('\'', ''), ('{', ''), ('}', '')]),
                    },
                    options_fit={
                        'linestyle': '--', 
                        'color': color, 
                        'alpha': 0.5
                    }
                )

        plt.legend()
        x_label, y_label = f.replace_string(x_key, replace), f.replace_string(y_key, replace), 
        fix_title = f.replace_string(' (' + ', '.join(f'{k}{c}{v}' for k, c, v in constraints) + ')', replace) if constraints else ''
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"'{y_label}' vs '{x_label}'" + fix_title)
        plt.show()


class SweepUnion(Sweep):
    def __init__(self, sweeps):
        self.sweeps = sweeps
        self.n_sweeps = len(sweeps)
        self.n_points = sum([sweep.n_points for sweep in sweeps])

    @property
    def points(self):
        for sweep in self.sweeps:
            for point in sweep.points:
                yield {**point, 'sweep_uuid': sweep.uuid}

    @property
    def points_w_data(self):
        for sweep in self.sweeps:
            for point in sweep.points_w_data:
                yield {**point, 'sweep_uuid': sweep.uuid}     

    @property
    def points_only_w_data(self):
        for sweep in self.sweeps:
            for point in sweep.points_only_w_data:
                yield {**point, 'sweep_uuid': sweep.uuid}

    @property
    def file_name(self):
        raise NotImplementedError

    def __iter__(self):
        return iter(self.points)

    def add_data(self, idx, **info):
        raise NotImplementedError

    def get_data(self, idx):
        raise NotImplementedError
    
    def save(self, csv=True, pickle=True):
        raise NotImplementedError

    @classmethod
    def load(cls, initial_path, day, file_name):
        raise NotImplementedError
    
    def print_sweep_status(self, idx):
        raise NotImplementedError