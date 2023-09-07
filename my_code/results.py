import json
import matplotlib.pyplot as plt

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
        **attributes
    ):
        """
            Example:
            results = Results(
                loss = keep(last=True, best=True, history=True),
                accuracy = keep(last=True, best=True, history=True),
            )
        """
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
        dict_to_dump = {'metadata': metadata, 'results': dict(self)}
        with open(path, 'w') as f:
            json.dump(dict_to_dump, f)

    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            dict_to_load = json.load(f)['results']

        results_obj = cls(**{k: keep('last' in v, 'best' in v, 'history' in v) for k, v in dict_to_load.items()})
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