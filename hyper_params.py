# -*- coding: utf-8 -*-
import itertools
import copy
import pickle
import inspect
import numpy as np
from datetime import datetime
from collections import OrderedDict

class HyperParamsBase(object):
    def __init__(self):
        # default params
        pass

    def get(self, attr, default=None):
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            return default

    def print_args(self):
        print('===== ARGS =====')
        for x in dir(self):
            if x.startswith('__'):
                continue
            val = getattr(self, x)
            if inspect.ismethod(val):
                continue
            print('{} = {}'.format(x,val))
        print('================')

    def dump_args_as_text(self, filename, print_date=True, comment=''):
        with open(filename, 'w') as f:
            if print_date:
                date = datetime.now().strftime("%Y%m%d-%H%M%S")
                f.write('# {}\n'.format(date))
            if len(comment) > 0:
                f.write('# {}\n'.format(comment))
            for x in dir(self):
                if x.startswith('__'):
                    continue
                val = getattr(self, x)
                if inspect.ismethod(val):
                    continue
                f.write('{}:{}\n'.format(x,val))

    def dump_args_as_pkl(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

class ParamGenerator(object):
    def __init__(self):
        self.fixed_params = OrderedDict()
        self.var_params = OrderedDict()
        self.links = OrderedDict()
        
    def add_params(self, name, param_list, forced_var=False):
        if (isinstance(param_list, list)) or (isinstance(param_list, tuple)):
            if len(param_list) > 1 or forced_var:
                self.var_params[name] = param_list
            else:
                self.fixed_params[name] = param_list
        else:
            self.fixed_params[name] = [param_list]
        
    def add_link(self, parent, child):
        if parent in self.var_params.keys() and child in self.var_params.keys():
            assert len(self.var_params[parent]) == len(self.var_params[child])
            if parent in self.links.keys():
                self.links[parent].append(child)
            else:
                self.links[parent] = [child]
        else:
            assert False, 'Not register {} or {} in var_params'.format(parent, child)
        
    def generate(self, base_params=None, shuffle=False, idx_offset=0):
        if base_params is None:
            base_params = HyperParamsBasic()
        
        # Generate variable param combinations
        var_param_names = [k for k in self.var_params.keys()]
        var_param_vals = [v for v in self.var_params.values()]
        
        var_link_names = []
        var_link_vals = []
        var_link_parents = []
        
        # delete dependency params
        for parent, children in self.links.items():
            for child in children:
                for i, name in enumerate(var_param_names):
                    if name == child:
                        var_link_names.append(var_param_names.pop(i)) 
                        var_link_vals.append(var_param_vals.pop(i))
                        var_link_parents.append(parent)
                        break

        var_param_list = list(itertools.product(*var_param_vals))
        all_params = []

        indices = np.arange(len(var_param_list))
        if shuffle:
            np.random.shuffle(indices)
        
        for i in indices:
            cur_params = copy.deepcopy(base_params)
            
            # param_str = '{:02d}_'.format(i+idx_offset)
            param_str = ''
            
            # Set fix params
            for (name, values) in self.fixed_params.items():
                if not hasattr(cur_params, name):
                    print('[Warning] hyparams doesnot have <{}> attribute'.format(name))
                setattr(cur_params, name, values[0])
                
            # Set variable params
            for name, value in zip(var_param_names, var_param_list[i]):
                if not hasattr(cur_params, name):
                    print('[Warning] hyparams doesnot have <{}> attribute'.format(name))
                setattr(cur_params, name, value)
                param_str += '{}-{}/'.format(name, value)
            
            # Set link params
            for parent, child, values in zip(var_link_parents, var_link_names, var_link_vals):
                pval = getattr(cur_params, parent)
                idx = self.var_params[parent].index(pval)
                setattr(cur_params, child, values[idx])
                param_str += '{}-{}/'.format(child, values[idx])
            
            param_str = param_str[:-1] # remove last '_'
            setattr(cur_params, 'param_str', param_str)
            all_params.append(cur_params)
            
        return all_params