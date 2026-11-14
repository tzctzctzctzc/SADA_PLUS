import torch.nn as nn
import numpy as np
from abc import abstractmethod

class BaseModel(nn.Module):
        
    def get_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params


    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        return super().__str__() + '\nTrainable parameters: {}'.format(self.get_params())