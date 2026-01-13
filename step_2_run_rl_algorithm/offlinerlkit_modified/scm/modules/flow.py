"""Generic conditional flow class without specified archtitecture: to be implemented by subclasses."""

from offlinerlkit_modified.scm.modules import StructuralEquation


class GCondFlow(StructuralEquation):
    def __init__(self, name, var_dim):
        super().__init__(var_dim)
        self.name = name

    def forward(self, x, x_pa):
        return self.flow(x, x_pa)
    
    def encode(self, x, x_pa):
        return self.flow.inverse(x, x_pa)
    
    def decode(self, u, x_pa):
        return self.flow(u, x_pa)
    
    # new added 24.04.2024
    # def log_prob(self, x):
    #     return self.flow.log_prob(x)
    
    # new added 21.09.2024
    def log_prob(self, x, context=None):
        if context is None:
            return self.flow.log_prob(x)
        else:
            return self.flow.log_prob(x, context)
    
    