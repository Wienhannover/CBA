from offlinerlkit_modified.scm.modules import GCondFlow
# from offlinerlkit_modified.custom_components import CondFlow, SigmoidFlow, ConstAddScaleFlow
import normflows as nf

from normflows.flows import AutoregressiveRationalQuadraticSpline, MaskedAffineAutoregressive
from normflows.flows import affine

class StateFlow(GCondFlow):
    def __init__(self, name="state", n_layers=3, var_dim=11):
        super().__init__(name, var_dim)
        base = nf.distributions.base.DiagGaussian(var_dim) # hopper
        layers = [] 
        for _ in range(n_layers):
            layers.append(AutoregressiveRationalQuadraticSpline(var_dim, 1, 1))
        layers.append(affine.coupling.AffineConstFlow((1,)))
        self.flow = nf.NormalizingFlow(base, layers)

    def forward(self, x, x_pa): # no condition here, need to rewrite
        return self.flow(x)

    def encode(self, x, x_pa): # no condition here, need to rewrite
        return self.flow.inverse(x)

    def decode(self, u, x_pa): # no condition here, need to rewrite
        return self.flow(u)
    

class ActionFlow(GCondFlow):
    def __init__(self, name="action", n_layers=4, var_dim=3, context_size=11):
        super().__init__(name, var_dim)

        # Set base distribution
        base = nf.distributions.DiagGaussian(var_dim, trainable=False)
        # Define flows, follow the design in normflows
        context_size = context_size # state ---> action
        latent_size = var_dim

        hidden_units = 128
        hidden_layers = 2
        flows = []
        for i in range(n_layers):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, 
                                                                    num_context_channels=context_size)]
            flows += [nf.flows.LULinearPermute(latent_size)]

        self.flow = nf.ConditionalNormalizingFlow(base, flows)


class NextStateFlow(GCondFlow):

    def __init__(self, name="next_state", n_layers=4, var_dim=11, context_size=14):
        super().__init__(name, var_dim)

        # Set base distribution
        base = nf.distributions.DiagGaussian(var_dim, trainable=False)
        # Define flows, follow the design in normflows
        context_size = context_size # state + action ---> next state
        latent_size = var_dim

        hidden_units = 128
        hidden_layers = 2
        flows = []
        for i in range(n_layers):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, 
                                                                    num_context_channels=context_size)]
            flows += [nf.flows.LULinearPermute(latent_size)]

        self.flow = nf.ConditionalNormalizingFlow(base, flows)

class RewardFlow(GCondFlow):

    def __init__(self, name="reward", n_layers=4, var_dim=1, context_size=25):
        super().__init__(name, var_dim)

        # Set base distribution
        base = nf.distributions.DiagGaussian(var_dim, trainable=False)
        # Define flows, follow the design in normflows
        context_size = context_size # state + action + next state ---> reward
        latent_size = var_dim

        hidden_units = 128
        hidden_layers = 2
        flows = []
        for i in range(n_layers):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, 
                                                                    num_context_channels=context_size)]
            flows += [nf.flows.LULinearPermute(latent_size)]

        self.flow = nf.ConditionalNormalizingFlow(base, flows)
