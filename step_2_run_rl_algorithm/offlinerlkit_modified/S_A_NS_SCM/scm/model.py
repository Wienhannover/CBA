"""This class integrates all individual generative models into a single SCM model."""

from offlinerlkit_modified.S_A_NS_SCM.scm.modules import StateFlow, ActionFlow, NextStateFlow, RewardFlow
from offlinerlkit_modified.S_A_NS_SCM.data.meta_data import graph_structure
from offlinerlkit_modified.scm.model import SCM
from json import load

class TransitionSCM(SCM):

    def __init__(self, ckpt_path, vars_dims):
        # vars_dims, the dimensions of state, action and next_state
        models = {"state" : StateFlow(name="state", var_dim=vars_dims[0]), 
                  "action" : ActionFlow(name="action", var_dim=vars_dims[1], context_size=vars_dims[0]),
                  "reward" : RewardFlow(name="reward", var_dim=vars_dims[2], context_size=vars_dims[0]+vars_dims[1]+vars_dims[3]),
                  "next_state" : NextStateFlow(name="next_state", var_dim=vars_dims[3], context_size=vars_dims[0]+vars_dims[1])}
        

        super().__init__(ckpt_path=ckpt_path, graph_structure=graph_structure, **models)