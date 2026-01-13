"""meta data for Navigation 2D"""

# vars = ['state', 'action', 'next state']

# attrs = ['state', 'action', 'next state']

# attr2int = {'state' : 0, 'action' : 1, 'next state' : 2}

graph_structure = {'state' : [],
                   'action' : ['state'],
                   'next_state': ['state', 'action'],
                   'reward': ['state', 'action', 'next_state']
                   }