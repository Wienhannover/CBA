import argparse

import torch
import numpy as np

import normflows as nf
from normflows.flows import AutoregressiveRationalQuadraticSpline, MaskedAffineAutoregressive
from normflows.flows import affine

from matplotlib import pyplot as plt

from tqdm import tqdm
import pickle
from sklearn.model_selection import train_test_split
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="to_write")
parser.add_argument("--loss_save_path", type=str, default="to_write")
parser.add_argument("--model_save_path", type=str, default="to_write")


args = parser.parse_args()

# load data
with open(args.data_path, 'rb') as f:
    data = pickle.load(f)

state = data[0]
action = data[1]
reward = data[2]
next_state = data[3]
terminal = data[4]

print('state: ', data[0].shape) # state data
print('action: ', data[1].shape) # action data
print('reward: ', data[2].shape) # reward data
print('next state: ', data[3].shape) # next state data
print('terminal: ', data[4].shape) # terminal data

# utils
def random_sample(data, smaple_size):

    tmp = np.zeros(data.shape[0])
    tmp_index = np.random.choice(range(data.shape[0]), smaple_size, replace=False)
    tmp[tmp_index] = 1
    tmp = tmp.astype(bool)

    return torch.from_numpy(data[tmp]).type(torch.FloatTensor)

def random_sample_with_context(data, data_context, smaple_size):

    tmp = np.zeros(data.shape[0])
    tmp_index = np.random.choice(range(data.shape[0]), smaple_size, replace=False)
    tmp[tmp_index] = 1
    tmp = tmp.astype(bool)

    tmp_1 = torch.from_numpy(data[tmp]).type(torch.FloatTensor)
    tmp_2 = torch.from_numpy(data_context[tmp]).type(torch.FloatTensor)

    return tmp_1, tmp_2

# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
print("using: ......")
print(device)
#------------------------------------------------------------------------------
# model state
# Set up model
# Define 2D Gaussian base distribution
base = nf.distributions.base.DiagGaussian(state.shape[1])

# the model design in deepbc
n_layers = 3
layers= [] 
for _ in range(n_layers):
    layers.append(AutoregressiveRationalQuadraticSpline(state.shape[1], 1, 1))
layers.append(affine.coupling.AffineConstFlow((1,)))

# Construct flow model
model_state = nf.NormalizingFlow(base, layers).to(device)
#*************************************************************
target_data = state
# Train model
max_iter = 20000

num_samples = 2 ** 9
show_iter = 500

loss_hist = np.array([])

optimizer = torch.optim.Adam(model_state.parameters(), lr=5e-4, weight_decay=1e-5)

for it in tqdm(range(max_iter)):
    optimizer.zero_grad()
    
    # Get training samples
    # x = target.sample(num_samples).to(device)
    x = random_sample(target_data, num_samples).to(device)

    # Compute loss
    loss = model_state.forward_kld(x)
    
    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()
    
    # Log loss
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

# Plot loss
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.savefig(f"{args.loss_save_path}/model_state_loss.png")
#------------------------------------------------------------------------------
# model action
# # Define flows
K = 4
context_size = state.shape[1]

latent_size = action.shape[1] # to check, shou be same as the target variable dimension?
hidden_units = 128
hidden_layers = 2

flows = []
for i in range(K):
    flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, 
                                                             num_context_channels=context_size)]
    flows += [nf.flows.LULinearPermute(latent_size)]

# Set base distribution
q0 = nf.distributions.DiagGaussian(action.shape[1], trainable=False)
    
# Construct flow model
model_action = nf.ConditionalNormalizingFlow(q0, flows)

# Move model on GPU if available
model_action = model_action.to(device)
#*************************************************************
target_data = action # action
context_data = state # state


# Train model
max_iter = 20000

num_samples = 2 ** 9
show_iter = 500


loss_hist = np.array([])

optimizer = torch.optim.Adam(model_action.parameters(), lr=5e-4, weight_decay=1e-5)

for it in tqdm(range(max_iter)):
    optimizer.zero_grad()
    
    # Get training samples
    x, x_context = random_sample_with_context(target_data, context_data, num_samples)

    # Compute loss
    loss = model_action.forward_kld(x.to(device), x_context.to(device))
    
    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()
    
    # Log loss
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

# Plot loss
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.savefig(f"{args.loss_save_path}/model_action_loss.png")
#------------------------------------------------------------------------------
# model next state
# # Define flows
K = 4
context_size = state.shape[1] + action.shape[1] # state + action ---> next state

latent_size = next_state.shape[1]
hidden_units = 128
hidden_layers = 2

flows = []
for i in range(K):
    flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, 
                                                             num_context_channels=context_size)]
    flows += [nf.flows.LULinearPermute(latent_size)]

# Set base distribution
q0 = nf.distributions.DiagGaussian(next_state.shape[1], trainable=False)
    
# Construct flow model
model_next_state = nf.ConditionalNormalizingFlow(q0, flows)

# Move model on GPU if available
model_next_state = model_next_state.to(device)
#*************************************************************
target_data = next_state # next state
context_data = np.concatenate((state, action), axis=-1) # state + action

# Train model
max_iter = 20000

num_samples = 2 ** 9
show_iter = 500


loss_hist = np.array([])

optimizer = torch.optim.Adam(model_next_state.parameters(), lr=5e-4, weight_decay=1e-5)

for it in tqdm(range(max_iter)):
    optimizer.zero_grad()
    
    # Get training samples
    x, x_context = random_sample_with_context(target_data, context_data, num_samples)

    # Compute loss
    loss = model_next_state.forward_kld(x.to(device), x_context.to(device))
    
    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()
    
    # Log loss
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

# Plot loss
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.savefig(f"{args.loss_save_path}/model_next_state_loss.png")
#------------------------------------------------------------------------------
# model reward
# Define flows
K = 4
context_size = state.shape[1] + action.shape[1] + next_state.shape[1] # state + action + next state ---> reward

latent_size = 1
hidden_units = 128
hidden_layers = 2

flows = []
for i in range(K):
    flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units, 
                                                             num_context_channels=context_size)]
    flows += [nf.flows.LULinearPermute(latent_size)]

# Set base distribution
q0 = nf.distributions.DiagGaussian(1, trainable=False)
    
# Construct flow model
model_reward_with_next_state = nf.ConditionalNormalizingFlow(q0, flows)

# Move model on GPU if available
model_reward_with_next_state = model_reward_with_next_state.to(device)
#*************************************************************
target_data = reward.reshape(-1,1) # reward
context_data = np.concatenate((state, action, next_state), axis=-1) # state + action + next state

# Train model
max_iter = 20000

num_samples = 2 ** 9
show_iter = 500


loss_hist = np.array([])

optimizer = torch.optim.Adam(model_reward_with_next_state.parameters(), lr=5e-4, weight_decay=1e-5)

for it in tqdm(range(max_iter)):
    optimizer.zero_grad()
    
    # Get training samples
    x, x_context = random_sample_with_context(target_data, context_data, num_samples)

    # Compute loss
    loss = model_reward_with_next_state.forward_kld(x.to(device), x_context.to(device))
    
    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()

    # Log loss
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())

# Plot loss
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.savefig(f"{args.loss_save_path}/model_reward_with_next_state_loss.png")
#------------------------------------------------------------------------------

#*************************************************************
torch.save(model_state.state_dict(), f"{args.model_save_path}/state_flow.pt")
torch.save(model_action.state_dict(), f"{args.model_save_path}/action_flow.pt")
torch.save(model_next_state.state_dict(), f"{args.model_save_path}/next_state_flow.pt")
torch.save(model_reward_with_next_state.state_dict(), f"{args.model_save_path}/reward_with_next_state.pt")
#*************************************************************

#------------------------------------------------------------------------------