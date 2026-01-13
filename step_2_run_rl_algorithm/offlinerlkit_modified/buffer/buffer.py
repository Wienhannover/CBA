import numpy as np
import torch

from typing import Optional, Union, Tuple, Dict


class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)

        self.device = torch.device(device)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)

    def unflatten(self, us_pr_flat, us, scm):
        """Convert flattened tensor back to dict (and detach)."""
        us_pr_new = {}
        prev = 0
        for key in scm.graph_structure.keys():
            us_pr_new[key] = us_pr_flat[:, prev:(prev + us[key].shape[1])]
            prev += us[key].shape[1]
        return us_pr_new
    

    def load_dataset_and_augment(self, dataset, scm, num_samples_val, smaple_range_val, prob_l, prob_r, index_1, index_2, topk, weight_aug) -> None:

        # # 1
        # observations = np.array(dataset["observations"][:1000], dtype=self.obs_dtype)
        # next_observations = np.array(dataset["next_observations"][:1000], dtype=self.obs_dtype)
        # actions = np.array(dataset["actions"][:1000], dtype=self.action_dtype)
        # rewards = np.array(dataset["rewards"][:1000], dtype=np.float32).reshape(-1, 1)
        # terminals = np.array(dataset["terminals"][:1000], dtype=np.float32).reshape(-1, 1)

        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self.num_samples_val = num_samples_val

        self._ptr = len(observations)
        self._size = len(observations)

        # 2
        # print(self._size)

        #-------------------------------------------------------------------------------------------------------
        tmp_device = torch.device('cpu') # to avoid gpu limit
        #-------------------------------------------------------------------------------------------------------
        print("after loading dataset, start to augment......")
        original_data = {}
        original_data['observations'] = torch.tensor(self.observations).to(tmp_device)
        original_data['next_observations'] = torch.tensor(self.next_observations).to(tmp_device)
        original_data['actions'] = torch.tensor(self.actions).to(tmp_device)
        original_data['rewards'] = torch.tensor(self.rewards).to(tmp_device)
        original_data['terminals'] = torch.tensor(self.terminals).to(tmp_device)

        terminal_index = original_data['terminals'].reshape(-1).to(torch.bool)
        non_terminal_index = ~terminal_index
        print("how many terminals ", torch.sum(original_data['terminals']))
        #-------------------------------------------------------------------------------------------------------
        xs = {}
        xs['state'] = original_data['observations'][non_terminal_index]
        xs['action'] = original_data['actions'][non_terminal_index]
        xs['reward'] = original_data['rewards'][non_terminal_index]
        xs['next_state'] = original_data['next_observations'][non_terminal_index]

        for var in scm.graph_structure.keys():
            scm.models[var] = scm.models[var].to(tmp_device)

        us = {}
        for var in scm.graph_structure.keys():
            if len(scm.graph_structure[var]) == 0:
                us[var] = scm.models[var].encode(xs[var], torch.tensor([]).view(xs[var].shape[0], 0))
            else:
                us[var] = scm.models[var].encode(xs[var], torch.cat([xs[pa] for pa in scm.graph_structure[var]], dim=1))
        us_flat = torch.cat([us[val] for val in scm.graph_structure.keys()], dim=1).to(tmp_device)

        random_perturbations = torch.rand((us_flat.shape[0], self.num_samples_val, us_flat.shape[1])) * 2 * smaple_range_val - smaple_range_val
        # partial perturbate
        if index_1 != 0:
            random_perturbations[:, :, :index_1] = 0 # only perturbate 
        if index_2 != -1:
            random_perturbations[:, :, index_2:] = 0 # only perturbate 
        print(random_perturbations[0][0])
        
        tmp_us_flat = us_flat.unsqueeze(1)
        us_flat_perturbations = (tmp_us_flat + random_perturbations.to(tmp_device)).reshape(-1, us_flat.shape[1])

        # 3 
        # print(random_perturbations.shape) # torch.Size([10, 2, 26]), 10 is #instances, 2 is #samples
        # print(tmp_us_flat.shape) # torch.Size([10, 1, 26])
        # print(us_flat_perturbations.shape) # torch.Size([20, 26])

        tmp_us_ast = self.unflatten(us_flat_perturbations, us, scm)
        tmp_xs_ast = scm.decode(**tmp_us_ast)

        #-------------------------------------------------------
        counterfactual_data_prob = {}
        counterfactual_data_prob['observations'] = tmp_xs_ast['state']
        counterfactual_data_prob['actions'] = tmp_xs_ast['action']
        counterfactual_data_prob['rewards'] = tmp_xs_ast['reward']
        counterfactual_data_prob['next_observations'] = tmp_xs_ast['next_state']
        counterfactual_data_prob['terminals'] = torch.zeros_like(counterfactual_data_prob['rewards'])

        counterfactual_data = {}
        counterfactual_data['observations'] = tmp_xs_ast['state'].reshape(-1, self.num_samples_val, original_data['observations'].shape[1]) # torch.Size([10, 2, 11]) 10 is #instances, 2 is #samples
        counterfactual_data['actions'] = tmp_xs_ast['action'].reshape(-1, self.num_samples_val, original_data['actions'].shape[1])
        counterfactual_data['rewards'] = tmp_xs_ast['reward'].reshape(-1, self.num_samples_val, original_data['rewards'].shape[1])
        counterfactual_data['next_observations'] = tmp_xs_ast['next_state'].reshape(-1, self.num_samples_val, original_data['next_observations'].shape[1])
        counterfactual_data['terminals'] = torch.zeros_like(counterfactual_data['rewards']).to(tmp_device)
        #-------------------------------------------------------

        # calculate instance probability
        tmp_state = scm.models['state'].log_prob(counterfactual_data_prob['observations'])
        tmp_action = scm.models['action'].log_prob(counterfactual_data_prob['actions'], counterfactual_data_prob['observations'])
        tmp_next_state = scm.models['next_state'].log_prob(counterfactual_data_prob['next_observations'], torch.cat((counterfactual_data_prob['observations'], counterfactual_data_prob['actions']), dim=1))
        tmp_reward = scm.models['reward'].log_prob(counterfactual_data_prob['rewards'], torch.cat((counterfactual_data_prob['observations'], counterfactual_data_prob['actions'], counterfactual_data_prob['next_observations']), dim=1))
        print("tmp_state", tmp_state.shape)
        tmp_transition = (tmp_state + tmp_action + tmp_next_state + tmp_reward).reshape(-1, self.num_samples_val)
        print("tmp_transition", tmp_transition.shape)

        tmp_original_state = scm.models['state'].log_prob(xs['state'])
        tmp_original_action = scm.models['action'].log_prob(xs['action'], xs['state'])
        tmp_original_next_state = scm.models['next_state'].log_prob(xs['next_state'], torch.cat((xs['state'], xs['action']), dim=1))
        tmp_original_reward = scm.models['reward'].log_prob(xs['reward'], torch.cat((xs['state'], xs['action'], xs['next_state']), dim=1))
        tmp_original_transition = (tmp_original_state + tmp_original_action + tmp_original_next_state + tmp_original_reward)
        print("tmp_original_transition", tmp_original_transition.shape)

        compared_transition = torch.abs(tmp_transition - tmp_original_transition.unsqueeze(1))
        #-------------------------------------------------------
        _, top_indices = torch.topk(compared_transition, topk, dim=1, largest=False) # ascending order
        # Unsqueeze to match the dimensions of the 3D tensor
        indices_expanded_s = top_indices.unsqueeze(-1).expand(-1, -1, original_data['observations'].shape[1])
        indices_expanded_a = top_indices.unsqueeze(-1).expand(-1, -1, original_data['actions'].shape[1])
        indices_expanded_r = top_indices.unsqueeze(-1).expand(-1, -1, original_data['rewards'].shape[1])
        indices_expanded_ns = top_indices.unsqueeze(-1).expand(-1, -1, original_data['next_observations'].shape[1])

        counterfactual_data_topk = {}
        counterfactual_data_topk['observations'] = torch.gather(counterfactual_data['observations'], 1, indices_expanded_s)
        counterfactual_data_topk['actions'] = torch.gather(counterfactual_data['actions'], 1, indices_expanded_a)
        counterfactual_data_topk['rewards'] = torch.gather(counterfactual_data['rewards'], 1, indices_expanded_r)
        counterfactual_data_topk['next_observations'] = torch.gather(counterfactual_data['next_observations'], 1, indices_expanded_ns)
        counterfactual_data_topk['terminals'] =  torch.zeros_like(counterfactual_data_topk['rewards']).to(self.device)
        print("counterfactual_data_topk", counterfactual_data_topk['observations'].shape)
        #-------------------------------------------------------
        quantile_l = torch.quantile(tmp_original_transition, prob_l)
        quantile_r = torch.quantile(tmp_original_transition, prob_r)
        quantile_mask = (tmp_original_transition >= quantile_l) & (tmp_original_transition <= quantile_r)

        counterfactual_data_topk_quantile = {}
        counterfactual_data_topk_quantile['observations'] = counterfactual_data_topk['observations'][quantile_mask].reshape(-1,  original_data['observations'].shape[1]).to(self.device)
        counterfactual_data_topk_quantile['actions'] = counterfactual_data_topk['actions'][quantile_mask].reshape(-1,  original_data['actions'].shape[1]).to(self.device)
        counterfactual_data_topk_quantile['rewards'] = counterfactual_data_topk['rewards'][quantile_mask].reshape(-1,  original_data['rewards'].shape[1]).to(self.device)
        counterfactual_data_topk_quantile['next_observations'] = counterfactual_data_topk['next_observations'][quantile_mask].reshape(-1,  original_data['next_observations'].shape[1]).to(self.device)
        counterfactual_data_topk_quantile['terminals'] =  torch.zeros_like(counterfactual_data_topk_quantile['rewards']).to(self.device)
        print("counterfactual_data_topk_quantile", counterfactual_data_topk_quantile['observations'].shape)
        #-------------------------------------------------------------------------------------------------------

        self.orginal_data_non_terminal = {
            "observations": original_data['observations'][non_terminal_index].to(self.device),
            "actions": original_data['actions'][non_terminal_index].to(self.device),
            "next_observations": original_data['next_observations'][non_terminal_index].to(self.device),
            "terminals": original_data['terminals'][non_terminal_index].to(self.device),
            "rewards": original_data['rewards'][non_terminal_index].to(self.device)
        }
        self.orginal_data_non_terminal_augmented = counterfactual_data_topk_quantile

        self.orginal_data_terminal = {
            "observations": original_data['observations'][terminal_index].to(self.device),
            "actions": original_data['actions'][terminal_index].to(self.device),
            "next_observations": original_data['next_observations'][terminal_index].to(self.device),
            "terminals": original_data['terminals'][terminal_index].to(self.device),
            "rewards": original_data['rewards'][terminal_index].to(self.device)
        }

        self.orginal_data = {
            "observations": torch.cat((original_data['observations'][non_terminal_index],original_data['observations'][terminal_index]), dim=0).to(self.device),
            "actions": torch.cat((original_data['actions'][non_terminal_index],original_data['actions'][terminal_index]), dim=0).to(self.device),
            "next_observations": torch.cat((original_data['next_observations'][non_terminal_index],original_data['next_observations'][terminal_index]), dim=0).to(self.device),
            "terminals": torch.cat((original_data['terminals'][non_terminal_index],original_data['terminals'][terminal_index]), dim=0).to(self.device),
            "rewards": torch.cat((original_data['rewards'][non_terminal_index],original_data['rewards'][terminal_index]), dim=0).to(self.device)
        }
        #-------------------------------------------------------------------------------------------------------
        original_reward_quantile = self.orginal_data_non_terminal['rewards'][quantile_mask].repeat(1, topk).flatten().reshape(-1, 1)

        higher_reward_index = (self.orginal_data_non_terminal_augmented['rewards'] > original_reward_quantile).reshape(1,-1)[0]
        self.orginal_data_non_terminal_augmented_higher_reward = {
            "observations": self.orginal_data_non_terminal_augmented['observations'][higher_reward_index],
            "actions": self.orginal_data_non_terminal_augmented['actions'][higher_reward_index],
            "next_observations": self.orginal_data_non_terminal_augmented['next_observations'][higher_reward_index],
            "terminals": self.orginal_data_non_terminal_augmented['terminals'][higher_reward_index],
            "rewards": self.orginal_data_non_terminal_augmented['rewards'][higher_reward_index]
        }
        # #--------------------------hopper specific-----------------------------------------------------------------------------
        tmp = self.orginal_data_non_terminal_augmented_higher_reward['observations'][:, 1:]
        index_0 = torch.all((tmp > -100) & (tmp < 100), dim=1)
        tmp = self.orginal_data_non_terminal_augmented_higher_reward['observations'][:, 0]
        index_1 = 0.7 < tmp
        tmp = self.orginal_data_non_terminal_augmented_higher_reward['observations'][:, 1] 
        index_2 = (-0.2 < tmp) & (tmp < 0.2)
        #----------------------------------
        tmp = self.orginal_data_non_terminal_augmented_higher_reward['next_observations'][:, 1:]
        index_0_next_state = torch.all((tmp > -100) & (tmp < 100), dim=1)
        tmp = self.orginal_data_non_terminal_augmented_higher_reward['next_observations'][:, 0]
        index_1_next_state = 0.7 < tmp
        tmp = self.orginal_data_non_terminal_augmented_higher_reward['next_observations'][:, 1] 
        index_2_next_state = (-0.2 < tmp) & (tmp < 0.2)
        #----------------------------------
        final_index = index_0 & index_1 & index_2 & index_0_next_state & index_1_next_state & index_2_next_state
        action_index = torch.any((self.orginal_data_non_terminal_augmented_higher_reward['actions'] > -1) & (self.orginal_data_non_terminal_augmented_higher_reward['actions'] < 1), dim=1)
        final_index_with_proper_action = final_index & action_index
        self.orginal_data_non_terminal_augmented_higher_reward_healthy = {
            "observations": self.orginal_data_non_terminal_augmented_higher_reward['observations'][final_index_with_proper_action],
            "actions": self.orginal_data_non_terminal_augmented_higher_reward['actions'][final_index_with_proper_action],
            "next_observations": self.orginal_data_non_terminal_augmented_higher_reward['next_observations'][final_index_with_proper_action],
            "terminals": self.orginal_data_non_terminal_augmented_higher_reward['terminals'][final_index_with_proper_action],
            "rewards": self.orginal_data_non_terminal_augmented_higher_reward['rewards'][final_index_with_proper_action]
        }
        #-------------------------------------------------------------------------------------------------------
        print("original non terminal data")
        print(self.orginal_data_non_terminal['observations'].shape)
        print("original non terminal augmented data")
        print(self.orginal_data_non_terminal_augmented['observations'].shape)
        print("original non terminal augmented data higher reward")
        print(self.orginal_data_non_terminal_augmented_higher_reward['observations'].shape)
        # print("original non terminal augmented data healthy")
        # print(self.orginal_data_non_terminal_augmented_healthy['observations'].shape)
        print("original non terminal augmented data higher reward and healthy")
        print(self.orginal_data_non_terminal_augmented_higher_reward_healthy['observations'].shape)
        #-------------------------------------------------------------------------------------------------------

        #-------------------------------------------------------------------------------------------------------
        self.training_data = {
            "observations": torch.cat((self.orginal_data_non_terminal_augmented_higher_reward_healthy['observations'],self.orginal_data['observations']), dim=0),
            "actions": torch.cat((self.orginal_data_non_terminal_augmented_higher_reward_healthy['actions'],self.orginal_data['actions']), dim=0),
            "next_observations": torch.cat((self.orginal_data_non_terminal_augmented_higher_reward_healthy['next_observations'],self.orginal_data['next_observations']), dim=0),
            "terminals": torch.cat((self.orginal_data_non_terminal_augmented_higher_reward_healthy['terminals'],self.orginal_data['terminals']), dim=0),
            "rewards": torch.cat((self.orginal_data_non_terminal_augmented_higher_reward_healthy['rewards'],self.orginal_data['rewards']), dim=0)
        }
        
        cutoff = self.orginal_data_non_terminal_augmented_higher_reward_healthy['observations'].shape[0]
        weight_before = weight_aug / cutoff              # Total weight before the cutoff (e.g., 0.6)
        weight_after = (1-weight_aug) / self.orginal_data['observations'].shape[0]
        self.sample_weights = np.array([weight_before] * cutoff + [weight_after] * self.orginal_data['observations'].shape[0])


    def sample_augmented_data(self, batch_size: int) -> Dict[str, torch.Tensor]:

        sample_array = np.arange(self.training_data['observations'].shape[0])

        batch_indexes = np.random.choice(sample_array, size=batch_size, replace=False, p=self.sample_weights)

        return {
            "observations": self.training_data['observations'][batch_indexes],
            "actions": self.training_data['actions'][batch_indexes],
            "next_observations": self.training_data['next_observations'][batch_indexes],
            "terminals": self.training_data['terminals'][batch_indexes],
            "rewards": self.training_data['rewards'][batch_indexes]
            
        }


    # used in td3bc
    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.training_data['observations'].cpu().numpy().mean(0, keepdims=True)
        std = self.training_data['observations'].cpu().numpy().std(0, keepdims=True) + eps
        self.training_data['observations'] = torch.tensor((self.training_data['observations'].cpu().numpy() - mean) / std).to(self.device)
        self.training_data['next_observations'] = torch.tensor((self.training_data['next_observations'].cpu().numpy() - mean) / std).to(self.device)
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std



    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy()
        }