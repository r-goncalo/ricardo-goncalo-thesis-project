

import numpy as np
import gymnasium as gym
import torch

from math import prod

# INPUT LAYER SIZE --------------------------------------------------------------------------

def discrete_input_layer_size_of_space_gym(state_space : gym.spaces.Space) -> int:
    
    """
    Determines the number of input nodes needed for a given state space.

    :param state_space: A Gymnasium observation space
    :return: Integer representing the required input layer size
    """
    
    if isinstance(state_space, gym.spaces.Box):
        return int(np.prod(state_space.shape))  # Flatten Box spaces (e.g., images, continuous vectors)
    
    elif isinstance(state_space, gym.spaces.Discrete):
        return 1  # Discrete space is a single integer (can be used as an index or one-hot encoded)
    
    elif isinstance(state_space, gym.spaces.MultiDiscrete):
        return len(state_space.nvec)  # Number of discrete dimensions
    
    elif isinstance(state_space, gym.spaces.MultiBinary):
        return state_space.n  # Number of binary values

    elif isinstance(state_space, gym.spaces.Tuple):
        return sum(discrete_input_layer_size_of_space_gym(s) for s in state_space.spaces)  # Sum of all subspaces
    
    elif isinstance(state_space, gym.spaces.Dict):
        return sum(discrete_input_layer_size_of_space_gym(s) for s in state_space.spaces.values())  # Sum of all dictionary subspaces

    
    else:
        raise NotImplementedError(f"Unknown space type: {type(state_space)}")
    
    
    
def discrete_input_layer_size_of_space_torch(state_space : torch.Size) -> int:

    return prod(state_space)
    
    
    
def discrete_input_layer_size_of_space(state_space) -> int:
        
    if isinstance(state_space, torch.Size):
        return discrete_input_layer_size_of_space_torch(state_space)
    
    elif isinstance(state_space, gym.spaces.Space):
        return discrete_input_layer_size_of_space_gym(state_space)
    
    elif isinstance(state_space, tuple):
        
        return prod([ (s if isinstance(s, int) else discrete_input_layer_size_of_space(s)) for s in state_space])
    
    else:
        raise NotImplementedError(f"Unkown space type: {type(state_space)}")
    
    

# TORCH STATE SHAPE FROM SPACE ------------------------------------------------------

def torch_state_shape_from_space_gym(state_space : gym.Space) -> torch.Size:
    
    if isinstance(state_space, gym.spaces.Box):
        return torch.Size(state_space.shape)
    
    else:
        raise NotImplementedError(f"Unknown state space type: {type(state_space)}")
    
    
def torch_state_shape_from_space(state_space) -> torch.Size:

    if isinstance(state_space, gym.Space):
        return torch_state_shape_from_space_gym(state_space)
    
    else:
        raise NotImplementedError(f"Unknown state space type: {type(state_space)}")

# OUTPUT LAYER SIZE ---------------------------------------------------------------------

def discrete_output_layer_size_of_space_gym(action_space : gym.Space):
    
    """
    Determines the number of output nodes needed for a given action gym space.

    :param action_space: A Gymnasium action space
    :return: Integer representing the required output layer size
    """
    
    if isinstance(action_space, gym.spaces.Discrete):
        return action_space.n  # Number of discrete actions (one-hot encoded output)
    
    elif isinstance(action_space, gym.spaces.Box):
        return int(np.prod(action_space.shape))  # Continuous action space (vector output)
    
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        return len(action_space.nvec)  # Number of discrete action dimensions
    
    elif isinstance(action_space, gym.spaces.MultiBinary):
        return action_space.n  # Number of binary actions
    
    elif isinstance(action_space, gym.spaces.Tuple):
        return sum(discrete_output_layer_size_of_space(s) for s in action_space.spaces)  # Sum of all subspaces
    
    elif isinstance(action_space, gym.spaces.Dict):
        return sum(discrete_output_layer_size_of_space(s) for s in action_space.spaces.values())  # Sum of all dictionary subspaces
    
    else:
        raise NotImplementedError(f"Unknown action space type: {type(action_space)}")
    
    
def discrete_output_layer_size_of_space(action_space):
    
    """
    Determines the number of output nodes needed for a given action space.
    """
    
    if isinstance(action_space, gym.spaces.Space):
        return discrete_output_layer_size_of_space_gym(action_space)
    
    elif isinstance(action_space, int):
        return action_space
    
    else:
        raise NotImplementedError(f"Unknown action space type: {type(action_space)}")
    
    
# ACTION SHAPE SIZE ---------------------------------------------------------------------

def single_action_shape_gym(action_space : gym.Space):
    
    """
    Determines the shape needed to encode single actions
    """
    
    if isinstance(action_space, gym.spaces.Discrete):
        return 1  # Number of discrete actions (one-hot encoded output)
    
    else:
        raise NotImplementedError(f"Unknown action space type: {type(action_space)}")

    
def single_action_shape(action_space):
    
    """
    Determines the shape needed to encode single actions
    """
    
    if isinstance(action_space, gym.spaces.Space):
        return single_action_shape_gym(action_space)
    
    elif isinstance(action_space, int):
        return 1
    
    else:
        raise NotImplementedError(f"Unknown action space type: {type(action_space)}")

# ALLOCATING MEMORY FOR SPACE ------------------------------------------------------------------

def torch_zeros_for_space_gym(space, device):
    
    raise NotImplementedError(f"Unsupported space type: {type(space)}")
    
    
def torch_zeros_for_space_torch(space : torch.Size, device) -> int:
    
    return torch.zeros(size=space, device=device)
    
    
def torch_zeros_for_space(state_space, device) -> torch.Tensor:
    
    if isinstance(state_space, torch.Size):
        return torch_zeros_for_space_torch(state_space, device=device)
    
    elif isinstance(state_space, gym.spaces.Space):
        return torch_zeros_for_space_gym(state_space, device=device)
    
    elif isinstance(state_space, tuple):
        return torch.zeros(size=state_space, device=device)
    
    else:
        raise NotImplementedError(f"Unkown space type: {type(state_space)}")
    
