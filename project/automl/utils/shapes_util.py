

from collections.abc import Iterable
import numpy as np
import gymnasium
import torch

import automl.utils.json_utils.shape_json_utils # this is just so the code runs and we're sure shapes are serialized / deserialized

from math import prod

# INPUT LAYER SIZE --------------------------------------------------------------------------

def discrete_input_layer_size_of_space_gym(state_space : gymnasium.spaces.Space) -> int:
    
    """
    Determines the number of input nodes needed for a given state space.

    :param state_space: A Gymnasium observation space
    :return: Integer representing the required input layer size
    """
    
    if isinstance(state_space, gymnasium.spaces.Box):
        return int(np.prod(state_space.shape))  # Flatten Box spaces (e.g., images, continuous vectors)
    
    elif isinstance(state_space, gymnasium.spaces.Discrete):
        return 1  # Discrete space is a single integer (can be used as an index or one-hot encoded)
    
    elif isinstance(state_space, gymnasium.spaces.MultiDiscrete):
        return len(state_space.nvec)  # Number of discrete dimensions
    
    elif isinstance(state_space, gymnasium.spaces.MultiBinary):
        return state_space.n  # Number of binary values

    elif isinstance(state_space, gymnasium.spaces.Tuple):
        return sum(discrete_input_layer_size_of_space_gym(s) for s in state_space.spaces)  # Sum of all subspaces
    
    elif isinstance(state_space, gymnasium.spaces.Dict):
        return sum(discrete_input_layer_size_of_space_gym(s) for s in state_space.spaces.values())  # Sum of all dictionary subspaces

    
    else:
        raise NotImplementedError(f"Unknown space type: {type(state_space)}")
    
    
    
def discrete_input_layer_size_of_space_torch(state_space : torch.Size) -> int:

    return prod(state_space)
    
    
    
def discrete_input_layer_size_of_space(state_space) -> int:
        
    if isinstance(state_space, torch.Size):
        return discrete_input_layer_size_of_space_torch(state_space)
    
    elif isinstance(state_space, gymnasium.spaces.Space):
        return discrete_input_layer_size_of_space_gym(state_space)
    
    elif isinstance(state_space, (tuple, list)):
        
        return prod([ discrete_input_layer_size_of_space(s) for s in state_space])
    
    elif isinstance(state_space, int):
        return state_space
    
    else:
        raise NotImplementedError(f"Unkown space type: {type(state_space)} with value: {state_space}")
    
    

# TORCH STATE SHAPE FROM SPACE ------------------------------------------------------

def torch_state_shape_from_space_gym(state_space : gymnasium.Space) -> torch.Size:
    
    if isinstance(state_space, gymnasium.spaces.Box):
        return torch.Size(state_space.shape)
    
    else:
        raise NotImplementedError(f"Unknown state space type: {type(state_space)}")
    
    
def torch_state_shape_from_space(state_space) -> torch.Size:

    if isinstance(state_space, gymnasium.Space):
        return torch_state_shape_from_space_gym(state_space)
    
    else:
        raise NotImplementedError(f"Unknown state space type: {type(state_space)}")

# OUTPUT LAYER SIZE ---------------------------------------------------------------------

def discrete_output_layer_size_of_space_gym(action_space : gymnasium.Space):
    
    """
    Determines the number of output nodes needed for a given action gym space.

    :param action_space: A Gymnasium action space
    :return: Integer representing the required output layer size
    """
    
    if isinstance(action_space, gymnasium.spaces.Discrete):
        return action_space.n  # Number of discrete actions (one-hot encoded output)
    
    elif isinstance(action_space, gymnasium.spaces.Box):
        return int(np.prod(action_space.shape))  # Continuous action space (vector output)
    
    elif isinstance(action_space, gymnasium.spaces.MultiDiscrete):
        return len(action_space.nvec)  # Number of discrete action dimensions
    
    elif isinstance(action_space, gymnasium.spaces.MultiBinary):
        return action_space.n  # Number of binary actions
    
    elif isinstance(action_space, gymnasium.spaces.Tuple):
        return sum(discrete_output_layer_size_of_space(s) for s in action_space.spaces)  # Sum of all subspaces
    
    elif isinstance(action_space, gymnasium.spaces.Dict):
        return sum(discrete_output_layer_size_of_space(s) for s in action_space.spaces.values())  # Sum of all dictionary subspaces
    
    else:
        raise NotImplementedError(f"Unknown gym action space type: {type(action_space)} with value {action_space}")
    
    
def discrete_output_layer_size_of_space(action_space):
    
    """
    Determines the number of output nodes needed for a given action space.
    """
    
    if isinstance(action_space, gymnasium.spaces.Space):
        return discrete_output_layer_size_of_space_gym(action_space)
    
    elif isinstance(action_space, int):
        return action_space
    
    else:
        raise NotImplementedError(f"Unknown action space type: {type(action_space)} with value {action_space}")
    
    
# ACTION SHAPE SIZE ---------------------------------------------------------------------

def single_action_shape_gym(action_space : gymnasium.Space):
    
    """
    Determines the shape needed to encode single actions
    """
    
    if isinstance(action_space, gymnasium.spaces.Discrete):
        return 1  # Number of discrete actions (one-hot encoded output)
    
    else:
        raise NotImplementedError(f"Unknown action space type: {type(action_space)}")

    
def single_action_shape(action_space):
    
    """
    Determines the shape needed to encode single actions
    """
    
    if isinstance(action_space, gymnasium.spaces.Space):
        return single_action_shape_gym(action_space)
    
    elif isinstance(action_space, int):
        return 1
    
    else:
        raise NotImplementedError(f"Unknown action space type: {type(action_space)}")

# ALLOCATING MEMORY FOR SPACE ------------------------------------------------------------------

def torch_zeros_for_space_gym(space, device):

    """
    Returns a zero tensor (or structure of tensors) matching the Gymnasium space.
    """
    if isinstance(space, gymnasium.spaces.Box):
        # Continuous observation/action space
        return torch.zeros(space.shape, dtype=torch.float32, device=device)

    elif isinstance(space, gymnasium.spaces.Discrete):
        # Single integer (scalar) state; often encoded as one-hot or int.
        # We’ll use a single scalar zero by default.
        return torch.zeros(1, dtype=torch.long, device=device)

    elif isinstance(space, gymnasium.spaces.MultiBinary):
        return torch.zeros(space.n, dtype=torch.float32, device=device)

    elif isinstance(space, gymnasium.spaces.MultiDiscrete):
        return torch.zeros(len(space.nvec), dtype=torch.long, device=device)

    elif isinstance(space, gymnasium.spaces.Dict):
        # Recursively allocate for each subspace
        return {k: torch_zeros_for_space_gym(v, device=device) for k, v in space.spaces.items()}

    elif isinstance(space, gymnasium.spaces.Tuple):
        # Recursively allocate for each element
        return tuple(torch_zeros_for_space_gym(s, device=device) for s in space.spaces)

    else:
        raise NotImplementedError(f"Unsupported space type: {type(space)}")
    
    
def torch_zeros_for_space_torch(space : torch.Size, device) -> int:
    
    return torch.zeros(size=space, device=device)
    
    
def torch_zeros_for_space(state_space, device) -> torch.Tensor:
    
    if isinstance(state_space, torch.Size):
        return torch_zeros_for_space_torch(state_space, device=device)
    
    elif isinstance(state_space, gymnasium.spaces.Space):
        return torch_zeros_for_space_gym(state_space, device=device)
    
    elif isinstance(state_space, tuple):
        return torch.zeros(size=state_space, device=device)
    
    else:
        raise NotImplementedError(f"Unkown space type: {type(state_space)}")
    


# GYM AND NOT GYMNASIUM??? GYMNASIUM IS NEWER, A FORK OF GYM ####################################

def gymnasium_to_gym_space(space):
    """Convert gymnasium or gym spaces into gym spaces (for SB3)."""
    import gym

    # If it's already a gym space, just return
    if isinstance(space, gym.spaces.Space):
        return space

    # If it's a gymnasium space, convert manually
    if isinstance(space, gymnasium.spaces.Box):
        return gym.spaces.Box(
            low=np.array(space.low, dtype=np.float32),
            high=np.array(space.high, dtype=np.float32),
            shape=space.shape,
            dtype=np.float32,
        )

    if isinstance(space, gymnasium.spaces.Discrete):
        return gym.spaces.Discrete(space.n)

    raise NotImplementedError(f"Unsupported space type: {space}")


def gym_to_gymnasium_space(space):
    """Convert gym spaces into gymnasium spaces."""
    import gym

    # If it's already a gym space, just return
    if isinstance(space, gymnasium.spaces.Space):
        return space

    # If it's a gymnasium space, convert manually
    if isinstance(space, gym.spaces.Box):
        return gymnasium.spaces.Box(
            low=np.array(space.low, dtype=np.float32),
            high=np.array(space.high, dtype=np.float32),
            shape=space.shape,
            dtype=np.float32,
        )

    if isinstance(space, gym.spaces.Discrete):
        return gymnasium.spaces.Discrete(space.n)

    raise NotImplementedError(f"Unsupported space type: {space}")


# -----------------------------------------------------------------------------------------------
# TORCH SHAPE FROM SPACE / SHAPE / TUPLE
# -----------------------------------------------------------------------------------------------

def torch_shape_from_space_gym(space: gymnasium.Space) -> torch.Size:
    """Convert a Gymnasium space into a torch.Size."""
    if isinstance(space, gymnasium.spaces.Box):
        return torch.Size(space.shape)

    elif isinstance(space, gymnasium.spaces.Discrete):
        return torch.Size([1])

    elif isinstance(space, gymnasium.spaces.MultiDiscrete):
        return torch.Size([len(space.nvec)])

    elif isinstance(space, gymnasium.spaces.MultiBinary):
        return torch.Size([space.n])

    elif isinstance(space, gymnasium.spaces.Tuple):
        # Concatenate all subspace shapes along the last dimension
        flat_shapes = [int(np.prod(torch_shape_from_space_gym(s))) for s in space.spaces]
        return torch.Size([sum(flat_shapes)])

    elif isinstance(space, gymnasium.spaces.Dict):
        # Concatenate all dict subspace shapes along the last dimension
        flat_shapes = [int(np.prod(torch_shape_from_space_gym(s))) for s in space.spaces.values()]
        return torch.Size([sum(flat_shapes)])

    else:
        raise NotImplementedError(f"Unsupported Gymnasium space type: {type(space)}")


def torch_shape_from_space(space_like) -> torch.Size:
    """Converts any space-like input into a torch.Size."""
    if isinstance(space_like, torch.Size):
        return space_like

    elif isinstance(space_like, gymnasium.Space):
        return torch_shape_from_space_gym(space_like)

    elif isinstance(space_like, tuple):
        # Normalize nested tuples (e.g., ((3,), (4,)) → (3, 4))
        flat = []
        for s in space_like:
            if isinstance(s, Iterable) and not isinstance(s, (str, bytes)):
                flat.extend(s)
            else:
                flat.append(s)
        return torch.Size(flat)

    elif isinstance(space_like, int):
        return torch.Size([space_like])

    else:
        raise NotImplementedError(f"Unknown space/shape type: {type(space_like)}")


# -----------------------------------------------------------------------------------------------
# TUPLE OF TORCH SHAPES FROM MULTIPLE SPACES
# -----------------------------------------------------------------------------------------------

def tuple_of_torch_shapes_from_spaces(spaces_like) -> tuple[torch.Size, ...]:
    """
    Converts a tuple/list/dict of Gym spaces or shapes into a tuple of torch.Size.
    """
    if isinstance(spaces_like, dict):
        return tuple(torch_shape_from_space(v) for v in spaces_like.values())

    elif isinstance(spaces_like, (list, tuple)):
        return tuple(torch_shape_from_space(s) for s in spaces_like)

    elif isinstance(spaces_like, gymnasium.Space) or isinstance(spaces_like, torch.Size):
        # Wrap a single space into a tuple
        return (torch_shape_from_space(spaces_like),)

    else:
        raise NotImplementedError(f"Unknown composite type: {type(spaces_like)}")