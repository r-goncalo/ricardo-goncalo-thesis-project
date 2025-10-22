import json
import numpy as np
import gymnasium as gym
from automl.utils.json_utils.custom_json_logic import CustomJsonLogic, register_custom_strategy


class CustomSpaceJsonEncoderDecoder(CustomJsonLogic):


    class CustomSpaceEncoder(json.JSONEncoder):
        
        '''Encodes elements of a component input or exposed value, which can be a component (defined by its localization) or a primitive type'''
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
                    
        def default(self, space):

            if isinstance(space, gym.spaces.Discrete):
                space_dictionary =  {
                    "space_type": "Discrete",
                    "n": int(space.n)
                }

            elif isinstance(space, gym.spaces.Box):
                space_dictionary =  {
                    "space_type": "Box",
                    "shape": space.shape,
                    "low": space.low.tolist() if isinstance(space.low, np.ndarray) else float(space.low),
                    "high": space.high.tolist() if isinstance(space.high, np.ndarray) else float(space.high),
                    "dtype": str(space.dtype)
                }

            elif isinstance(space, gym.spaces.MultiBinary):
                space_dictionary =  {
                    "space_type": "MultiBinary",
                    "n": int(space.n)
                }

            elif isinstance(space, gym.spaces.MultiDiscrete):
                space_dictionary =  {
                    "space_type": "MultiDiscrete",
                    "nvec": space.nvec.tolist()
                }

            else:
                return super().default(space) # this is for the case where there are elements which should be serialized using the common strategy, such as a dictionary of spaces


            return space_dictionary



    def to_dict(space):
        
        encoder = CustomSpaceJsonEncoderDecoder.CustomSpaceEncoder()
        
        return encoder.default(space) # we call explicitly our default method so we have full control over the logic of serialization



    def from_dict(dict, decode_elements_fun, source_component):

        """Reconstruct a Gym/Gymnasium space from its JSON dict."""

        print("Custom space json encoder decoder called from_dict")

        
        type = dict["space_type"]

        space_to_return = None

        if type == "Discrete":
            space_to_return =  gym.spaces.Discrete(dict["n"])
        
        elif type == "Box":
            space_to_return =  gym.spaces.Box(
                low=np.array(dict["low"], dtype=np.float32),
                high=np.array(dict["high"], dtype=np.float32),
                shape=tuple(dict["shape"]),
                dtype=np.dtype(dict["dtype"])
            )
        
        elif type == "MultiBinary":
            space_to_return =  gym.spaces.MultiBinary(dict["n"])
        
        elif type == "MultiDiscrete":
            space_to_return =  gym.spaces.MultiDiscrete(np.array(dict["nvec"]))
        
        else:
            raise ValueError(f"Unknown space type: {type}")
        
        return space_to_return
    

register_custom_strategy(gym.spaces.Space, CustomSpaceJsonEncoderDecoder)