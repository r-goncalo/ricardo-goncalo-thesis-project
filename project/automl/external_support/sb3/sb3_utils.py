

from automl.utils.shapes_util import gym_to_gymnasium_space, gymnasium_to_gym_space
from huggingface_sb3 import load_from_hub
from stable_baselines3 import DQN
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.dqn.policies import DQNPolicy


def load_sb3_dqn_model(model_name : str):
        

        checkpoint = load_from_hub(
        	repo_id=f"sb3/{model_name}",
        	filename=f"{model_name}.zip",
        )

        model_sb3 = DQN.load(checkpoint)
        
        return model_sb3
    
    
def load_sb3_q_net(model_name: str):
    
    checkpoint = load_from_hub(
        repo_id=f"sb3/{model_name}",
        filename=f"{model_name}.zip",
    )

    # Load the state_dict and metadata without creating a full DQN object
    data, params, pytorch_variables = load_from_zip_file(checkpoint)

    print(f"Observation space in model: {data['observation_space']} of type {type(data['observation_space'])}")


    print(f"data: \n{data}")
    print(f"kwargs: \n{data['policy_kwargs']}")

    # Rebuild just the policy (with the right architecture)
    policy = DQNPolicy(
        observation_space=gym_to_gymnasium_space(data["observation_space"]),
        action_space=gym_to_gymnasium_space(data["action_space"]),
        lr_schedule=lambda _: 0.0,  # dummy,
        **data["policy_kwargs"]
    )

    # Load weights into the policy
    policy.load_state_dict(params["policy"], strict=True)

    return policy.q_net