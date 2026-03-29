
from automl.ml.models.torch_model_components import TorchModelComponent
from automl.basic_components.dynamic_value import get_value_or_dynamic_value
import torch



def compute_values_estimates(critic_model : TorchModelComponent, 
                             observation_batch, 
                             next_observation_batch, 
                             done_batch):

    '''
    Computes values estimates using the critic
    Note that this may differ from the critic values computed from the critic before learning starts, as learning may be called multiple times
    '''

    observation_critic_values = critic_model.predict(observation_batch).squeeze(-1)

    with torch.no_grad():
        next_obs_critic_values = critic_model.predict(next_observation_batch).squeeze(-1)

    next_obs_critic_values = next_obs_critic_values * (1 - done_batch)

    return observation_critic_values, next_obs_critic_values



def compute_critic_loss(observation_critic_values, returns, obs_old_critic_values, clip_epsilon, value_loss_coef):

    '''
    Computes the loss for the critic clipped 
    '''

    observation_critic_values = observation_critic_values.view(-1)
    returns = returns.view(-1)
    obs_old_critic_values = obs_old_critic_values.view(-1)

    value_loss_unclipped = (observation_critic_values - returns).pow(2)
    
    values_clipped = obs_old_critic_values + torch.clamp(
        observation_critic_values - obs_old_critic_values,
        -clip_epsilon,
        clip_epsilon
    )
    
    value_loss_clipped = (values_clipped - returns).pow(2)
    
    value_loss_batch = torch.max(value_loss_unclipped, value_loss_clipped)
    
    value_loss_mean = value_loss_batch.mean()
    value_loss = value_loss_mean * value_loss_coef

    return value_loss_unclipped, value_loss_clipped, value_loss_batch, value_loss_mean, value_loss


def evaluate_actions(policy, state_batch: dict, action_vals_batch):
    """
    Evaluate saved actions under the current policy.

    Args:
        policy: stochastic policy
        state_batch: already-prepared state dict for policy forward
        action_vals_batch: batch of saved action values

    Returns:
        model_output,
        new_log_probs,
        entropy
    """

    model_output = policy.predict_model_output(state_batch)
    action_distribution = policy.distribution_from_model_output(model_output, state_batch)

    new_log_probs = policy.log_probability_of_action_val(
        action_distribution,
        action_vals_batch,
        state_batch
    )

    entropy = action_distribution.entropy()
    if entropy.dim() > 1:
        entropy = entropy.sum(dim=-1)

    entropy = entropy.mean()

    return model_output, new_log_probs, entropy


def compute_gae_and_returns(
    reward_batch,
    observation_critic_values,
    next_obs_critic_values,
    done_batch,
    discount_factor: float,
    lambda_gae: float,
    eps: float = 1e-8,
):
    """
    Compute TD residuals, GAE advantages and returns.

    Returns:
        critic_obs_pred_error,
        non_normalized_advantages,
        advantages,
        returns
    """

    reward_batch = reward_batch.view(-1)
    observation_critic_values = observation_critic_values.view(-1)
    next_obs_critic_values = next_obs_critic_values.view(-1)
    done_batch = done_batch.view(-1)

    critic_obs_pred_error = (
        reward_batch
        + discount_factor * next_obs_critic_values
        - observation_critic_values
    )

    non_normalized_advantages = torch.zeros_like(
        critic_obs_pred_error,
        device=critic_obs_pred_error.device
    )

    running_advantage = 0
    for t in reversed(range(len(critic_obs_pred_error))):
        running_advantage = (
            critic_obs_pred_error[t]
            + discount_factor * lambda_gae * running_advantage * (1 - done_batch[t])
        )
        non_normalized_advantages[t] = running_advantage

    returns = non_normalized_advantages + observation_critic_values.detach()

    advantages = (
            (non_normalized_advantages - non_normalized_advantages.mean()) /
            (non_normalized_advantages.std() + eps)
    )

    return critic_obs_pred_error, non_normalized_advantages, advantages, returns




def compute_policy_loss(
    new_log_probs,
    old_log_probs,
    advantages,
    entropy,
    clip_epsilon: float,
    entropy_coef,
):
    """
    Compute PPO clipped actor loss.
    """

    log_ratio = new_log_probs - old_log_probs
    log_ratio = torch.clamp(log_ratio, -20, 20)
    ratio = torch.exp(log_ratio)

    surrogate1 = ratio * advantages
    surrogate2 = torch.clamp(
        ratio,
        1 - clip_epsilon,
        1 + clip_epsilon
    ) * advantages

    policy_loss_batch = -torch.min(surrogate1, surrogate2)
    mean_policy_loss = policy_loss_batch.mean()
    policy_loss = mean_policy_loss - get_value_or_dynamic_value(entropy_coef) * entropy

    return ratio, surrogate1, surrogate2, policy_loss_batch, mean_policy_loss, policy_loss