from automl.component import Component, InputSignature, requires_input_proccess

from automl.core.advanced_input_management import ComponentInputSignature
from automl.ml.models.neural_model import FullyConnectedModelSchema
from automl.ml.optimizers.optimizer_components import OptimizerSchema, AdamOptimizer
from automl.rl.learners.learner_component import LearnerSchema

from automl.rl.policy.stochastic_policy import StochasticPolicy
import torch

from automl.rl.policy.policy import Policy
from automl.ml.models.model_components import ModelComponent

from automl.utils.class_util import get_class_from

import torch.nn.functional as F


class PPOLearner(LearnerSchema):
    
    '''
    Proximal Policy Optimization Learner
    '''

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
                               
                        "device" : InputSignature(ignore_at_serialization=True),
                        
                        "critic_model" : ComponentInputSignature(
                            default_component_definition=(FullyConnectedModelSchema, {"hidden_layers" : 1, "hidden_size" : 64})    
                        ),
                        
                        "optimizer" : ComponentInputSignature(
                            default_component_definition=( AdamOptimizer, {} ) #the default optimizer is Adam with no specific input
                        ),
                        
                        "clip_epsilon" : InputSignature(default_value=0.2, description="The clip range"),
                        "entropy_coef" : InputSignature(default_value=0.01, description="How much weight entropy has"),
                        "value_loss_coef" : InputSignature(default_value=0.5, description="The weight given to the critic value loss"),
                        "lamda_gae" : InputSignature(default_value=0.95, description="Controls trade-off between bias and variance, higher means more variance and less bias")

                        }    
    
    def proccess_input(self): #this is the best method to have initialization done right after, input is already defined
        
        super().proccess_input()
        
        self.agent : Component = self.input["agent"]
        
        self.device = self.input["device"]
                        
        self.policy : StochasticPolicy = self.agent.get_policy()
        
        if isinstance(self.policy, StochasticPolicy) is False:
            raise Exception("PPO Learner requires a Stochastic Policy, but got {}".format(get_class_from(self.policy)))
        
        self.model : ModelComponent = self.policy.model
        
        self.initialize_critic_model()
        self.initialize_optimizer()
        
        self.clip_epsilon = self.input["clip_epsilon"]
        self.entropy_coef = self.input["entropy_coef"]
        self.value_loss_coef = self.input["value_loss_coef"]
        self.lamda_gae = self.input["lamda_gae"]

        
        
        
    def initialize_optimizer(self):
        self.optimizer : OptimizerSchema = ComponentInputSignature.get_component_from_input(self, "optimizer")
        self.optimizer.pass_input({"model" : self.model})

    
    def initialize_critic_model(self):
        
        self.critic : ModelComponent = ComponentInputSignature.get_component_from_input(self, "critic_model")
        
        
    
    # EXPOSED METHODS --------------------------------------------------------------------------
    
    def evaluate_actions(self, model, states, actions):
        """
        Evaluate given actions under the current policy.
        Computes log probabilities of actions and entropy of the policy distribution.
        """
        action_distribution = self.model.forward(states)
        log_probs = action_distribution.log_prob(actions).sum(dim=-1)
        entropy = action_distribution.entropy().mean()
        return log_probs, entropy
    
    
    
    @requires_input_proccess
    def learn(self, trajectory, discount_factor):
        super().learn(trajectory, discount_factor)

        states = torch.stack(trajectory.state).to(self.device)
        actions = torch.tensor(trajectory.action, device=self.device)
        rewards = torch.tensor(trajectory.reward, device=self.device)
        next_states = torch.stack([s for s in trajectory.next_state if s is not None]).to(self.device)
        dones = torch.tensor([s is None for s in trajectory.next_state], dtype=torch.float, device=self.device)
        
        old_log_probs = torch.tensor(trajectory.log_prob, device=self.device)
        
        # Compute value estimates
        values = self.critic.forward(states).squeeze()
        next_values = torch.zeros_like(values, device=self.device)
        next_values[~dones.bool()] = self.critic.forward(next_states).squeeze().detach()

        # Compute advantages using Generalized Advantage Estimation (GAE)
        deltas = rewards + discount_factor * next_values * (1 - dones) - values
        advantages = torch.zeros_like(deltas, device=self.device)
        advantage = 0
        for t in reversed(range(len(deltas))):
            advantage = deltas[t] + discount_factor * self.lambda_gae * (1 - dones[t]) * advantage
            advantages[t] = advantage
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)


        # Compute new log probabilities from the policy
        new_log_probs, entropy = self.evaluate_actions(self.model, states, actions)

        # Compute ratio (pi_theta / pi_theta_old)
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Compute surrogate loss
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # Compute value loss
        values = self.critic.forward(states).squeeze()
        value_loss = F.mse_loss(values, rewards)

        # Total loss
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

        # Optimize the model
        self.optimizer.optimize_model(loss)
