from automl.component import Component, InputSignature, requires_input_proccess

from automl.core.advanced_input_management import ComponentInputSignature
from automl.ml.models.neural_model import FullyConnectedModelSchema
from automl.rl.learners.learner_component import LearnerSchema

from automl.rl.policy.stochastic_policy import StochasticPolicy
from automl.ml.optimizers.optimizer_components import AdamOptimizer, OptimizerSchema
from automl.ml.models.torch_model_components import TorchModelComponent
import torch

from automl.rl.policy.policy import Policy
from automl.ml.models.model_components import ModelComponent

from automl.utils.class_util import get_class_from

import torch.nn.functional as F

import torch.optim as optim


class PPOLearner(LearnerSchema):
    
    '''
    Proximal Policy Optimization Learner
    '''

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
                               
                        "device" : InputSignature(ignore_at_serialization=True),
                        
                        "critic_model" : ComponentInputSignature(
                            default_component_definition=(FullyConnectedModelSchema, {"hidden_layers" : 1, "hidden_size" : 64, "output_shape" : 1})    
                        ),

                        "critic_model_input" : InputSignature(mandatory=False, ignore_at_serialization=True),

                        "optimizer" : ComponentInputSignature(
                            default_component_definition=(
                                AdamOptimizer,
                                {}
                            )
                            ),
                        "critic_optimizer" : ComponentInputSignature(mandatory=False),
                        
                        "clip_epsilon" : InputSignature(default_value=0.2, description="The clip range"),
                        "entropy_coef" : InputSignature(default_value=0.01, description="How much weight entropy has"),
                        "value_loss_coef" : InputSignature(default_value=0.5, description="The weight given to the critic value loss"),
                        "lamda_gae" : InputSignature(default_value=0.95, description="Controls trade-off between bias and variance, higher means more variance and less bias"),
                        
                        "critic_learning_rate" : InputSignature(default_value=3e-4),
                        "model_learning_rate" : InputSignature(default_value=3e-4)

                        }    
    
    def _proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._proccess_input_internal()
                
        self.device = self.get_input_value("device")
                        
        self.policy : StochasticPolicy = self.agent.get_policy()
        
        if isinstance(self.policy, StochasticPolicy) is False:
            raise Exception("PPO Learner requires a Stochastic Policy, but got {}".format(get_class_from(self.policy)))
        
        self.model : TorchModelComponent = self.policy.model
        
        self.initialize_critic_model()
        self.initialize_optimizer()
        
        self.clip_epsilon = self.get_input_value("clip_epsilon")
        self.entropy_coef = self.get_input_value("entropy_coef")
        self.value_loss_coef = self.get_input_value("value_loss_coef")
        self.lamda_gae = self.get_input_value("lamda_gae")
        
        self.number_of_times_optimized = 0

        

    
    def initialize_critic_model(self):
        
        self.critic : TorchModelComponent = self.get_input_value("critic_model")

        if not self.critic.has_custom_name_passed():
            self.critic.pass_input({"name" : "critic"})
        
        critic_model_passed_input = self.get_input_value("critic_model_input")
        if critic_model_passed_input != None:
            self.critic.pass_input(critic_model_passed_input)

        self.critic.pass_input({"input_shape" : self.agent.model_input_shape})

        self.critic.proccess_input_if_not_proccesd()
        
        
    def initialize_optimizer(self):
        
        # Policy optimizer
        self.actor_optimizer : OptimizerSchema = self.get_input_value("optimizer")
        
        if not self.actor_optimizer.has_custom_name_passed():
            self.actor_optimizer.pass_input({"ActorOptimizer"})

        # Critic optimizer
        self.critic_optimizer : OptimizerSchema  = self.get_input_value("critic_optimizer")

        if self.critic_optimizer == None:
            self.critic_optimizer = self.actor_optimizer.clone()
            self.critic_optimizer.pass_input({"name" : "CriticOptimizer"})

        elif not self.critic_optimizer.has_custom_name_passed():
            self.critic_optimizer.pass_input({"name" : "CriticOptimizer"})


        self.actor_optimizer.pass_input({"model" : self.model})
        self.critic_optimizer.pass_input({"model" : self.critic})
    
    # EXPOSED METHODS --------------------------------------------------------------------------
    
    def _evaluate_actions(self, states, actions):
        """
        Evaluate given actions under the current policy.
        Computes log probabilities of actions and entropy of the policy distribution.
        """
        
        action_logits = self.policy.predict_logits(states) #note we can call directly from the policy because we're using states as they were saved in the trajectory
        action_distribution = torch.distributions.Categorical(logits=action_logits)
                
        log_probs = action_distribution.log_prob(actions).sum(dim=-1)
        entropy = action_distribution.entropy().mean()
        return log_probs, entropy



    def _interpret_trajectory(self, trajectory):
        
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = super()._interpret_trajectory(trajectory)
        
        if not isinstance(trajectory.log_prob, torch.Tensor):
            log_prob_batch = torch.stack(trajectory.log_prob, dim=0)  # Stack tensors along a new dimension (dimension 0)
        
        else:
            log_prob_batch = trajectory.log_prob
        
            
        return state_batch, action_batch, next_state_batch, reward_batch, done_batch, log_prob_batch
    
    
    
    @requires_input_proccess
    def learn(self, trajectory, discount_factor):
        super().learn(trajectory, discount_factor)
        
        self.number_of_times_optimized += 1
        
        state_batch, action_batch, next_state_batch, reward_batch, done_batch, log_prob_batch = self._interpret_trajectory(trajectory)
                
                        
        # Compute value estimates
        values = self.critic.predict(state_batch).squeeze(-1)
        with torch.no_grad():
            next_values = self.critic.predict(next_state_batch).squeeze(-1)

        # Mask out terminal states (no bootstrapping after done)
        next_values = next_values * (1 - done_batch)
        
        
        # Compute advantages using Generalized Advantage Estimation (GAE)
        deltas = reward_batch + discount_factor * next_values - values 
        advantages = torch.zeros_like(deltas, device=self.device)
        
        
        # GAE computation in reverse
        running_advantage = 0
        for t in reversed(range(len(deltas))):
            running_advantage = deltas[t] + discount_factor * self.lamda_gae * running_advantage
            advantages[t] = running_advantage
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute new log probabilities from the policy
        new_log_probs, entropy = self._evaluate_actions(state_batch, action_batch)

        # Compute ratio (pi_theta / pi_theta_old)
        ratio = torch.exp(new_log_probs - log_prob_batch)

        # Compute surrogate loss
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surrogate1, surrogate2).mean()

        # Compute value loss
        values = self.critic.predict(state_batch).squeeze()
        value_loss = F.mse_loss(values, reward_batch)

        # Total loss
        loss : torch.Tensor = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

        self.actor_optimizer.clear_optimizer_gradients()
        self.critic_optimizer.clear_optimizer_gradients()

        loss.backward() # we do the optimization here so it goes to both optimizers



        self.actor_optimizer.optimize_with_backward_pass_done()
        self.critic_optimizer.optimize_with_backward_pass_done()
