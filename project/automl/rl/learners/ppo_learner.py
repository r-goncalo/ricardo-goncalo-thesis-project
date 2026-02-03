from automl.component import Component, InputSignature, requires_input_proccess

from automl.core.advanced_input_management import ComponentInputSignature
from automl.loggers.logger_component import ComponentWithLogging
from automl.ml.models.neural_model import FullyConnectedModelSchema
from automl.rl.learners.learner_component import LearnerSchema

from automl.rl.policy.stochastic_policy import StochasticPolicy
from automl.ml.optimizers.optimizer_components import AdamOptimizer, OptimizerSchema
from automl.ml.models.torch_model_components import TorchModelComponent
import torch

from automl.utils.class_util import get_class_from

import torch.nn.functional as F

SHOULD_INITIALIZE_NEW_CRITIC = False

class PPOLearner(LearnerSchema, ComponentWithLogging):
    
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


    def _split_actor_critic_params(self):
        actor_params = list(self.model.get_model_params())
        critic_params = list(self.critic.get_model_params())

        actor_ids = set(id(p) for p in actor_params)
        critic_ids = set(id(p) for p in critic_params)

        shared_ids = actor_ids & critic_ids

        shared_params = [p for p in actor_params if id(p) in shared_ids]
        actor_only = [p for p in actor_params if id(p) not in shared_ids]
        critic_only = [p for p in critic_params if id(p) not in shared_ids]

        return shared_params, actor_only, critic_only
        
        
    def initialize_optimizer(self):
        
        # Policy optimizer
        self.actor_optimizer : OptimizerSchema = self.get_input_value("optimizer")
        self.actor_optimizer.pass_input({"model" : self.model})
        
        if not self.actor_optimizer.has_custom_name_passed():
            self.actor_optimizer.pass_input({"ActorOptimizer"})

        # Critic optimizer
        self.critic_optimizer : OptimizerSchema  = self.get_input_value("critic_optimizer")

        if self.critic_optimizer is None:

            if SHOULD_INITIALIZE_NEW_CRITIC:
                self.critic_optimizer = self.actor_optimizer.clone()
                self.critic_optimizer.pass_input({"name" : "CriticOptimizer"})
                self.critic_optimizer.pass_input({"model" : self.critic})
            
            else:
                self.lg.writeLine(f"No critic optimizer passed, will use the same optimizer for both policy and critic")
                shared_params, actor_only, critic_only = self._split_actor_critic_params()
                all_params = actor_only + critic_only + shared_params
                self.actor_optimizer.set_params(all_params)

        else: 

            self.lg.writeLine(f"Critic optimizer was passed for learner")

            if not self.critic_optimizer.has_custom_name_passed():
                self.critic_optimizer.pass_input({"name" : "CriticOptimizer"})
        
            self.critic_optimizer.pass_input({"model" : self.critic})
    
    # EXPOSED METHODS --------------------------------------------------------------------------
    
    def _evaluate_actions(self, states, actions):
        """
        Evaluate given actions under the current policy.
        Computes log probabilities of actions and entropy of the policy distribution.
        """
        
        action_logits = self.policy.predict_logits(states) #note we can call directly from the policy because we're using states as they were saved in the trajectory
        action_distribution = torch.distributions.Categorical(logits=action_logits)
                
        log_probs = action_distribution.log_prob(actions.squeeze(-1))
        entropy = action_distribution.entropy().mean()
        return action_logits, log_probs, entropy



    def _interpret_trajectory(self, trajectory):
        
        state_batch, action_batch, next_state_batch, reward_batch, done_batch = super()._interpret_trajectory(trajectory)
        
        if not isinstance(trajectory.log_prob, torch.Tensor):
            log_prob_batch = torch.stack(trajectory.log_prob, dim=0).to(self.device)  # Stack tensors along a new dimension (dimension 0)
        
        else:
            log_prob_batch = trajectory.log_prob.view(-1).to(self.device)
        
            
        return state_batch, action_batch, next_state_batch, reward_batch, done_batch, log_prob_batch
    


    def _compute_values_estimates(self, state_batch, action_batch, next_state_batch, done_batch):

        '''Computes values estimates using the critic'''

        values = self.critic.predict(state_batch).squeeze(-1)

        with torch.no_grad():
            next_values = self.critic.predict(next_state_batch).squeeze(-1)

        next_values = next_values * (1 - done_batch)

        return values, next_values
    
    
    def _compute_error_and_advantage(self, discount_factor, reward_batch, next_values, values):

        # Compute advantages using Generalized Advantage Estimation (GAE)
        values_error = reward_batch + discount_factor * next_values - values 
        advantages = torch.zeros_like(values_error, device=self.device)
        
        # GAE computation in reverse
        running_advantage = 0
        for t in reversed(range(len(values_error))):
            running_advantage = values_error[t] + discount_factor * self.lamda_gae * running_advantage
            advantages[t] = running_advantage
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # this is the correction of the values computed by the critic using the advantage
        returns = advantages + values.detach()

        return values_error, advantages, returns
    

    
    def _compute_losses(self, new_log_probs, entropy, log_prob_batch, advantages, values, returns):

        # Compute ratio (pi_theta / pi_theta_old)
        ratio = torch.exp(new_log_probs - log_prob_batch)

        # Compute surrogate loss
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        
        policy_loss = -torch.min(surrogate1, surrogate2).mean()
        policy_loss = policy_loss - self.entropy_coef * entropy.mean() # entropy regulates the exploration

        # Compute value loss for critic
        value_loss = F.mse_loss(values, returns) * self.value_loss_coef 
        value_loss = value_loss * self.value_loss_coef 

        # Total loss
        if self.critic_optimizer is None:
            loss : torch.Tensor = policy_loss + value_loss

        else:
            loss = None
        

        return ratio, surrogate1, surrogate2, policy_loss, value_loss, loss
    

    
    def _optimize_using_loss(self, policy_loss, value_loss, loss):

        if self.critic_optimizer is not None: # if we are to optimize the critic and policy optimizer separatly

            self.actor_optimizer.clear_optimizer_gradients()
            self.critic_optimizer.clear_optimizer_gradients()

            policy_loss.backward(retain_graph=True)
            value_loss.backward() # will use and then free the graph

            self.actor_optimizer.optimize_with_backward_pass_done()
            self.critic_optimizer.optimize_with_backward_pass_done()

        else:

            self.actor_optimizer.clear_optimizer_gradients()
            loss.backward()
            self.actor_optimizer.optimize_with_backward_pass_done()


    
    def _learn(self, trajectory, discount_factor):

        super()._learn(trajectory, discount_factor)
        
        self.number_of_times_optimized += 1
        
        state_batch, action_batch, next_state_batch, reward_batch, done_batch, log_prob_batch = self._interpret_trajectory(trajectory)
                
        values, next_values = self._compute_values_estimates(state_batch, action_batch, next_state_batch, done_batch)
        
        values_error, advantages, returns = self._compute_error_and_advantage(discount_factor, reward_batch, next_values, values)

        # Compute new log probabilities from the policy
        action_logits, new_log_probs, entropy = self._evaluate_actions(state_batch, action_batch)

        ratio, surrogate1, surrogate2, policy_loss, value_loss, loss = self._compute_losses(new_log_probs, entropy, log_prob_batch, advantages, values, returns)

        self._optimize_using_loss(policy_loss, value_loss, loss)


