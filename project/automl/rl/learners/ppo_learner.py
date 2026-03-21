from automl.basic_components.dynamic_value import get_value_or_dynamic_value
from automl.component import Component, ParameterSignature, requires_input_proccess

from automl.core.advanced_input_management import ComponentParameterSignature
from automl.core.advanced_input_utils import get_value_of_type_or_component
from automl.loggers.logger_component import ComponentWithLogging
from automl.ml.memory.memory_utils import interpret_unit_values, interpret_values
from automl.ml.models.neural_model import FullyConnectedModelSchema
from automl.ml.models.torch_model_utils import split_shared_params
from automl.rl.learners.learner_component import LearnerSchema

from automl.rl.policy.stochastic_policy import StochasticPolicy
from automl.ml.optimizers.optimizer_components import AdamOptimizer, OptimizerSchema
from automl.ml.models.torch_model_components import TorchModelComponent
import torch

from automl.utils.class_util import get_class_from

SHOULD_INITIALIZE_NEW_CRITIC = False

class PPOLearner(LearnerSchema, ComponentWithLogging):
    
    '''
    Proximal Policy Optimization Learner
    '''

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
                               
                        "device" : ParameterSignature(ignore_at_serialization=True),
                        
                        "critic_model" : ComponentParameterSignature(
                            default_component_definition=(FullyConnectedModelSchema, {"hidden_layers" : 1, "hidden_size" : 64, "output_shape" : 1})    
                        ),

                        "critic_model_input" : ParameterSignature(mandatory=False, ignore_at_serialization=True),

                        "optimizer" : ComponentParameterSignature(
                            default_component_definition=(
                                AdamOptimizer,
                                {}
                            )
                            ),
                        "critic_optimizer" : ComponentParameterSignature(mandatory=False),
                        
                        "clip_epsilon" : ParameterSignature(default_value=0.2, description="The clip range",
                                                        custom_dict={"hyperparameter_suggestion" : [ "float", {"low": 0.1, "high": 0.3 }]}),
                        
                        "entropy_coef" : ParameterSignature(default_value=0.01, description="How much weight entropy has", 
                                                        custom_dict={"hyperparameter_suggestion" : [ "float", {"low": 0.0, "high": 0.3 }]}),
                        
                        "value_loss_coef" : ParameterSignature(default_value=0.5, description="The weight given to the critic value loss",
                                                           custom_dict={"hyperparameter_suggestion" : [ "float", {"low": 0.3, "high": 0.7 }]}),
                        
                        "lambda_gae" : ParameterSignature(default_value=0.95, description="Controls trade-off between bias and variance, higher means more variance and less bias",
                                                     custom_dict={"hyperparameter_suggestion" : [ "float", {"low": 0.9, "high": 0.999 }]}),

                        }    
    
    def _proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._proccess_input_internal()
                
        self.device = self.get_input_value("device")
                        
        self.policy : StochasticPolicy = self.agent.get_policy()
        
        if isinstance(self.policy, StochasticPolicy) is False:
            raise Exception("PPO Learner requires a Stochastic Policy, but got {}".format(get_class_from(self.policy)))
        
        self.custom_data_beyond_obs = [key for key in self.policy.input_state_shape.keys() if key != "observation"]
    
        if len(self.custom_data_beyond_obs) > 0:
            self.lg.writeLine(f"Noticed that policy also receives in its shape: {self.custom_data_beyond_obs}")

        self.model : TorchModelComponent = self.policy.model
        
        self.initialize_critic_model()
        self.initialize_optimizer()
        
        self.clip_epsilon = self.get_input_value("clip_epsilon")
        self.entropy_coef = get_value_of_type_or_component(self, "entropy_coef", float)
        self.value_loss_coef = self.get_input_value("value_loss_coef")
        self.lambda_gae = self.get_input_value("lambda_gae")
        
        self.number_of_times_optimized = 0

    @requires_input_proccess
    def critic_pred(self, observation):
        return self.critic.predict(observation)

    
    def initialize_critic_model(self):
        
        self.critic : TorchModelComponent = self.get_input_value("critic_model")

        if not self.critic.has_custom_name_passed():
            self.critic.pass_input({"name" : "critic"})
        
        critic_model_passed_input = self.get_input_value("critic_model_input")
        if critic_model_passed_input != None:
            self.critic.pass_input(critic_model_passed_input)

        self.critic.pass_input({"input_shape" : self.agent.processed_state_shape["observation"]})
        self.critic.pass_input({"output_shape" : 1})

        self.critic.proccess_input_if_not_processed()


    def _split_actor_critic_params(self):
        shared_params, actor_only, critic_only = split_shared_params(self.model, self.critic)

        return shared_params, actor_only, critic_only
        
        
    def initialize_optimizer(self):
        
        # Policy optimizer
        self.actor_optimizer : OptimizerSchema = self.get_input_value("optimizer")
        self.actor_optimizer.pass_input({"model" : self.model})
        
        if not self.actor_optimizer.has_custom_name_passed():
            self.actor_optimizer.pass_input({"name" : "ActorOptimizer"})

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
    
    def _evaluate_actions(self, interpreted_trajectory):
        """
        Evaluate given actions under the current policy.
        Computes log probabilities of actions and entropy of the policy distribution.
        """

        observation_batch = interpreted_trajectory["observation"]
        action_vals_batch = interpreted_trajectory["action_val"]

        state_batch = {"observation" : observation_batch}

        for custom_data_key in self.custom_data_beyond_obs:
            state_batch[custom_data_key] = interpreted_trajectory[custom_data_key]

        
        model_output = self.policy.predict_model_output(state_batch) #note we can call directly from the policy because we're using states as they were saved in the trajectory
        action_distribution = self.policy.distribution_from_model_output(model_output, state_batch)
                        
        log_probs = self.policy.log_probability_of_action_val(action_distribution, action_vals_batch, state_batch) # representing probabilities of taking previous actions with current policy

        entropy = action_distribution.entropy()
        if entropy.dim() > 1:
            entropy = entropy.sum(dim=-1)
        
        entropy = entropy.mean()        

        interpreted_trajectory["model_output"] = model_output
        interpreted_trajectory["new_log_probs"] = log_probs
        interpreted_trajectory["entropy"] = entropy
        


    def interpret_trajectory(self, trajectory):
        
        interpreted_trajectory = super().interpret_trajectory(trajectory)

        interpreted_trajectory["log_prob"] = interpret_unit_values(trajectory["log_prob"], self.device).detach()

        interpreted_trajectory["critic_pred"] = interpret_unit_values(trajectory["critic_pred"], self.device).detach()

        interpreted_trajectory["action_val"] = interpret_values(trajectory["action_val"], self.device).detach()

        for key in self.custom_data_beyond_obs:
            interpreted_trajectory[key] = interpret_values(trajectory[key], self.device).detach()

        return interpreted_trajectory    


    def compute_values_estimates(self, interpreted_trajectory):

        '''Computes values estimates using the critic'''

        observation_batch = interpreted_trajectory["observation"]
        next_observation_batch = interpreted_trajectory["next_observation"]
        done_batch = interpreted_trajectory["done"]

        old_values = self.critic.predict(observation_batch).squeeze(-1)

        with torch.no_grad():
            next_values = self.critic.predict(next_observation_batch).squeeze(-1)

        next_values = next_values * (1 - done_batch)

        return old_values, next_values
    
    
    def compute_error_and_advantage(self, discount_factor, interpreted_trajectory):

        reward_batch = interpreted_trajectory["reward"]
        next_values = interpreted_trajectory["values"]
        values = interpreted_trajectory["old_values"]
        done_batch = interpreted_trajectory["done"]

        # Compute advantages using Generalized Advantage Estimation (GAE)
        values_error = reward_batch + discount_factor * next_values - values 
        non_normalized_advantages = torch.zeros_like(values_error, device=self.device)
        
        # GAE computation in reverse, note this assumes seq data
        running_advantage = 0
        for t in reversed(range(len(values_error))):
            running_advantage = values_error[t] + discount_factor * self.lambda_gae * running_advantage  * (1 - done_batch[t])
            non_normalized_advantages[t] = running_advantage

        returns = non_normalized_advantages + values.detach()

        advantages = (non_normalized_advantages - non_normalized_advantages.mean()) / (non_normalized_advantages.std() + 1e-8)

        return values_error, non_normalized_advantages, advantages, returns
    
    def _compute_policy_loss(self, new_log_probs, log_prob_batch, advantages, entropy):
    
        # Compute ratio (pi_theta / pi_theta_old)
        log_ratio = new_log_probs - log_prob_batch
        log_ratio = torch.clamp(log_ratio, -20, 20)
        ratio = torch.exp(log_ratio)

        # Compute surrogate loss
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        
        policy_loss_batch = -torch.min(surrogate1, surrogate2)
        mean_policy_loss = policy_loss_batch.mean()
        policy_loss = mean_policy_loss - get_value_or_dynamic_value(self.entropy_coef) * entropy # entropy regulates the exploration

        return ratio, surrogate1, surrogate2, policy_loss_batch, mean_policy_loss, policy_loss


    def _compute_critic_loss(self, values, returns, old_values):

        value_loss_unclipped = (values - returns).pow(2)

        values_clipped = old_values + torch.clamp(
            values - old_values,
            -self.clip_epsilon,
            self.clip_epsilon
        )

        value_loss_clipped = (values_clipped - returns).pow(2)

        value_loss_batch = torch.max(
            value_loss_unclipped,
            value_loss_clipped
        )

        value_loss_mean = value_loss_batch.mean()

        value_loss = value_loss_mean * self.value_loss_coef 

        return value_loss_unclipped, value_loss_clipped, value_loss_batch, value_loss_mean, value_loss


    def _compute_losses(self, interpreted_trajectory):

        new_log_probs = interpreted_trajectory["new_log_probs"]
        entropy = interpreted_trajectory["entropy"]
        log_prob_batch = interpreted_trajectory["log_prob"]
        
        advantages = interpreted_trajectory["advantages"]
        old_values = interpreted_trajectory["old_values"]
        values = interpreted_trajectory["values"]
        returns = interpreted_trajectory["returns"]


        ratio, surrogate1, surrogate2, policy_loss_batch, mean_policy_loss, policy_loss = self._compute_policy_loss(new_log_probs, log_prob_batch, advantages, entropy)

        value_loss_unclipped, value_loss_clipped, value_loss_batch, value_loss_mean, value_loss = self._compute_critic_loss(values, returns, old_values)

        # Total loss
        if self.critic_optimizer is None:
            loss : torch.Tensor = policy_loss + value_loss

        else:
            loss = None
        
        return ratio, policy_loss, value_loss, loss
    

    
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


    
    def _learn(self, trajectory : dict, discount_factor):

        super()._learn(trajectory, discount_factor)
        
        self.number_of_times_optimized += 1
        
        interpreted_trajectory = self.interpret_trajectory(trajectory)

        # NOTE: PPO REQUIRES VALUES AND ADVANTAGES TO BE PRE COMPUTED

        self._evaluate_actions(interpreted_trajectory)

        ratio, policy_loss, value_loss, loss = self._compute_losses(interpreted_trajectory)

        self._optimize_using_loss(policy_loss, value_loss, loss)
        

        return {"log_prob" : interpreted_trajectory["log_prob"], "new_log_probs" : interpreted_trajectory["new_log_probs"]}


