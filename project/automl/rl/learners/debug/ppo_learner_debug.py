import torch
from automl.rl.learners.ppo_learner import PPOLearner
from automl.rl.learners.debug.learner_debug import LearnerDebug
from automl.core.input_management import InputSignature
from automl.ml.models.torch_model_components import TorchModelComponent

import torch.nn.functional as F

from automl.basic_components.dynamic_value import get_value_or_dynamic_value


class PPOLearnerDebug(LearnerDebug, PPOLearner):

    is_debug_schema = True

    parameters_signature = {
        "interval_between_debug_writes": InputSignature(default_value=15),
        "compare_old_and_new_critic_predictions_interval": InputSignature(default_value=15),
    }

    def _proccess_input_internal(self):
        super()._proccess_input_internal()

        self.interval_between_debug_writes = self.get_input_value("interval_between_debug_writes")

        self.compare_old_and_new_critic_predictions_interval = self.get_input_value("compare_old_and_new_critic_predictions_interval")

        self.__current_learning_step = 0
        self.__last_interval_checked_ppo_learning = -1

        self.lg.open_or_create_relative_folder("_ppo_debug")

        self.__compare_old_and_new_critic_predictions = self.compare_old_and_new_critic_predictions_interval >= 1

        if self.__compare_old_and_new_critic_predictions:
            self.__old_critic_model: TorchModelComponent = self.critic.clone(input_for_clone={"base_directory" : self, "artifact_relative_directory" : "__temp_comp_critic", "create_new_directory" : False}, is_deep_clone=True)


    def _split_actor_critic_params(self):
        shared_params, actor_only, critic_only = super()._split_actor_critic_params()

        self.lg.writeLine(f"Split actor and critic params: Shared: {len(shared_params)}, Actor only: {len(actor_only)}, Critic only: {len(critic_only)}")

        return shared_params, actor_only, critic_only


    def _should_log(self):

        # if we stil have not checked in the current interval
        if self.__last_interval_checked_ppo_learning < self.__current_learning_step:
    
            if not self.__current_learning_step % self.interval_between_debug_writes == 0:
                self.__debug_path = None
            
            else:
                self.__debug_path = self.lg.new_relative_path_if_exists(
                    "ppo_step.txt", dir="_ppo_debug"
                )

            self.__last_interval_checked_ppo_learning = self.__current_learning_step

        
        return self.__debug_path is not None
        

            

    
    def compute_values_estimates(self, state_batch, action_batch, next_state_batch, done_batch):
        values, next_values = super().compute_values_estimates(state_batch, action_batch, next_state_batch, done_batch)

        if self._should_log():

            self.lg.writeLine(
                    "\nCritic for state, action, Critic for next state, done batch",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )
            
            for i in range(len(values)):
                
                self.lg.writeLine(
                    f"{i}: state_value {values[i]} and action {action_batch[i]} -> next state value {next_values[i]}, done: {done_batch[i]}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )

        return values, next_values
    
    def compute_error_and_advantage(self, discount_factor, reward_batch, next_values, values, done_batch):
        values_error, non_normalized_advantages, advantages, returns = super().compute_error_and_advantage(
            discount_factor, reward_batch, next_values, values, done_batch
        )

        if self._should_log():

            self.lg.writeLine(f"\nValue error calculation: (reward + discount_factor * next_values - values)",
                file=self.__debug_path,
                use_time_stamp=False,
            )

            for i in range(len(values)):
                
                self.lg.writeLine(
                    f"{i}: {values_error[i]} = {reward_batch[i]} + {discount_factor} * {next_values[i]} - {values[i]}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )

            self.lg.writeLine(f"\nAdvantage calculation:  values_error + ( discount_factor * self.lamda * prev_advantage if not done) -> normalized value",
                file=self.__debug_path,
                use_time_stamp=False,
            )

            prev_advantage = 0

            for i in reversed(range(len(non_normalized_advantages))):
                
                self.lg.writeLine(
                    f"{i}: {non_normalized_advantages[i]} = {values_error[i]} + {discount_factor} * {self.lambda_gae} * {prev_advantage} * (1 - {done_batch[i]}) -> {advantages[i]}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )

                prev_advantage = non_normalized_advantages[i]

            self.lg.writeLine(
                "\nAdvantage statistics:",
                file=self.__debug_path,
                use_time_stamp=False,
            )

            self.lg.writeLine(
                f"mean={advantages.mean():.6f}, std={advantages.std():.6f}, "
                f"min={advantages.min():.6f}, max={advantages.max():.6f}",
                file=self.__debug_path,
                use_time_stamp=False,
            )

            for i in range(len(values)):
                
                self.lg.writeLine(
                    f"{i}: {values_error[i]} = {reward_batch[i]} + {discount_factor} * {next_values[i]} - {values[i]}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )

            self.lg.writeLine(f"\nReturns (for critic) = values critic predicted + advantages",
                file=self.__debug_path,
                use_time_stamp=False,
            )

            for i in range(len(values)):
                
                self.lg.writeLine(
                    f"{i}: {returns[i]} = {values[i]} + {non_normalized_advantages[i]}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )

        return values_error, non_normalized_advantages, advantages, returns
    
    def _evaluate_actions(self, states, actions):
        action_logits, new_log_probs, entropy = super()._evaluate_actions(states, actions)

        if self._should_log():

            self.lg.writeLine(
                "\nEvaluation of actions:\npolicy_predicted_action_logits selected by action -> log_probs of chosen actions",
                file=self.__debug_path,
                use_time_stamp=False,
            )

            for i in range(len(action_logits)):

                self.lg.writeLine(
                    f"{action_logits[i]} ({actions[i]}) -> {new_log_probs[i]}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )


        return action_logits, new_log_probs, entropy
    

    def _compute_policy_loss(self, new_log_probs, log_prob_batch, advantages, entropy):
    
        ratio, surrogate1, surrogate2, policy_loss_batch, mean_policy_loss, policy_loss = super()._compute_policy_loss(new_log_probs, log_prob_batch, advantages, entropy)
    
        if self._should_log():

            self.lg.writeLine(f"\nRatio calculation: exp(new_log_probs - log_prob_batch)", file=self.__debug_path, use_time_stamp=False)

            for i in range(len(ratio)):
                self.lg.writeLine(
                    f"    {i}: {ratio[i]} = exp({new_log_probs[i]} - {log_prob_batch[i]})",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )

            self.lg.writeLine(f"\nPolicy loss calculation:\nMin(Ratio * Advantages, Clipped values)", file=self.__debug_path, use_time_stamp=False)

            for i in range(len(ratio)):
                self.lg.writeLine(
                    f"    {i}: Min({ratio[i]} * {advantages[i]} = {surrogate1[i]}, {surrogate2[i]}) -> {policy_loss_batch[i]}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )

            self.lg.writeLine(f"\nMean policy loss: {mean_policy_loss}, After entropy (entropy {entropy} * coef {get_value_or_dynamic_value(self.entropy_coef)}): {policy_loss}\n", file=self.__debug_path, use_time_stamp=False)

        return ratio, surrogate1, surrogate2, policy_loss_batch, mean_policy_loss, policy_loss

    def _compute_critic_loss(self, values, returns, old_values):

        value_loss_unclipped, value_loss_clipped, value_loss_batch, value_loss_mean, value_loss = super()._compute_critic_loss(values, returns, old_values)
        
        if self._should_log():
        
            self.lg.writeLine(f"\nCritic loss calculation:\nMin(error (critic_predicted, return) , Clipped values)", file=self.__debug_path, use_time_stamp=False)

            for i in range(len(value_loss_unclipped)):
                self.lg.writeLine(
                    f"    {i}: Min( ( {values[i]} - {returns[i]} )^2 = {value_loss_unclipped[i]}, {value_loss_clipped[i]}) -> {value_loss_batch[i]}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )
        
            self.lg.writeLine(f"\nMean value loss: {value_loss_mean}, After coef ({self.value_loss_coef}): {value_loss}\n", file=self.__debug_path, use_time_stamp=False)

        return value_loss_unclipped, value_loss_clipped, value_loss_batch, value_loss_mean, value_loss


    def _learn(self, trajectory, discount_factor):


        state_batch, action_batch, next_state_batch, reward_batch, done_batch, log_prob_batch, critic_pred_batch = self.interpret_trajectory(trajectory)

        if self._should_log():
            

            self.lg.writeLine(f"\nDoing learning with state batch {state_batch.shape}, action_batch {action_batch.shape}, next_state_batch {next_state_batch.shape}, reward_batch {reward_batch.shape}, done_batch {done_batch.shape}, log_prob_batch {log_prob_batch.shape}, critic_pred_batch {critic_pred_batch.shape}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )

        should_compare_old_and_new_critic = self.__compare_old_and_new_critic_predictions and self.__current_learning_step % self.compare_old_and_new_critic_predictions_interval == 0

        if should_compare_old_and_new_critic:
            self.__old_critic_model.clone_other_model_into_this(self.critic)

        super()._learn(trajectory, discount_factor)
            
        if should_compare_old_and_new_critic:

            with torch.no_grad():

                old_values = self.__old_critic_model.predict(state_batch).squeeze(-1)
                new_values = self.critic.predict(state_batch).squeeze(-1)

                self.lg.writeLine(
                    "\naction, reward, done, old_model_predictions, new_model_predictions\n",
                    file="batch_comparison_critic.txt",
                    use_time_stamp=False,
                )

                for i in range(len(state_batch)):

                    action_val = action_batch[i].detach().cpu().numpy()
                    reward_val = reward_batch[i].detach().cpu().numpy()
                    done_val = done_batch[i].detach().cpu().numpy()
                    old_model_prediction_val = old_values[i].detach().cpu().numpy()
                    new_model_precitions_val = new_values[i].detach().cpu().numpy()

                    self.lg.writeLine(
                        f"    {i}: {action_val}, {reward_val}, {done_val}, {old_model_prediction_val}, {new_model_precitions_val}",
                        file="batch_comparison_critic.txt",
                        use_time_stamp=False,
                    )

        self.__current_learning_step += 1