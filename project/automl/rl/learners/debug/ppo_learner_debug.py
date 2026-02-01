import torch
from automl.rl.learners.ppo_learner import PPOLearner
from automl.rl.learners.debug.learner_debug import LearnerDebug
from automl.core.input_management import InputSignature
from automl.ml.models.torch_model_components import TorchModelComponent

import torch.nn.functional as F


class PPOLearnerDebug(LearnerDebug, PPOLearner):

    is_debug_schema = True

    parameters_signature = {
        "interval_between_debug_writes": InputSignature(default_value=10),
        "compare_old_and_new_critic_predictions": InputSignature(default_value=True),
        "log_advantage_stats": InputSignature(default_value=True),
        "log_ratio_and_clipping": InputSignature(default_value=True),
    }

    def _proccess_input_internal(self):
        super()._proccess_input_internal()

        self.interval_between_debug_writes = self.get_input_value("interval_between_debug_writes")

        self.compare_old_and_new_critic_predictions = self.get_input_value("compare_old_and_new_critic_predictions")

        self.__current_interval = 1

        self.lg.open_or_create_relative_folder("_ppo_debug")

        if self.compare_old_and_new_critic_predictions:
            self.__old_critic_model: TorchModelComponent = self.critic.clone(input_for_clone={"base_directory" : self, "artifact_relative_directory" : "__temp_comp_critic"})

    def _should_log(self):
        return self.__current_interval % self.interval_between_debug_writes == 0
    
    
    def _compute_values_estimates(self, state_batch, action_batch, next_state_batch, done_batch):
        values, next_values = super()._compute_values_estimates(state_batch, action_batch, next_state_batch, done_batch)

        if self.__debug_path is not None:

            self.lg.writeLine(
                    "\nComputed values for state, action next state, with done batch",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )
            
            for i in range(len(values)):
                
                self.lg.writeLine(
                    f"{i}: state_value {values[i]} + action {action_batch[i]} -> next state value {next_state_batch[i]}, done: {done_batch[i]}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )

        return values, next_values
    
    def _compute_error_and_advantage(self, discount_factor, reward_batch, next_values, values):
        values_error, advantages, returns = super()._compute_error_and_advantage(
            discount_factor, reward_batch, next_values, values
        )

        if self.__debug_path is not None:
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

        return values_error, advantages, returns
    
    def _evaluate_actions(self, states, actions):
        action_logits, new_log_probs, entropy = super()._evaluate_actions(states, actions)

        if self.__debug_path is not None:

            self.lg.writeLine(
                "\nEvaluation of actions:\npolicy_predicted_action_logits -> log_probs",
                file=self.__debug_path,
                use_time_stamp=False,
            )

            for i in range(len(action_logits)):

                self.lg.writeLine(
                    f"{action_logits[i]} -> {new_log_probs[i]}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )


        return action_logits, new_log_probs, entropy
    
    def _compute_losses_debug(self, new_log_probs, entropy, log_prob_batch, advantages, values, returns):

        # Compute ratio (pi_theta / pi_theta_old)
        ratio = torch.exp(new_log_probs - log_prob_batch)

        # Compute surrogate loss
        surrogate1 = ratio * advantages
        surrogate2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_batch_loss_debug = -torch.min(surrogate1, surrogate2)
        policy_loss = policy_batch_loss_debug.mean()

        # Compute value loss for critic
        value_loss = F.mse_loss(values, returns)

        # Total loss
        loss : torch.Tensor = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

        return ratio, surrogate1, surrogate2, policy_loss, value_loss, loss, policy_batch_loss_debug
    


    def _compute_losses(self, new_log_probs, entropy, log_prob_batch, advantages, values, returns):
        
        ratio, surrogate1, surrogate2, policy_loss, value_loss, loss, policy_batch_loss_debug =  self._compute_losses_debug(  #super()._compute_losses(
            new_log_probs, entropy, log_prob_batch, advantages, values, returns
        )

        if self.__debug_path is not None:

            self.lg.writeLine(f"\nPolicy loss calculation:\nMin(Ratio * Advantages, Clipped values)", file=self.__debug_path, use_time_stamp=False)

            for i in range(len(ratio)):
                self.lg.writeLine(
                    f"{i}: Min({ratio[i]} * {advantages[i]} = {surrogate1[i]}, {surrogate2[i]}) -> {policy_batch_loss_debug[i]}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )


            self.lg.writeLine(f"\nValue loss calculation:\nadvantage + critic values = returns vs critic values", file=self.__debug_path, use_time_stamp=False)

            for i in range(len(ratio)):
                self.lg.writeLine(
                    f"{i}: {advantages[i]} + {values[i]} = {returns[i]} vs {values[i]}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )

            self.lg.writeLine(
                f"\nPolicy_loss: {policy_loss}, value (critic) loss: {value_loss}, Total loss: {loss}\nRatio, surrogate loss, clipped surrogate loss, policy_loss, value loss, loss",
                file=self.__debug_path,
                use_time_stamp=False,
            )

        return ratio, surrogate1, surrogate2, policy_loss, value_loss, loss



    def _learn(self, trajectory, discount_factor):

        self.__debug_path = None

        state_batch, action_batch, next_state_batch, reward_batch, done_batch, log_prob_batch = self._interpret_trajectory(trajectory)

        if self._should_log():
            
            self.__debug_path = self.lg.new_relative_path_if_exists(
                "ppo_step.txt", dir="_ppo_debug"
            )

            self.lg.writeLine(f"Doing learning with state batch {state_batch.shape}, action_batch {action_batch.shape}, next_state_batch {next_state_batch.shape}, reward_batch {reward_batch.shape}, done_batch {done_batch.shape}, log_prob_batch {log_prob_batch.shape}",
                    file=self.__debug_path,
                    use_time_stamp=False,
                )



        if self.compare_old_and_new_critic_predictions:
            self.__old_critic_model.clone_other_model_into_this(self.critic)

        super()._learn(trajectory, discount_factor)


            
        if self.compare_old_and_new_critic_predictions:

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

        self.__current_interval += 1