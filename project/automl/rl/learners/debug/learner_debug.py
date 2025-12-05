

from automl.rl.learners.learner_component import LearnerSchema
from automl.loggers.logger_component import ComponentWithLogging
from automl.component import requires_input_proccess
from automl.rl.learners.q_learner import DeepQLearnerSchema, QLearnerSchema
from automl.ml.models.torch_model_components import TorchModelComponent
from automl.ml.models.torch_model_utils import model_parameter_distance
from automl.core.input_management import InputSignature
import torch

class LearnerDebug(LearnerSchema, ComponentWithLogging):

    is_debug_schema = True

    parameters_signature = {
        "compare_old_and_new_model_predictions" : InputSignature(default_value=True)
    }

    def _proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._proccess_input_internal()

        self.__agent_model = self.agent.policy.model

        self.compare_old_and_new_model_predictions = self.get_input_value("compare_old_and_new_model_predictions")
        
        if self.compare_old_and_new_model_predictions:
            self.__temporary_model : TorchModelComponent = self.__agent_model.clone()        
    
    def _learn(self, trajectory, discount_factor) -> None:

        if self.compare_old_and_new_model_predictions:
            self.__temporary_model.clone_other_model_into_this(self.__agent_model)

            state_batch, action_batch, next_state_batch, reward_batch, done_batch, *_ = self._interpret_trajectory(trajectory)

        super()._learn(trajectory, discount_factor)

        self.lg.writeLine("\naction, reward, done, old_model_predictions, new_model_predictions, new_model_precitions\n", file="batch_comparison.txt", use_time_stamp=False)

        if self.compare_old_and_new_model_predictions:

            with torch.no_grad():
                old_model_predictions = self.__temporary_model.predict(state_batch)
                new_model_precitions = self.__agent_model.predict(state_batch)

            self.lg.writeLine(f"Used previous version of model and new version on state_batch with shape {state_batch.shape} to produce predicitons with shape {new_model_precitions.shape} and {old_model_predictions.shape}", file="batch_comparison.txt", use_time_stamp=False)

            for i in range(len(state_batch)):

                action_val = action_batch[i].detach().cpu().numpy()
                reward_val = reward_batch[i].detach().cpu().numpy()
                done_val = done_batch[i].detach().cpu().numpy()
                old_model_prediction_val = old_model_predictions[i].detach().cpu().numpy()
                new_model_precitions_val = new_model_precitions[i].detach().cpu().numpy()

                self.lg.writeLine(f"{i}: {action_val}, {reward_val}, {done_val}, {old_model_prediction_val}, {new_model_precitions_val}", file="batch_comparison.txt", use_time_stamp=False)


class QLearnerDebug(LearnerDebug, QLearnerSchema):

    is_debug_schema = True

    parameters_signature = {
    }

    def _proccess_input_internal(self):
        super()._proccess_input_internal()

        self.lg.open_or_create_relative_folder("learning")


    def _apply_model_prediction_given_state_action_pairs(self, state_batch, action_batch):

        '''Returns the values predicted by the current model and the values for the specific actions that were passed''' 

        predicted_actions_values, predicted_values_for_actions = super()._apply_model_prediction_given_state_action_pairs(state_batch, action_batch)

        self.lg.writeLine(f"\nComputed predicted_actions_values and value for action chosen:\n", file=self.__path_to_write, use_time_stamp=False)

        for i in range(self.batch_size):
            self.lg.writeLine(f"{i}: {predicted_actions_values[i]} [ {action_batch[i]} ] -> {predicted_values_for_actions[i]}", file=self.__path_to_write, use_time_stamp=False)
    
        return predicted_actions_values, predicted_values_for_actions


    def _apply_value_prediction_to_next_state(self, next_state_batch, done_batch, reward_batch, discount_factor):

        '''
        Returns the predicted values for the next state
        
        They are given by appying the Q function to them and then chosing the next 

        '''

        next_state_q_values, next_state_v_values = super()._apply_value_prediction_to_next_state(next_state_batch, done_batch, reward_batch, discount_factor)

        self.lg.writeLine(f"\nComputed done, next_state_values computed by target and q value of action chosen:\n", file=self.__path_to_write, use_time_stamp=False)

        for i in range(self.batch_size):
            self.lg.writeLine(f"{i}: {done_batch[i]}, {next_state_q_values[i]} -> {next_state_v_values[i]}", file=self.__path_to_write, use_time_stamp=False)


        return next_state_q_values, next_state_v_values
    

    def _calculate_chosen_actions_correct_q_values(self, next_state_v_values, discount_factor, reward_batch):

        old_action_values = next_state_v_values.clone()

        correct_q_values_for_chosen_action = super()._calculate_chosen_actions_correct_q_values(next_state_v_values, discount_factor, reward_batch)

        self.lg.writeLine(f"\nNext action values after multiplying by discount factor {discount_factor} and adding reward:\n", file=self.__path_to_write, use_time_stamp=False)

        for i in range(self.batch_size):
            self.lg.writeLine(f"{i}: {correct_q_values_for_chosen_action[i]} = {old_action_values[i]} * {discount_factor} + {reward_batch[i]}", file=self.__path_to_write, use_time_stamp=False)

        return correct_q_values_for_chosen_action
    
    def _optimize_with_predicted_model_values_and_correct_values(self, predicted_values, correct_values):

        self.lg.writeLine(f"\nOptimizing using error of original predicted action values and target done on future state:\n", file=self.__path_to_write, use_time_stamp=False)

        for i in range(self.batch_size):
            self.lg.writeLine(f"{i}: {predicted_values[i]} vs {correct_values[i]}", file=self.__path_to_write, use_time_stamp=False)

        super()._optimize_with_predicted_model_values_and_correct_values(predicted_values, correct_values)


    def _learn(self, trajectory, discount_factor) -> None:

        self.__path_to_write = self.lg.new_relative_path_if_exists("computation.txt", dir="learning")
        
        super()._learn(trajectory, discount_factor)






class DQNLearnerDebug(QLearnerDebug, DeepQLearnerSchema):

    is_debug_schema = True

    parameters_signature = {
        "compare_old_and_new_target_predictions" : InputSignature(default_value=True),
        "compare_old_and_new_target_model_params" : InputSignature(default_value=True),
    }

    def _proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._proccess_input_internal()

        self.compare_old_and_new_target_predictions = self.get_input_value("compare_old_and_new_target_predictions")  
        self.compare_old_and_new_target_model_params = self.get_input_value("compare_old_and_new_target_model_params")        
        
        if self.compare_old_and_new_target_predictions: 
            self.__temporary_target_model = self.target_net.clone()

        if self.compare_old_and_new_target_model_params:
            self.__temporary_target_model_v2 = self.target_net.clone()
    
    def _learn(self, trajectory, discount_factor) -> None:

        if self.compare_old_and_new_target_predictions:

            self.__temporary_target_model.clone_other_model_into_this(self.target_net)

            state_batch, action_batch, next_state_batch, reward_batch, done_batch, *_ = self._interpret_trajectory(trajectory)

        super()._learn(trajectory, discount_factor)

        if self.compare_old_and_new_target_predictions and self.number_optimizations_done % self.target_update_learn_interval == 0:

            self.lg.writeLine("\naction, reward, done, old_target_predictions, new_target_precitions\n", file="target_batch_comparison.txt", use_time_stamp=False)

            with torch.no_grad():
                old_model_predictions = self.__temporary_target_model.predict(state_batch)
                new_model_precitions = self.target_net.predict(state_batch)

            for i in range(len(state_batch)):

                action_val = action_batch[i].detach().cpu().numpy()
                reward_val = reward_batch[i].detach().cpu().numpy()
                done_val = done_batch[i].detach().cpu().numpy()
                old_model_prediction_val = old_model_predictions[i].detach().cpu().numpy()
                new_model_precitions_val = new_model_precitions[i].detach().cpu().numpy()

                self.lg.writeLine(f"{i}: {action_val}, {reward_val}, {done_val}, {old_model_prediction_val}, {new_model_precitions_val}", file="target_batch_comparison.txt", use_time_stamp=False)

    @requires_input_proccess            
    def update_target_model(self):

        if self.compare_old_and_new_target_model_params:

            self.lg.writeLine(f"Updating target model\n", file="target_update.txt")
            self.__temporary_target_model_v2.clone_other_model_into_this(self.target_net)

        super().update_target_model()

        if self.compare_old_and_new_target_model_params:

            l2_distance, avg_distance, cosine_sim = model_parameter_distance(self.__temporary_target_model_v2, self.target_net)

            self.lg.writeLine(f"Difference between old and new target model", use_time_stamp=False, file="target_update.txt")
            self.lg.writeLine(f"    l2_distance: {l2_distance}\n    avg_distance: {avg_distance}\n    cosine_sime: {cosine_sim}\n\n", file="target_update.txt", use_time_stamp=False)
    