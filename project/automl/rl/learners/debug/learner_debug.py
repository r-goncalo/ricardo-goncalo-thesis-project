

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
            self.lg.writeLine(f"Will compare old and new model predictions")
            self.__temporary_model : TorchModelComponent = self.__agent_model.clone()        
    
    def _learn(self, trajectory, discount_factor) -> None:

        if self.compare_old_and_new_model_predictions:
            self.__temporary_model.clone_other_model_into_this(self.__agent_model)

            state_batch, action_batch, next_state_batch, reward_batch, done_batch, *_ = self._interpret_trajectory(trajectory)

        super()._learn(trajectory, discount_factor)

        self.lg.writeLine("\naction, reward, done, old_model_predictions, new_model_predictions\n", file="batch_comparison.txt", use_time_stamp=False)

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

