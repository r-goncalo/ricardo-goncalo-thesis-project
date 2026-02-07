


from automl.rl.trainers.agent_trainer_component import AgentTrainer
from automl.ml.models.torch_model_components import TorchModelComponent
from automl.core.input_management import InputSignature
from automl.ml.models.torch_model_utils import model_parameter_distance

class AgentTrainerDebug(AgentTrainer):
        
        '''A debug class that is not meant to be used '''

        is_debug_schema = True

        parameters_signature = {
             "verify_model_difference_after_optimize" : InputSignature(default_value=True)
        }

        def _proccess_input_internal(self):

            super()._proccess_input_internal()

            self.verify_model_difference_after_optimize = self.get_input_value("verify_model_difference_after_optimize")
    
            if self.verify_model_difference_after_optimize:
                self.model : TorchModelComponent = self.agent_policy.model

                self.lg.writeLine(f"Creating temporary model to note difference in optimizations...")

                self.__temporary_model : TorchModelComponent = self.model.clone(input_for_clone={"base_directory" : self, "artifact_relative_directory" : "__temp_comp_opti", "create_new_directory" : False}, is_deep_clone=True)
        
            self.lg.writeLine(f"total_step, episode, episode_step: state + action -> new_state, reward, done\n", file="observed_transitions.txt", use_time_stamp=False)
            self.lg.writeLine(f"total_step, episode, episode_step: reward, done\n", file="training_steps.txt", use_time_stamp=False)
    
            self.learner.pass_input({"logger_object" : self.lg})
    
    
        def do_training_step(self, i_episode, env):
        
            reward, done, truncated = super().do_training_step(i_episode, env)

            # we could do something here

            return reward, done, truncated
        
        
        def do_after_training_step(self, i_episode, action, observation, reward, done, truncated):
             
            super().do_after_training_step(i_episode, action, observation, reward, done, truncated)

            self.lg.writeLine(f"{self.values['total_steps']}, {self.values['episodes_done']}, {self.values['episode_steps']}: {reward}, {done}", file="training_steps.txt")

    
        def _optimize_policy_model(self):
        
            if self.verify_model_difference_after_optimize:
                self.__temporary_model.clone_other_model_into_this(self.model)
    
            super()._optimize_policy_model()
    
            if self.verify_model_difference_after_optimize:
                l2_distance, avg_distance, cosine_sim = model_parameter_distance(self.__temporary_model, self.model)

                self.lg.writeLine(f"Optimized agent, clone has id {id(self.__temporary_model.model)} and optimized_model {id(self.model.model)}:", file="model_optimization.txt")
                self.lg.writeLine(f"    l2_distance: {l2_distance}\n    avg_distance: {avg_distance}\n    cosine_sime: {cosine_sim}\n", file="model_optimization.txt")
    

    
        def _observe_transiction_to(self, new_state, action, reward, done):

            old_state = self.agent.get_current_state_in_memory().clone()
        
            super()._observe_transiction_to(new_state, action, reward, done)

            old_state_str = str(old_state)
            if len(old_state_str) > 30:
                old_state_str = f"{old_state_str[10:]}...{old_state_str[:10]}"   
    
            new_state_str = str(new_state)
            if len(new_state_str) > 30:
                new_state_str = f"{new_state_str[10:]}...{new_state_str[:10]}"   
    
            self.lg.writeLine(f"{self.values['total_steps']}, {self.values['episodes_done']}, {self.values['episode_steps']}: {old_state_str} + {action} -> {new_state_str}, {reward}, {done}", file="observed_transitions.txt")
    