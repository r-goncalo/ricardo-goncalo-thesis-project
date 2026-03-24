from automl.loggers.debug.component_with_logging_debug import ComponentDebug
from automl.rl.environment.parallel_environment import ParallelEnvironmentComponent
from automl.component import requires_input_process
from automl.loggers.logger_component import ComponentWithLogging
import torch

class ParallelEnvironmentComponentDebug(ParallelEnvironmentComponent, ComponentDebug):

    is_debug_schema = True
    
    parameters_signature =  {} 
    

    def step(self, actions):

        next_obs, rewards, terminations, truncations, infos = super().step(actions)  

        self.lg.writeLine(f"\nrewards: {rewards} with id {id(rewards)},\n terminations: {terminations} with id {id(terminations)},\n truncations: {truncations} with id {id(truncations)},\n infos: {infos}\n", file="steps.txt", use_time_stamp=False)

        return next_obs, rewards, terminations, truncations, infos

