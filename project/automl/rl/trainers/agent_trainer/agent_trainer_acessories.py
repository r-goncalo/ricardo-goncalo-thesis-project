from automl.basic_components.dynamic_value import get_value_or_dynamic_value
from automl.component import Component, InputSignature, requires_input_proccess

from automl.core.advanced_input_management import ComponentInputSignature
from automl.fundamentals.acessories import AcessoryComponent


from automl.loggers.logger_component import ComponentWithLogging
from automl.loggers.result_logger import ResultLogger
from automl.rl.learners.learner_component import LearnerSchema
from automl.rl.trainers.agent_trainer_component import AgentTrainer
import torch


class AgentTrainerConvergenceDetector(AcessoryComponent, ComponentWithLogging):
    
    '''
    Detects convergence by comparing old and new values and noting their differences
    '''

    # INITIALIZATION --------------------------------------------------------------------------

    parameters_signature = {
                                                       
                        "standard_deviation_treshold" : InputSignature(default_value=10),
                        
                        "value_key" : InputSignature(default_value="episode_reward"),

                        "n_values_to_use" : InputSignature(default_value=100),

                        "check_interval" : InputSignature(default_value=10)

                        }    
    
    
    def _proccess_input_internal(self): #this is the best method to have initialization done right after, input is already defined
        
        super()._proccess_input_internal()
                
        self.memory_size = self.get_input_value("memory_size")

        self.affected_component : AgentTrainer = self.affected_component

        if self.affected_component is None:
            raise Exception(f"Needs training")
        
        self.affected_component.initialize_external_end_request(self.name)

        self.standard_deviation_treshold = self.get_input_value("standard_deviation_treshold")

        self.value_key = self.get_input_value("value_key")
        self.n_values_to_use = self.get_input_value("n_values_to_use")
        self.check_interval = self.get_input_value("check_interval")

        self.lg.writeLine(f"Convergence will be noted with standard deviation, using an average of {self.n_values_to_use} values, that must be bellow {self.standard_deviation_treshold}")

        self.current_check_counter = 0

    def _check_last_n_values_standard_deviation(self):
        """
        Checks if the standard deviation of the last n_values_to_use
        values of value_key is below the defined threshold.
        """

        # Access trainer results logger
        results_logger : ResultLogger = self.affected_component.get_results_logger()

        # Get dataframe (assumes ResultLogger exposes dataframe attribute)
        df = results_logger.get_dataframe()

        if df is None or len(df) < self.n_values_to_use:
            return False

        if self.value_key not in df.columns:
            raise Exception(
                f"Value key '{self.value_key}' not found in results columns: {df.columns}"
            )

        # Get last N values
        last_values = df[self.value_key].tail(self.n_values_to_use).values

        # Compute standard deviation
        std = float(torch.tensor(last_values, dtype=torch.float32).std().item())

        self.lg.writeLine(
            f"[ConvergenceDetector] Last {self.n_values_to_use} "
            f"{self.value_key} std = {std}"
        )

        # Return True if below threshold
        return std < self.standard_deviation_treshold

    @requires_input_proccess
    def as_fun(self, values : dict):
        '''To be called after a functionality is executed'''

        if self.current_check_counter == 0:
            if self._check_last_n_values_standard_deviation():
                self.affected_component.request_end_from_external(self.name)
        
        self.current_check_counter += 1

        if self.current_check_counter == self.check_interval:
            self.current_check_counter = 0

    