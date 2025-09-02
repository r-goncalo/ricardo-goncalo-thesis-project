

from automl.basic_components.evaluator_component import EvaluatorComponent
from automl.component import Component, requires_input_proccess
import random

class RandomMockEvaluator(EvaluatorComponent):
    

    def get_metrics_strings(self) -> list[str]:
        return [*super().get_metrics_strings(), "result"]
    
    @requires_input_proccess
    def _evaluate(self, component_to_evaluate : Component):

        result = random.random()
        return {"result" : result, **super().evaluate(component_to_evaluate)}