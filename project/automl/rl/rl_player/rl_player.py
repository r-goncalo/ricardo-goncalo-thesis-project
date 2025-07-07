import os
import traceback
from automl.basic_components.evaluator_component import ComponentWithEvaluator
from automl.basic_components.exec_component import ExecComponent
from automl.component import InputSignature, Component, requires_input_proccess
from automl.core.advanced_component_creation import get_sub_class_with_correct_parameter_signature
from automl.loggers.component_with_results import ComponentWithResults
from automl.rl.agent.agent_components import AgentSchema
from automl.ml.optimizers.optimizer_components import AdamOptimizer
from automl.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automl.rl.trainers.rl_trainer_component import RLTrainerComponent
from automl.rl.environment.environment_components import EnvironmentComponent
from automl.rl.environment.pettingzoo_env import PettingZooEnvironmentWrapper
from automl.utils.files_utils import open_or_create_folder
from automl.basic_components.state_management import StatefulComponent

import torch

import gc

from automl.loggers.logger_component import LoggerSchema, ComponentWithLogging

from automl.utils.random_utils import generate_seed, do_full_setup_of_seed

# TODO this is missing the evaluation component on a RLPipeline
class RLPlayer(ExecComponent, ComponentWithLogging, ComponentWithResults, StatefulComponent, ComponentWithEvaluator):
    pass