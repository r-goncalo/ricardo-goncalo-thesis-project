from automl.component import InputSignature, Schema, requires_input_proccess
from automl.rl.agent.agent_components import AgentSchema
from automl.ml.optimizers.optimizer_components import AdamOptimizer
from automl.rl.exploration.epsilong_greedy import EpsilonGreedyStrategy
from automl.ml.models.model_components import ConvModelSchema
from automl.rl.trainers.rl_trainer_component import RLTrainerComponent
from automl.rl.environment.environment_components import EnvironmentComponent, PettingZooEnvironmentLoader
from automl.loggers.logger_component import LoggerSchema
from automl.loggers.result_logger import ResultLogger
from automl.utils.files_utils import open_or_create_folder
from automl.rl.rl_pipeline import RLPipelineComponent

import torch

import optuna



class HyperparameterOptimizationPipeline(LoggerSchema):
    
    parameters_signature = {
        
                        "base_component_configuration_path" : InputSignature(),                        
                        "hyperparameters_range_list" : InputSignature(),
                        "n_trials" : InputSignature(default_value=5)
                                                    
                       }
    

    # INITIALIZATION -----------------------------------------------------------------------------

    def proccess_input(self): #this is the best method to have initialization done right after
        
        super().proccess_input()
        
        self.rl_pipeline_config_path : str = self.input["base_component_configuration_path"]
        
        self.hyperparameters_range_list : list[tuple[list[str], list[str], any]] = self.input["hyperparameters_range_list"]
        
        parameter_names = [hyperparameter_specification[0] for hyperparameter_specification in self.hyperparameters_range_list]

        self.results_logger : ResultLogger = self.initialize_child_component(ResultLogger, { "logger_object" : self.lg,
            "keys" : [*parameter_names, "result"]})
        
        self.suggested_values = { parameter_name : 0 for parameter_name in parameter_names}
        
        self.n_trials = self.input["n_trials"]
        
        self.tried_configurations = 0
        

    # OPTIMIZATION -------------------------------------------------------------------------

    def create_component_to_test(self):
        
        self.lg.writeLine("Creating component to test")

        rl_pipeline : RLPipelineComponent = LoggerSchema.load_configuration(self.rl_pipeline_config_path)
        
        name = 'configuration_' + str(self.tried_configurations + 1)        
                
        configuration_logger = self.lg.openChildLog(logName=name)
        
        rl_pipeline.pass_input({"logger_object" : configuration_logger})
        
        self.lg.writeLine(f"Created component with name {name}")
        
        return rl_pipeline


    def generate_configuration(self, trial : optuna.trial, base_component : Schema):
        
        for (name, component_list, hyperparameter_name, [min, max]) in self.hyperparameters_range_list:
            
            suggested_value = trial.suggest_float(name, min, max)
            
            self.suggested_values[name] = [suggested_value]
            
            self.lg.writeLine(f"{name}: {suggested_value}")
            
            for component_localizer in component_list:
                
                if isinstance(component_localizer, str):
                    
                    component_to_change : Schema = base_component.get_child_by_name(component_localizer)
                    
                elif isinstance(component_localizer, list):
                    
                    component_to_change : Schema = base_component.get_child_by_localization(component_localizer)
                    
                component_to_change.pass_input({hyperparameter_name : suggested_value})
                
                
    # TODO: this is all assuming a component of type "Schema"
    def objective(self, trial : optuna.trial):
        
        component_to_test = self.create_component_to_test()

        self.lg.writeLine("Starting new training with hyperparameter cofiguration")
        
        self.generate_configuration(trial, component_to_test)
                
        component_to_test.train()
        
        self.tried_configurations += 1
        
        results = component_to_test.get_last_Results()
                
        reward = results["total_reward"]
        
        results_to_log = {**self.suggested_values, "total_reward" : [reward]}
        
        self.results_logger.log_results(results_to_log)
        
        return - reward #this function is minimized as a loss function, so we return the regret
    
    
    # EXPOSED METHODS -------------------------------------------------------------------------------------------------
                    
                    
    @requires_input_proccess
    def run(self):
        
        study = optuna.create_study()
        study.optimize( lambda trial : self.objective(trial), n_trials=self.n_trials)

        self.lg.writeLine(f"Best parameters: {study.best_params}")  # E.g. {'x': 2.002108042}
        

        
        
    def configure_device(self, str_device_str):
        
        try:

            self.lg.writeLine("Trying to use cuda...")
            self.device = torch.device(str_device_str)
    
        except Exception as e:
            self.device = torch.device("cpu")
            self.lg.writeLine(f"There was an error trying to setup the device in '{str_device_str}': {str(e)}")

        self.lg.writeLine("The model will trained and evaluated on: " + str(self.device))
        
    def setup_trainer(self):    
        
        rl_trainer_input = {
            "device" : self.device,
            "logger_object" : self.lg,
            "num_episodes" : self.num_episodes,
            "state_memory_size" : self.state_memory_size,
            "environment" : self.env,
            "limit_steps" : self.limit_steps ,
            "optimization_interval" : self.optimization_interval,
            "agents" : self.agents
        }        
        
        self.rl_trainer : RLTrainerComponent = self.input["rl_trainer"]
        
        self.rl_trainer.pass_input(rl_trainer_input)
            
            
    def initialize_agents_components(self):

        self.agents = self.input["agents"] #this is a dictionary with {agentName -> AgentSchema}, the environment must be able to return the agent name

        if self.agents  == {}:
            self.create_agents()
        
    def create_agents(self):

        self.lg.writeLine("Creating agents")

        agents = {}

        agentId = 1        
        for agent in self.env.agents(): #worth remembering that the order of the initialization of the agents is defined by the environment

            agent_input = {**self.input["created_agents_input"]} #the input of the agent, with base values defined by "agents_input"


            agent_name = "agent_" + str(agentId)
            agent_input["name"] = agent_name

            agent_logger = self.lg.openChildLog(logName=agent_name)
            agent_input["logger_object"] = agent_logger

            state = self.env.observe(agent)

            z_input_size = len(state)
            y_input_size = len(state[0])
            x_input_size = len(state[0][0])
            
            self.lg.writeLine("State for agent " + agent_name + " has shape: Z: " + str(z_input_size) + " Y: " + str(y_input_size) + " X: " + str(x_input_size))
            
            agent_input["state_memory_size"] = self.state_memory_size

            n_actions = self.env.action_space(agent).n
            print(f"Action space of agent {agent}: {self.env.action_space(agent)}")

            agent_input["state_shape"] = [x_input_size, y_input_size, z_input_size]
            agent_input["action_shape"] = n_actions
            
            agent_input["device"] = self.device       

            agents[agent] = self.initialize_child_component(AgentSchema, input=agent_input)

            self.lg.writeLine("Created agent in training " + agent_name)

            agentId += 1

        self.lg.writeLine("Initialized " + str(agents) + " agents")

        self.agents = agents  
        self.input["agents"] = agents #this is done because we want to save these agents in the configuration
        
        
        
    # TRAINING_PROCCESS ----------------------
        
    @requires_input_proccess
    def train(self):        
        self.rl_trainer.run_episodes()
        
        
    # RESULTS --------------------------------------
    
    def plot_graphs(self):
        
        self.rl_trainer.plot_results_graph()