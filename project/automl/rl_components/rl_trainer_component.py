




from ..component import InputSignature, Component, requires_input_proccess
from ..logger_component import LoggerComponent

import torch
import time

class RLTrainerComponent(LoggerComponent):

    TRAIN_LOG = 'train.txt'
    
    input_signature = {"device" : InputSignature(),
                       "num_episodes" : InputSignature(),
                       "environment" : InputSignature(),
                       "state_memory_size" : InputSignature(),
                       "agents" : InputSignature(),
                       "limit_steps" : InputSignature(),
                       "optimization_interval" : InputSignature(),
                       "save_interval" : InputSignature(default_value=100)}
    
    exposed_values = {"total_steps" : 0} #this means we'll have a dic "values" with this starting values

    def proccess_input(self): #this is the best method to have initialization done right after
        
        super().proccess_input()
        
        self.device = self.input["device"]
    
        self.limit_steps = self.input["limit_steps"]
        self.num_episodes =self.input["num_episodes"]  
        
        self.env = self.input["environment"]
        
        self.state_memory_size = self.input["state_memory_size"]
                
        self.agents = self.input["agents"] #this is a dictionary with {agentName -> agentComponent}, the environment must be able to return the agent name
        
        self.optimization_interval = self.input["optimization_interval"]
        
        self.save_interval = self.input["save_interval"]
        
        self.total_score = []
        
        self.episode_durations = []
        
        self.episode_time_per_step_durations = []


    @requires_input_proccess
    def run_episodes(self):
        
        self.lg.writeLine("Starting to run episodes of training")
    
        timeBeforeTraining =  time.time()
        
        #each episode is an instance of playing the game
        for i_episode in range(self.num_episodes):
            
            self.lg.writeLine(f"Starting to run episode {i_episode}")
            
            timeBeforeEpisode = time.time()
            
            self.env.reset()
            
            # Initialize the environment and get its state
            state, reward, done, info = self.env.last()
                                    
            state = torch.stack([state[i] for i in list(range(len(state))) * self.state_memory_size ]) #fill the memory with the initial state              
            self.total_score.append(0) #adds total score for this episode, starting at 0
            
            
            t = 0 #tracker of the number of steps we'll do in this episode
            
            for agent in self.env.agent_iter(): #iterates infinitely over the agents that should be acting in the environment
                                    
                agentInTraining = self.agents[agent] #gets the agent in the format we're using
                
                action = agentInTraining.select_action(state) # decides the next action to take (can be random)
                                             
                self.env.step(action) #makes the game proccess the action that was taken
                
                reward = self.env.rewards()[agent] #the individual reward of the agent
        
                boardObs, reward, done, info = self.env.last()
                                                
                if done:
                    next_state = None
                else:
                    
                    if self.state_memory_size > 1: #if we have memory in our states (we use previous states as inputs for actions)
                    
                        next_state = torch.stack([  state[i][u] for u in range(0, self.state_memory_size)  for i in range(1, self.state_memory_size)]) #adds the previous perceived states to the memory of the next state

                        for window in boardObs:
                            next_state = torch.stack((next_state, window)) #adds the new perceived state
                            
                    else:
                        
                        next_state = boardObs #if we have no memory (if we just use the current state)
                                            
            
                # Store the transition in memory
                agentInTraining.memory.push(state, action, next_state, reward)
                
                # Save the (next) previous state
                state = next_state
                
                if self.values["total_steps"] % self.optimization_interval == 0:
                    
                    self.lg.writeLine(f"In episode {i_episode}, optimizing at step {t} that is the total step {self.values['total_steps']}")
                    self.optimizeAgents()
                    

                t += 1
                self.values["total_steps"] += 1 #we just did a step
                                
                if done:
                    self.episode_durations.append(t)
                    break


                if self.limit_steps >= 1 and t >= self.limit_steps:
                    self.lg.writeLine("In episode " + str(i_episode) + ", reached step " + str(t) + " that is beyond the current limit, " + str(self.limit_steps))
                    self.episode_durations.append(t)
                    break
               
            timeDurationOfEpisode = time.time() - timeBeforeEpisode
            
            self.episode_time_per_step_durations.append(timeDurationOfEpisode / t) 
                
            self.lg.writeLine("Ended episode: " + str(i_episode) + " with duration: " + str(t) + ", total reward: " + str(self.total_score[len(self.episode_durations) - 1]) + " and real time duration of " + str(timeDurationOfEpisode) + " seconds", file=self.TRAIN_LOG)                
                
                
            #if we reached a point where it is supposed to save
            if(i_episode > 0 and i_episode < self.num_episodes - 1 and i_episode % self.save_interval == 0):
                self.lg.writeLine("Doing intermedian saving of results during training...", file=self.TRAIN_LOG)
                self.saveData()
        
        timeTrainingTook = time.time() - timeBeforeTraining
        self.lg.writeLine("\nTraining took " + str(timeTrainingTook) + " seconds, " + str(timeTrainingTook / self.values['total_steps']) + " per step (" + str(self.values['total_steps']) + ")")     


    def optimizeAgents(self):
        
        for agentInTraining in  self.agents.values():
            
            self.lg.writeLine("Optimizing agent " + str(agentInTraining.name))
            
            timeBeforeOptimizing = time.time()
                                
            agentInTraining.optimize_policy_model() # TODO : Take attention to this, the agents optimization strategy is too strict
            
            self.lg.writeLine("Optimization took " + str(time.time() - timeBeforeOptimizing) + " seconds")