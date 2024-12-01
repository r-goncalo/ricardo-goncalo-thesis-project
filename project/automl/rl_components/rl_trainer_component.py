






from ..component import Component
import torch
import time

class RLTrainerComponent(Component):

    TRAIN_LOG = 'train.txt'

    def proccess_input(self, input): #this is the best method to have initialization done right after
        
        super().proccess_input(input)
        
        self.name = self.input["name"]                
        self.device = self.input["device"]
        self.lg = self.input["logger"]
    
        self.limit_steps = self.input["limit_steps"]
        self.num_episodes =self.input["num_episodes"]  
        
        self.env = self.input["environment"]
        
        self.state_memory_size = self.input["state_memory_size"]
        
        self.board_state_translator = self.input["board_state_translator"]
        
        self.agents = self.input["agents"] #this a dictionary with {agentName -> agentComponent}, the environment must be able to return the agent name
        


    def run_episodes(self):
    
        timeBeforeTraining =  time.time()
        
        #each episode is an instance of playing the game
        for i_episode in range(self.num_episodes):
            
            timeBeforeEpisode = time.time()
            
            self.env.reset()
            
            # Initialize the environment and get its state
            boardObs, reward, done, info = self.env.last()
                                    
            state = self.board_state_translator(boardObs) #translates the stacked matrixes into the format the model will read       
            state = torch.stack([state[i] for i in list(range(len(state))) * self.state_memory_size ]) #fill the memory with the initial state              
            self.total_score.append(0) #adds total score for this episode, starting at 0
            
            
            t = 0 #tracker of the number of steps we'll do in this episode
            
            for agent in self.env.agent_iter(): #iterates infinitely over the agents that should be acting in the environment
                                    
                agentInTraining = self.agents[agent] #gets the agent in the format we're using
                
                action = agentInTraining.select_action(state, self) # decides the next action to take (can be random)
                             
                self.env.step(action) #makes the game proccess the action that was taken
                
                reward = self.env.rewards()[agent] #the individual reward of the agent

                #processes the reward
                reward = torch.tensor([reward], device=self.device)
                self.total_score[len(self.total_score) - 1] += reward.item()
        
                boardObs, reward, done, info = self.env.last()
                                
                if done:
                    next_state = None
                else:
                    next_state = [  state[i][u] for u in range(0, agentInTraining.values["z_input_size"])  for i in range(1, self.state_memory_size)] #adds the previous perceived states to the memory of the next state
                    
                    translatedBoard = self.board_state_translator(boardObs)
                    
                    for window in translatedBoard:
                        next_state.append(window) #adds the new perceived state
                    
            
                # Store the transition in memory
                agentInTraining.memory.push(state, action, next_state, reward)
                
                # Save the (next) previous state
                state = next_state
                
                if self.totalSteps % self.values["optimization_interval"] == 0:
                    
                    self.lg.writeLine("In episode " + str(i_episode) + ", optimizing at step " + str(t) + " that is the total step " + str(self.totalSteps))
                    self.optimizeAgents()
                    

                t += 1
                self.totalSteps += 1 #we just did a step
                                
                if done:
                    self.episode_durations.append(t)
                    break
                
                if self.stop_event:
                    self.lg.writeLine("Stop event received, ceasing training proccess...")
                    return


                if self.limit_steps >= 1 and t >= self.limit_steps:
                    self.lg.writeLine("In episode " + str(i_episode) + ", reached step " + str(t) + " that is beyond the current limit, " + str(self.limit_steps))
                    self.episode_durations.append(t)
                    break
               
            timeDurationOfEpisode = time.time() - timeBeforeEpisode
            
            self.episode_time_per_step_durations.append(timeDurationOfEpisode / t) 
                
            self.lg.writeLine("Ended episode: " + str(i_episode) + " with duration: " + str(t) + ", total reward: " + str(self.total_score[len(self.episode_durations) - 1]) + " and real time duration of " + str(timeDurationOfEpisode) + " seconds", file=TRAIN_LOG)                
                
                
            #if we reached a point where it is supposed to save
            if(i_episode > 0 and i_episode < self.num_episodes - 1 and i_episode % self.SAVE_INTERVAL == 0):
                self.lg.writeLine("Doing intermedian saving of results during training...", file=self.TRAIN_LOG)
                self.saveData()
        
        timeTrainingTook = time.time() - timeBeforeTraining
        self.lg.writeLine("\nTraining took " + str(timeTrainingTook) + " seconds, " + str(timeTrainingTook / self.totalSteps) + " per step (" + str(self.totalSteps) + ")")     


    def optimizeAgents(self):
        
        for agentInTraining in  self.agents.values():
            
            self.lg.writeLine("Optimizing agent " + str(agentInTraining))
            
            timeBeforeOptimizing = time.time()
                                
            agentInTraining.optimize_policy_model()
            agentInTraining.update_target_model()
            
            self.lg.writeLine("Optimization took " + str(time.time() - timeBeforeOptimizing) + " seconds")