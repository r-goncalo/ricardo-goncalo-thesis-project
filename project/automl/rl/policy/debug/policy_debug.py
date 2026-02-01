from automl.rl.policy.policy import Policy
from automl.rl.policy.qpolicy import QPolicy
from automl.rl.policy.stochastic_policy import StochasticPolicy
from automl.component import requires_input_proccess
import torch

class PolicyDebug(Policy):
        
    '''
    It abstracts the usage of a model for the agent in determining its actions
    '''

    is_debug_schema = True
        
    parameters_signature = {

    }   

    
    def _proccess_input_internal(self):
        
        super()._proccess_input_internal()

        self.lg.writeLine(f"Processing policy debug input...\n")
        self.lg.writeLine(f"Finished policy debug input...\n")
    
        
        
    def _setup_model(self):
        self.model.pass_input({
            "input_shape" : self.input_state_shape, 
            "output_shape" : self.output_action_shape, 
            "device" : self.device
            }) 

        
        
    def predict(self, state):
        
        '''
        Uses the state and the policy's model to predict an action
        Returns the action value for each of the passed states in a tensor
        '''
        
        predicted_value = super().predict(state)

        self.lg.writeLine(f"Predicted value: {predicted_value}", file='predicted_values.txt')

        return predicted_value
    
        
    def random_prediction(self):    

        random_predicted_value = super().random_prediction()

        self.lg.writeLine(f"Randomly predicted value: {random_predicted_value}", file='random_predicted_values.txt')

        return random_predicted_value
    


class QPolicyDebug(PolicyDebug, QPolicy):

    is_debug_schema = True

    @requires_input_proccess
    def predict(self, state):
            
        valuesForActions : torch.Tensor = self.model.predict(state) #a tensor ether in the form of [q values for each action] or [[q value for each action]]?
        
        #tensor of max values and tensor of indexes
        _, max_indexes = valuesForActions.max(dim=1)

        self.lg.writeLine(f"Predicted value: {valuesForActions} -> {max_indexes}", file='predicted_values.txt')
                        
        return max_indexes
    


class StochasticPolicyDebug(PolicyDebug, StochasticPolicy):

    is_debug_schema = True

    
    def predict_logits(self, state) -> torch.Tensor:

        probabilitiesForActionsLogits = super().predict_logits(state)
        
        self.lg.writeLine(f"Probabilities for action logits: {probabilitiesForActionsLogits}", file='predicted_values.txt')

                
        return probabilitiesForActionsLogits
    
    
    def probabilities_from_logits(self, logits) -> torch.Tensor:

        probs = super().probabilities_from_logits(logits)

        self.lg.writeLine(f"Probabilities from logits: {logits} -> {probs}", file='predicted_values.txt')
        
        return probs
    
    
    def predict_from_probability(self, probs):
    
        action = super().predict_from_probability(probs)

        self.lg.writeLine(f"Action from probabilities: {probs} -> {action}", file='predicted_values.txt')
        
        return action

    
    def predict_from_probability_with_log(self, probs):

        action, log_prob = super().predict_from_probability_with_log(probs)

        self.lg.writeLine(f"Action, log prob from probabilities: {probs} -> {action}, {log_prob}", file='predicted_values.txt')
                
        return action, log_prob  

    
    def predict_with_log(self, state):
        
        logits = self.predict_logits(state) # real numbers higher the higher probability        
        
        probs = self.probabilities_from_logits(logits) # probabilities computed from logits
        
        return self.predict_from_probability_with_log(probs)