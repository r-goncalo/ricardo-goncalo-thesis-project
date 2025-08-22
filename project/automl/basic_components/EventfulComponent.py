

from automl.component import Component


class Event:
    
    
    def __init__(self):
        self.subscriptions = []

    def subscribe(self, subscriber : Component, fn):
        self.subscriptions.append((subscriber, fn))
        
    def unsubscribe(self, subscriber, fn=None):
        pass
        

    def notify(self, *args, **kwargs):
        for (_, fn) in self.subscriptions:
            fn(*args, **kwargs)


    
class EventfulComponent(Component):
    
    '''A component with events'''
    
    STATIC_EVENTS : dict[str, Event] = {} # static events of component
    
    
    def __init__(self):
        
        super().__init__()
        
        self.EVENTS = {**self.STATIC_EVENTS} # initialized events attribute of component, with access to the 
        
    
    def subscribe_event(self, event_key, fn):
        
        self.EVENTS[event_key].subscribe(fn)
        
        
    def _event_activated(self, event_key):
        
        self.EVENTS[event_key].notify()