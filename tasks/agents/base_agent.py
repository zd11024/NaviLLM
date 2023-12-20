
class MetaAgent(type):
    registry = {}

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        if 'name' in attrs:
            MetaAgent.registry[attrs['name']] = cls

class BaseAgent(metaclass=MetaAgent):
    def __init__(self, *args, **kwargs):
        pass

    def get_prompt(self, *args, **kwargs):
        raise NotImplementedError

