from .base_agent import MetaAgent

# import the agent class here
from .r2r import R2RAgent, R2RAugAgent
from .reverie import REVERIEAgent, REVERIEAugAgent
from .cvdn import CVDNAgent
from .soon import SOONAgent
from .scanqa import ScanQAAgent
from .llava import LLaVAAgent


def load_agent(name, *args, **kwargs):
    cls = MetaAgent.registry[name]
    return cls(*args, **kwargs)