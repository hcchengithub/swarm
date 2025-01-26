
#
# Porting to Columbus
#
import json
from swarm import Swarm, Agent
client = Swarm()
client.__version__ # '0.1.107'

agent_a = Agent(
    name="Agent A",
    instructions="Pass all requests to 'Agent B'",
    model = 'gpt-4o-mini',
)

agent_b = Agent(
    name="Agent B",
    instructions="Pass all requests to 'Agent A'",
    model = 'gpt-4o-mini'
)

def transfer_to_agent_a():
    """Transfer to Agent A immediately."""
    return agent_a

def transfer_to_agent_b():
    """Transfer to Agent B immediately."""
    return agent_b

agent_a.functions = [transfer_to_agent_b]
agent_b.functions = [transfer_to_agent_a]

messages = [{"role": "user", "content": "Hi!"}]
response = client.run(
    agent=agent_a, 
    messages=messages,
    model_override="gpt-4o-mini"    
)
print(response.messages[-1]["sender"], ':' ,response.messages[-1]["content"])
print(json.dumps(client.global_history,indent=2))
