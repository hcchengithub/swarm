
#
# Porting to Columbus
#

from swarm import Swarm, Agent
from columbus_api import Columbus
columbus = Columbus()
client_azure, headers = columbus.get_client()

client = Swarm(client_azure, headers, columbus.get_access_token)
client.__version__ # '0.1.882'

agent = Agent(
    name="Agent",
    instructions="You are a helpful agent.",
    model = 'gpt-4o-mini'
)

messages = [{"role": "user", "content": "Hi!"}]

# prompt = pyperclip.paste() # 自責分的 prompt 
# messages = [{"role": "user", "content": prompt}]

response = client.run(
    agent=agent, 
    messages=messages,
    model_override="gpt-4o-mini"    
)

print(response.messages[-1]["content"])
