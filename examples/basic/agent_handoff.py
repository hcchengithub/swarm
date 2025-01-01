from swarm import Swarm, Agent

client = Swarm()

english_agent = Agent(
    name="English Agent",
    instructions="You only speak English.",
)

spanish_agent = Agent(
    name="Spanish Agent",
    instructions="You only speak Spanish.",
)

taiwanease_agent = Agent(
    name="Taiwan Agent",
    instructions="You only speak 正體中文.",
)


def transfer_to_spanish_agent():
    """Transfer spanish speaking users immediately."""
    return spanish_agent

def transfer_to_taiwan_agent():
    """Transfer 正體中文 speaking users immediately."""
    return taiwanease_agent


english_agent.functions.append(transfer_to_spanish_agent)
english_agent.functions.append(transfer_to_taiwan_agent)

messages = [{"role": "user", "content": "Hola. ¿Como estás?"}]
messages = [{"role": "user", "content": "請講西文。"}]
response = client.run(
    agent=english_agent, 
    messages=messages,
    model_override="gpt-4o-mini"    
)
print(response.messages[-1]["content"])
