from swarm import Swarm, Agent

client = Swarm()

agent = Agent(
    name="Agent",
    instructions="You are a helpful agent.",
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
