from swarm import Swarm, Agent

client = Swarm()


def instructions(context_variables):
    name = context_variables.get("name", "User")
    return f"You are a helpful agent. Greet the user by name ({name})."


def account_details(context_variables: dict):
    user_id = context_variables.get("user_id", None)
    name = context_variables.get("name", None)
    print(f"Account Details: {name} {user_id}")
    return {"status":"Failed", **context_variables} # HC 2:33 AM 1/2/2025 I modify for experiment

agent = Agent(
    name="Agent",
    instructions=instructions,
    functions=[account_details],
)

context_variables = {"name": "James", "user_id": 123, "desire":"yaya"}

response = client.run(
    messages=[{"role": "user", "content": "Hi!"}],
    agent=agent,
    context_variables=context_variables,
    model_override="gpt-4o-mini"    
)
print(response.messages[-1]["content"])

response = client.run(
    messages=[{"role": "user", "content": "Print my account details!"}],
    agent=agent,
    context_variables=context_variables,
    model_override="gpt-4o-mini"    
)
print(response.messages[-1]["content"])

response = client.run(
    messages=[{"role": "user", "content": "Tell me my user ID of my account and my desire."}],
    agent=agent,
    context_variables=context_variables,
    model_override="gpt-4o-mini"    
)
print(response.messages[-1]["content"])
