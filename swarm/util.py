import inspect
from datetime import datetime
from typing import List

# HC 09:14 2025/01/14 
def old_update_global_history(
    global_history: List[dict] = None,
    system_message: dict = None,
    user_messages: List[dict] = None,
    assistant_message: dict = None,
    tool_responses: List[dict] = None,
) -> List[dict]:
    """
    Updates the global conversation history and returns the updated history.

    Parameters:
        global_history (List[dict]): The global conversation history to update. If None, the function does nothing.
        system_message (dict): A system message to add to history (optional).
        user_messages (List[dict]): A list of user messages to add (optional).
        assistant_message (dict): An assistant message to add (optional).
        tool_responses (List[dict]): Tool responses to add (optional).

    Returns:
        List[dict]: The updated global history or None if no history is provided.
    """

    if global_history is None:
        debug_print(True, "Warning: global_history is None. No updates were made.")
        return None

    if system_message:
        global_history.append(system_message)
    if user_messages:
        global_history.extend(user_messages)
    if assistant_message:
        global_history.append(assistant_message)
    if tool_responses:
        global_history.extend(tool_responses)

    return global_history

def update_global_history(
    global_history: List[dict],
    run_id: int,
    system_message: dict = None,
    user_messages: List[dict] = None,
    assistant_message: dict = None,
    tool_responses: List[dict] = None,
):
    """
    Updates the global history by associating messages with a specific run_id.

    Parameters:
        global_history (List[dict]): Global conversation history with grouped runs.
        run_id (int): Identifier for the current run.
        system_message (dict): A system message to add (optional).
        user_messages (List[dict]): User messages to add (optional).
        assistant_message (dict): Assistant message to add (optional).
        tool_responses (List[dict]): Tool responses to add (optional).

    Returns:
        None: Updates the global_history in place.
    """
    # Ensure a run entry exists for the given run_id
    run_entry = next((run for run in global_history if run["run_id"] == run_id), None)
    if not run_entry:
        run_entry = {"run_id": run_id, "messages": []}
        global_history.append(run_entry)

    # Add messages to the current run's messages
    if system_message:
        run_entry["messages"].append(system_message)
    if user_messages:
        run_entry["messages"].extend(user_messages)
    if assistant_message:
        run_entry["messages"].append(assistant_message)
    if tool_responses:
        run_entry["messages"].extend(tool_responses)

def debug_print(debug: bool, *args: str) -> None:
    if not debug:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = " ".join(map(str, args))
    print(f"\033[97m[\033[90m{timestamp}\033[97m]\033[90m {message}\033[0m")


def merge_fields(target, source):
    for key, value in source.items():
        if isinstance(value, str):
            target[key] += value
        elif value is not None and isinstance(value, dict):
            merge_fields(target[key], value)


def merge_chunk(final_response: dict, delta: dict) -> None:
    delta.pop("role", None)
    merge_fields(final_response, delta)

    tool_calls = delta.get("tool_calls")
    if tool_calls and len(tool_calls) > 0:
        index = tool_calls[0].pop("index")
        merge_fields(final_response["tool_calls"][index], tool_calls[0])


def function_to_json(func) -> dict:
    """
    Converts a Python function into a JSON-serializable dictionary
    that describes the function's signature, including its name,
    description, and parameters.

    Args:
        func: The function to be converted.

    Returns:
        A dictionary representing the function's signature in JSON format.
    """
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
