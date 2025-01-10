# Standard library imports
import copy, openai
import json
from collections import defaultdict
from typing import List, Callable, Union

# Package/library imports
from openai import OpenAI
from pydantic import BaseModel # Add response_format - HC 2:52 PM 1/10/2025

# Local imports
from .util import function_to_json, debug_print, merge_chunk
from .types import (
    Agent,
    AgentFunction,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    Function,
    Response,
    Result,
)

__CTX_VARS_NAME__ = "context_variables"


class Swarm:
    def __init__(self, client=None, extra_headers = None, get_access_token=None): # HC 16:03 2025/01/02
        if not client:
            client = OpenAI()
        self.client = client
        self.get_access_token = get_access_token # columbus.get_access_token() method  # HC 16:03 2025/01/02
        self.extra_headers = extra_headers  # HC 16:03 2025/01/02
        self.__version__ = "0.1.102"


        # Release Note
        # ============
        # * Always use "pip uninstall swarm" and then "pip install . " to update new version.
        #
        # 0.1.101 1. support Azure OpenAI and client.__version__
        #         1. REPL to accept 'q', 'quit', and 'exit'
        #         1. add history to context_variables
        # 0.1.102 support OpenAI structured outputs through response_format
        #

    def get_chat_completion(
        self,
        agent: Agent,
        history: List,
        context_variables: dict,
        model_override: str,
        stream: bool,
        debug: bool,
        response_format: BaseModel = None,  # Add response_format - HC 2:52 PM 1/10/2025
    ) -> ChatCompletionMessage:
        context_variables = defaultdict(str, context_variables)
        instructions = (
            agent.instructions(context_variables)
            if callable(agent.instructions)
            else agent.instructions
        )
        messages = [{"role": "system", "content": instructions}] + history
        debug_print(debug, "Getting chat completion for...:", messages)

        # HC 8:37 PM 1/10/2025 改寫 for response_format 為 functions 都加上 "strict": True 以及 "additionalProperties": False
        # tools = [function_to_json(f) for f in agent.functions]
        tools = [
            {
                **function_to_json(f),
                "function": {
                    **function_to_json(f)["function"],
                    "strict": True,
                    "parameters": {
                        **function_to_json(f)["function"]["parameters"],
                        "additionalProperties": False,  # Add this field
                    },
                },
            }
            for f in agent.functions
        ]

        # hide context_variables from model
        for tool in tools:
            params = tool["function"]["parameters"]
            params["properties"].pop(__CTX_VARS_NAME__, None)
            if __CTX_VARS_NAME__ in params["required"]:
                params["required"].remove(__CTX_VARS_NAME__)

        # HC 16:03 2025/01/02 所以用 Swarm client to run() 不再需要擔心 access_token expiration
        if self.get_access_token:
            self.extra_headers['Authorization'] = 'Bearer ' + self.get_access_token()

        create_params = {
            "model": model_override or agent.model,
            "messages": messages,
            "tools": tools or None,
            "tool_choice": agent.tool_choice,
            "stream": stream,
            "extra_headers" : self.extra_headers  # HC 16:03 2025/01/02
        }

        if tools:
            create_params["parallel_tool_calls"] = agent.parallel_tool_calls

        # return self.client.chat.completions.create(**create_params)
        # Use the OpenAI client to parse if response_format is provided  - HC 3:33 PM 1/10/2025
        if response_format:
            # 檢查 openai 如果尚未修正 is_given() 就發出警告
            if openai._utils._utils.is_given(None) or openai._utils._utils.is_given([]):
                assert False "OpenAI function is_given() in trouble that blocks response_format structured outputs from Swarm. See my workaround at https://www.notion.so/Activity-Log-14ae3eb16f9380928722d2a020aed0af?pvs=4#177e3eb16f9380ec9f99dd9764a60a7b"
            create_params.pop("stream") # TypeError: Completions.parse() got an unexpected keyword argument 'stream' - HC 4:28 PM 1/10/2025
            completion = self.client.beta.chat.completions.parse(
                **create_params, response_format=response_format
            )
        else:
            completion = self.client.chat.completions.create(**create_params)

        return completion


    def handle_function_result(self, result, debug) -> Result:
        match result:
            case Result() as result:
                return result

            case Agent() as agent:
                return Result(
                    value=json.dumps({"assistant": agent.name}),
                    agent=agent,
                )
            case _:
                try:
                    return Result(value=str(result))
                except Exception as e:
                    error_message = f"Failed to cast response to string: {result}. Make sure agent functions return a string or Result object. Error: {str(e)}"
                    debug_print(debug, error_message)
                    raise TypeError(error_message)

    def handle_tool_calls(
        self,
        tool_calls: List[ChatCompletionMessageToolCall],
        functions: List[AgentFunction],
        context_variables: dict,
        debug: bool,
    ) -> Response:
        function_map = {f.__name__: f for f in functions}
        partial_response = Response(
            messages=[], agent=None, context_variables={})

        for tool_call in tool_calls:
            name = tool_call.function.name
            # handle missing tool case, skip to next tool
            if name not in function_map:
                debug_print(debug, f"Tool {name} not found in function map.")
                partial_response.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "tool_name": name,
                        "content": f"Error: Tool {name} not found.",
                    }
                )
                continue
            args = json.loads(tool_call.function.arguments)
            debug_print(
                debug, f"Processing tool call: {name} with arguments {args}")

            func = function_map[name]
            # pass context_variables to agent functions
            if __CTX_VARS_NAME__ in func.__code__.co_varnames:
                args[__CTX_VARS_NAME__] = context_variables
            raw_result = function_map[name](**args)

            result: Result = self.handle_function_result(raw_result, debug)
            partial_response.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "tool_name": name,
                    "content": result.value,
                }
            )
            partial_response.context_variables.update(result.context_variables)
            if result.agent:
                partial_response.agent = result.agent

        return partial_response

    def run_and_stream(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ):
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)

        # history = copy.deepcopy(messages)
        # Ensure context_variables includes history so as to support get_history() function
        if "history" not in context_variables:
            context_variables["history"] = copy.deepcopy(messages)
        history = context_variables["history"]

        init_len = len(messages)

        while len(history) - init_len < max_turns:

            message = {
                "content": "",
                "sender": agent.name,
                "role": "assistant",
                "function_call": None,
                "tool_calls": defaultdict(
                    lambda: {
                        "function": {"arguments": "", "name": ""},
                        "id": "",
                        "type": "",
                    }
                ),
            }

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=True,
                debug=debug,
            )

            yield {"delim": "start"}
            for chunk in completion:
                delta = json.loads(chunk.choices[0].delta.json())
                if delta["role"] == "assistant":
                    delta["sender"] = active_agent.name
                yield delta
                delta.pop("role", None)
                delta.pop("sender", None)
                merge_chunk(message, delta)
            yield {"delim": "end"}

            message["tool_calls"] = list(
                message.get("tool_calls", {}).values())
            if not message["tool_calls"]:
                message["tool_calls"] = None
            debug_print(debug, "Received completion:", message)
            history.append(message)

            if not message["tool_calls"] or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # convert tool_calls to objects
            tool_calls = []
            for tool_call in message["tool_calls"]:
                function = Function(
                    arguments=tool_call["function"]["arguments"],
                    name=tool_call["function"]["name"],
                )
                tool_call_object = ChatCompletionMessageToolCall(
                    id=tool_call["id"], function=function, type=tool_call["type"]
                )
                tool_calls.append(tool_call_object)

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        yield {
            "response": Response(
                messages=history[init_len:],
                agent=active_agent,
                context_variables=context_variables,
            )
        }

    def run(
        self,
        agent: Agent,
        messages: List,
        context_variables: dict = {},
        model_override: str = None,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
        response_format: BaseModel = None,  # Add response_format - HC 3:39 PM 1/10/2025
    ) -> Response:
        if stream:
            return self.run_and_stream(
                agent=agent,
                messages=messages,
                context_variables=context_variables,
                model_override=model_override,
                debug=debug,
                max_turns=max_turns,
                execute_tools=execute_tools,
            )
        active_agent = agent
        context_variables = copy.deepcopy(context_variables)

        # history = copy.deepcopy(messages)
        # Ensure context_variables includes history so as to support get_history() function
        if "history" not in context_variables:
            context_variables["history"] = copy.deepcopy(messages)
        history = context_variables["history"]

        init_len = len(messages)

        while len(history) - init_len < max_turns and active_agent:

            # get completion with current history, agent
            completion = self.get_chat_completion(
                agent=active_agent,
                history=history,
                context_variables=context_variables,
                model_override=model_override,
                stream=stream,
                debug=debug,
                response_format=response_format,  # Pass response_format - HC 3:42 PM 1/10/2025
            )
            message = completion.choices[0].message
            debug_print(debug, "Received completion:", message)
            message.sender = active_agent.name
            history.append(
                json.loads(message.model_dump_json())
            )  # to avoid OpenAI types (?)

            if not message.tool_calls or not execute_tools:
                debug_print(debug, "Ending turn.")
                break

            # handle function calls, updating context_variables, and switching agents
            partial_response = self.handle_tool_calls(
                message.tool_calls, active_agent.functions, context_variables, debug
            )
            history.extend(partial_response.messages)
            context_variables.update(partial_response.context_variables)
            if partial_response.agent:
                active_agent = partial_response.agent

        return Response(
            messages=history[init_len:],
            agent=active_agent,
            context_variables=context_variables,
        )
