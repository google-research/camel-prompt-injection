import copy
from collections.abc import Sequence
from typing import Any

import jsonref
import vertexai.generative_models as genai
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.functions_runtime import EmptyEnv, Env, Function, FunctionCall, FunctionsRuntime
from agentdojo.types import ChatAssistantMessage, ChatMessage, text_content_block_from_string
from google.protobuf.struct_pb2 import Struct
from openapi_pydantic import OpenAPI
from openapi_pydantic.util import PydanticSchema, construct_open_api_with_schema_class
from proto.marshal.collections.maps import MapComposite
from proto.marshal.collections.repeated import RepeatedComposite
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_random_exponential


def construct_dummy_open_api(model: type[BaseModel]) -> OpenAPI:
    return OpenAPI.model_validate(
        {
            "info": {"title": "My own API", "version": "v0.0.1"},
            "paths": {
                "/": {
                    "post": {
                        "requestBody": {
                            "content": {"application/json": {"schema": PydanticSchema(schema_class=model)}}
                        },
                        "responses": {
                            "200": {
                                "description": "pong",
                                "content": {"application/json": {"schema": PydanticSchema(schema_class=model)}},
                            }
                        },
                    }
                }
            },
        }
    )


def make_openapi_from_pydantic_model(model: type[BaseModel]) -> dict[str, Any]:
    model = copy.deepcopy(model)
    model.__name__ = model.__name__.replace(" ", "_").replace("`", "_")
    dummy_open_api_original = construct_dummy_open_api(model)
    dummy_open_api_original = construct_open_api_with_schema_class(dummy_open_api_original).model_dump_json(
        by_alias=True, exclude_none=True
    )
    dummy_open_api: dict = jsonref.loads(dummy_open_api_original, lazy_load=False)  # type: ignore
    if "components" not in dummy_open_api:
        raise ValueError("No components found in the OpenAPI object")
    if "schemas" not in dummy_open_api["components"] is None:
        raise ValueError("No components found in the OpenAPI object")
    dummy_open_api["components"]["schemas"][model.__name__].pop("title")
    return dummy_open_api["components"]["schemas"][model.__name__]


def _parameters_to_google(parameters: type[BaseModel]) -> dict[str, Any]:
    openapi_parameters = make_openapi_from_pydantic_model(parameters)  # type: ignore

    # Clean up properties
    if "properties" in openapi_parameters:
        for _, prop_value in openapi_parameters["properties"].items():
            if "anyOf" in prop_value:
                # Filter out null types from anyOf
                non_null_types = [schema for schema in prop_value["anyOf"] if schema.get("type") != "null"]

                if non_null_types:
                    # If we have valid types, use the first one
                    prop_value.clear()
                    prop_value.update(non_null_types[0])
                else:
                    # Fallback to string if no valid types
                    prop_value.clear()
                    prop_value["type"] = "string"

    return openapi_parameters


def _function_to_google(f: Function) -> genai.FunctionDeclaration:
    return genai.FunctionDeclaration(
        name=f.name,
        description=f.description,
        parameters=_parameters_to_google(f.parameters),
    )


def _merge_tool_result_messages(messages: list[genai.Content]) -> list[genai.Content]:
    merged_messages = []
    current_tool_message = None

    for message in messages:
        if any(part.function_response for part in message.parts):
            if current_tool_message is None:
                current_tool_message = genai.Content(parts=[])
            current_tool_message = genai.Content(
                parts=current_tool_message.parts + [part for part in message.parts if part.function_response]
            )
        else:
            if current_tool_message is not None:
                merged_messages.append(current_tool_message)
                current_tool_message = None
            merged_messages.append(message)

    if current_tool_message is not None:
        merged_messages.append(current_tool_message)

    return merged_messages


def _parts_from_assistant_message(assistant_message: ChatAssistantMessage) -> list[genai.Part]:
    parts = []
    if assistant_message["content"] is not None:
        for part in assistant_message["content"]:
            parts.append(genai.Part.from_text(part["content"]))
    if assistant_message["tool_calls"]:
        for tool_call in assistant_message["tool_calls"]:
            part = genai.Part.from_dict(dict(function_call=dict(name=tool_call.function, args=tool_call.args)))
            parts.append(part)
    return parts


def _message_to_google(message: ChatMessage) -> genai.Content:
    match message["role"]:
        case "user":
            return genai.Content(
                role="user",
                parts=[genai.Part.from_text(block["content"]) for block in message["content"]],
            )
        case "assistant":
            return genai.Content(role="model", parts=_parts_from_assistant_message(message))
        case "tool":
            # tool_call singular here, plural above. This is intentional
            tool_call = message["tool_call"]
            return genai.Content(
                parts=[
                    genai.Part.from_function_response(
                        name=tool_call.function,
                        response={"content": message["content"]},
                    )
                ],
            )
        case _:
            raise ValueError(f"Invalid role for Google: {message['role']}")


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3), reraise=True)
def chat_completion_request(
    model: genai.GenerativeModel,
    contents: list[genai.Content],
    tools: list[genai.Tool],
    generation_config: genai.GenerationConfig,
) -> genai.GenerationResponse:
    response: genai.GenerationResponse = model.generate_content(  # type: ignore -- streaming=False, so no iterable
        contents,
        tools=tools if tools else None,
        generation_config=generation_config,
    )
    return response


def repeated_composite_to_dict(repeated_composite: RepeatedComposite) -> list:
    l = []
    for item in repeated_composite:
        if isinstance(item, RepeatedComposite):
            v = repeated_composite_to_dict(item)
        elif isinstance(item, Struct | MapComposite):
            v = struct_to_dict(item)
        else:
            v = item
        l.append(v)
    return l


def struct_to_dict(struct: Struct | MapComposite) -> dict:
    out = {}
    for key, value in struct.items():
        if isinstance(value, Struct | MapComposite):
            out[key] = struct_to_dict(value)
        elif isinstance(value, RepeatedComposite):
            out[key] = repeated_composite_to_dict(value)
        elif isinstance(value, list):
            out[key] = [struct_to_dict(v) for v in value]
        else:
            out[key] = value
    return out


EMPTY_FUNCTION_NAME = "<empty-function-name>"


def _google_to_tool_call(function_call: genai.FunctionCall) -> FunctionCall:
    if function_call.name == "":
        function = EMPTY_FUNCTION_NAME  # sometimes the API returns an empty string
    else:
        function = function_call.name
    return FunctionCall(function=function, args=function_call.args, id="")


def _google_to_assistant_message(message: genai.GenerationResponse) -> ChatAssistantMessage:
    tool_calls = []
    text_parts = []
    for part in message.candidates[0].content.parts:
        if fn := part.function_call:
            tool_calls.append(_google_to_tool_call(fn))
        try:
            if part.text:
                text_parts.append(text_content_block_from_string(part.text))
        except AttributeError:
            pass
    return ChatAssistantMessage(role="assistant", content=text_parts, tool_calls=tool_calls)


class GoogleLLM(BasePipelineElement):
    """LLM pipeline element that uses Google Vertex AI (i.e. Gemini models).

    !!! warning
        In order to use Google LLMs, you need to run `vertexai.init()`
        before instantiating this class. This is done automatically if you create a
        pipeline with [`AgentPipeline.from_config`][agentdojo.agent_pipeline.AgentPipeline.from_config]. However,
        you need to have Google Cloud CLI installed on your machine and be authenticated.
        After being authnticated, you need to run `gcloud auth application-default login` in the terminal
        to let Python access your Google Cloud account.

    Args:
        model: The model name.
        temperature: The temperature to use for generation.
    """

    def __init__(self, model: str, temperature: float | None = 0.0):
        self.model = model
        self.generation_config = genai.GenerationConfig(temperature=temperature)

    @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3), reraise=True)
    def query(
        self,
        query: str,
        runtime: FunctionsRuntime,
        env: Env = EmptyEnv(),
        messages: Sequence[ChatMessage] = [],
        extra_args: dict = {},
    ) -> tuple[str, FunctionsRuntime, Env, Sequence[ChatMessage], dict]:
        first_message, *other_messages = messages
        if first_message["role"] == "system":
            system_instruction = first_message["content"][0]["content"]
        else:
            system_instruction = None
            other_messages = messages
        google_messages = [_message_to_google(message) for message in other_messages]
        google_messages = _merge_tool_result_messages(google_messages)
        google_functions = [_function_to_google(tool) for tool in runtime.functions.values()]
        google_tools = [genai.Tool(function_declarations=google_functions)] if google_functions else []
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=genai.Part.from_text(text=system_instruction) if system_instruction else None,
        )
        completion = chat_completion_request(
            model,
            google_messages,
            tools=google_tools,
            generation_config=self.generation_config,
        )
        output = _google_to_assistant_message(completion)
        messages = [*messages, output]
        return query, runtime, env, messages, extra_args
