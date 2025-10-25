# Thank you for participating in the Unwrap Hackathon!
# Free OpenAI access is granted as part of the event to help you build unlimited by cost.
# With great throughput also comes great responsibility! There is an expectation you will not abuse the free credits, as that will hamper our ability to offer similar perks at future events.
# The api keys here will be revoked at the end of the event.


import asyncio
import os
from typing import List, Dict, Optional, Any
from openai import AsyncAzureOpenAI, pydantic_function_tool
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, Field
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

endpoint = "https://unwrap-hackathon-oct-20-resource.cognitiveservices.azure.com/"


class GPT5Deployment(str, Enum):
    GPT_5_NANO = "gpt-5-nano"  # very fast and cheap for high volume (>1000) tasks that.
    GPT_5_MINI = "gpt-5-mini"  # for when nano doesn't cut it
    GPT_5 = "gpt-5"  # quite expensive, this should be for small N tasks like final answers/analysis.


class ReasoningEffort(str, Enum):
    MINIMAL = "minimal"  # useful for fast, quick decisions
    LOW = "low"  # the most thinking you'll probably need
    MEDIUM = (
        "medium"  # extended for really hard tasks - can take 10s of seconds to return
    )
    HIGH = "high"  # math olympiad type tasks, takes forever and is expensive and you 99% don't need this


subscription_key = os.getenv("SUBSCRIPTION_KEY")

# Semaphore to limit concurrent OpenAI calls to 20
_openai_semaphore = asyncio.Semaphore(20)


async def create_openai_completion(
    messages: List[Dict[str, str]],
    model: GPT5Deployment = GPT5Deployment.GPT_5_NANO,
    reasoning_effort: ReasoningEffort = ReasoningEffort.MINIMAL,
    max_completion_tokens: int = 16384,
    tools: Optional[List[type[BaseModel]]] = None,
    tool_choice: Optional[str | Dict[str, Any]] = None,
    client: Optional[AsyncAzureOpenAI] = None,
) -> ChatCompletion:
    """
    Primary OpenAI call function that uses a semaphore to limit concurrency.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: GPT model deployment to use
        reasoning_effort: Reasoning effort level
        max_completion_tokens: Maximum tokens in completion
        tools: Optional list of Pydantic BaseModel classes to use as tools
        tool_choice: Optional tool choice control ("auto", "none", "required", or specific tool dict)
        client: Optional pre-configured client, creates new one if None

    Returns:
        ChatCompletion response from OpenAI
    """
    async with _openai_semaphore:
        if client is None:
            client = AsyncAzureOpenAI(
                api_version="2024-12-01-preview",
                azure_endpoint=endpoint,
                api_key=subscription_key,
            )

        openai_tools = None
        if tools:
            openai_tools = [pydantic_function_tool(tool) for tool in tools]

        request_params = {
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "model": model,
            "reasoning_effort": reasoning_effort,
        }

        if openai_tools:
            request_params["tools"] = openai_tools

            if tool_choice is not None:
                request_params["tool_choice"] = tool_choice

        response = await client.chat.completions.create(**request_params)

        return response


# Example Pydantic tool model
class GetWeatherTool(BaseModel):
    """Get current weather for a location."""

    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")
    unit: str = Field(
        default="celsius", description="Temperature unit (celsius or fahrenheit)"
    )

    def execute(self) -> Dict[str, Any]:
        """Execute the tool and return weather data."""
        # This would normally call a weather API
        return {
            "location": self.location,
            "temperature": "22°C" if self.unit == "celsius" else "72°F",
            "condition": "sunny",
            "unit": self.unit,
        }


def execute_tool_call(
    tool_call, available_tools: Dict[str, type[BaseModel]]
) -> Dict[str, Any]:
    """
    Execute a tool call using the appropriate Pydantic model.

    Args:
        tool_call: The tool call from OpenAI response
        available_tools: Dict mapping tool names to Pydantic model classes

    Returns:
        Result of the tool execution
    """
    import json

    tool_name = tool_call.function.name
    if tool_name not in available_tools:
        return {"error": f"Tool {tool_name} not found"}

    try:
        # Parse arguments and create tool instance
        args = json.loads(tool_call.function.arguments)
        tool_instance = available_tools[tool_name](**args)

        # Execute the tool if it has an execute method
        if hasattr(tool_instance, "execute"):
            return tool_instance.execute()
        else:
            return {"error": f"Tool {tool_name} does not have an execute method"}

    except Exception as e:
        return {"error": f"Error executing tool {tool_name}: {str(e)}"}


async def example_basic_chat() -> None:
    """Example of basic chat completion without tools."""
    print("=== Example: Basic Chat ===")
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "I am going to Paris, what should I see?",
        },
    ]

    response = await create_openai_completion(messages)
    print(response.choices[0].message.content)


async def example_auto_tool_selection() -> None:
    """Example of chat with tools where model decides whether to use them."""
    print("\n=== Example: Auto Tool Selection ===")
    messages_with_tools = [
        {
            "role": "system",
            "content": "You are a helpful assistant with access to weather information.",
        },
        {
            "role": "user",
            "content": "What's the weather like in San Francisco?",
        },
    ]

    response_with_tools = await create_openai_completion(
        messages=messages_with_tools,
        tools=[GetWeatherTool],
        tool_choice="auto",  # Let the model decide
    )
    print(f"Response: {response_with_tools.choices[0].message.content}")

    # Check if the model wants to use a tool
    if response_with_tools.choices[0].message.tool_calls:
        print("Model requested tool calls:")
        available_tools = {"GetWeatherTool": GetWeatherTool}

        for tool_call in response_with_tools.choices[0].message.tool_calls:
            print(f"Tool: {tool_call.function.name}")
            print(f"Arguments: {tool_call.function.arguments}")

            # Execute the tool
            result = execute_tool_call(tool_call, available_tools)
            print(f"Tool result: {result}")


async def example_required_tool_usage() -> None:
    """Example of forcing the model to use tools."""
    print("\n=== Example: Required Tool Usage ===")
    messages_forced_tool = [
        {
            "role": "user",
            "content": "Get me some weather data.",
        },
    ]

    response_forced = await create_openai_completion(
        messages=messages_forced_tool,
        tools=[GetWeatherTool],
        tool_choice="required",  # Force tool usage
    )

    print(f"Forced response: {response_forced.choices[0].message.content}")
    if response_forced.choices[0].message.tool_calls:
        print("Forced tool calls:")
        for tool_call in response_forced.choices[0].message.tool_calls:
            print(f"Tool: {tool_call.function.name}")
            result = execute_tool_call(tool_call, {"GetWeatherTool": GetWeatherTool})
            print(f"Result: {result}")


async def example_disabled_tools() -> None:
    """Example of explicitly disabling tool usage even when tools are available."""
    print("\n=== Example: Disabled Tools ===")
    messages_no_tools = [
        {
            "role": "user",
            "content": "Tell me about the weather without using any tools.",
        },
    ]

    response_no_tools = await create_openai_completion(
        messages=messages_no_tools,
        tools=[GetWeatherTool],
        tool_choice="none",  # Explicitly disable tools
    )

    print(f"Response without tools: {response_no_tools.choices[0].message.content}")
    print(
        f"Tool calls made: {len(response_no_tools.choices[0].message.tool_calls or [])}"
    )


# async def main() -> None:
#     """Run all example functions to demonstrate different OpenAI usage patterns."""
#     await example_basic_chat()
#     await example_auto_tool_selection()
#     await example_required_tool_usage()
#     await example_disabled_tools()


# if __name__ == "__main__":
#     asyncio.run(main())
