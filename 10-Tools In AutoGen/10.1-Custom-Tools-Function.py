import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.tools import FunctionTool

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

# Initialize the OpenAI model client Wiht OpenRouter Model
model_client = OpenAIChatCompletionClient(
    base_url="https://openrouter.ai/api/v1",
    model="mistralai/mistral-small-3.2-24b-instruct:free",
    api_key=api_key,
    model_info={
        "family": "mistral",
        "vision": True,
        "function_calling": True,
        "json_output": False,
        "structured_output": False
    }
)

# Define a custom function to reverse a string
def reverse_string(text: str) -> str:
    """Reverse the given text.
        input : str

        output : str 
        The reversed String is returned.
    """
    return text[::-1]


# Register the custom function as a tool
reverse_tool = FunctionTool(
    reverse_string,
    description='A tool to reverse a string'
)

# Create an agent with the custom tool
agent = AssistantAgent(
    name="ReverseAgent",
    model_client=model_client,
    system_message="You are a helpful assistant that can reverse text using the reverse_string tool. Give the result with the summary",
    tools=[reverse_tool],
    reflect_on_tool_use=True,
)

# Define a task
task = "Reverse the text 'Hello, how are you Doing?'"

# Run the agent
async def main():
    result = await agent.run(task=task)

    # print(result)    
    print(f"Agent Response: {result.messages[-1].content}")

    print(reverse_string('Hello, how are you Doing?'))

if __name__ == "__main__":
    asyncio.run(main())