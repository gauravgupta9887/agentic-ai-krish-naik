from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

import os
import asyncio
from dotenv import load_dotenv
load_dotenv()


custom_llm = ChatOpenAI(
    model="gemma-local",
    api_key=os.getenv("CHAT_API_KEY"),
    base_url=os.getenv("CHAT_API_BASE_URL"),
    temperature=0,
    verbose=True
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


async def main():
    client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": [os.path.join(SCRIPT_DIR, "mathserver.py")],
                "transport": "stdio"
            },
            "weather": {
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http"
            }
        }
    )

    tools = await client.get_tools()
    agent = create_react_agent(
        model=custom_llm,
        tools=tools
    )
    math_response = await agent.ainvoke(
        {"messages":  [{"role": "user", "content": "What is (3+5)*12?"}]}
    )

    print("Math response:", math_response["messages"][-1].content)
    
    weather_response = await agent.ainvoke(
        {"messages":  [{"role": "user", "content": "What is weather in jaipur?"}]}
    )
    print("Weather response:", weather_response["messages"][-1].content)

asyncio.run(main())
