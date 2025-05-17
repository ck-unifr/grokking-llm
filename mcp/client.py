from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import asyncio

# model = ChatOpenAI(model="gpt-4o")
model = ChatOpenAI(
    model="glm-4-flash", openai_api_base="https://open.bigmodel.cn/api/paas/v4/"
)


server_params = StdioServerParameters(
    command="python",
    args=["mcp_server.py"],
)


async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            tools = await load_mcp_tools(session)

            print(f"Tools: {tools}")
        
            # 创建并执行agent
            agent = create_react_agent(model, tools)
            agent_response = await agent.ainvoke({"messages": "(3 + 5) * 12 等于？"})
            return agent_response


# Run the async function
if __name__ == "__main__":
    result = asyncio.run(run_agent())
    print(result)
