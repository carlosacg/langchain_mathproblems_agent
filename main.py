import os

from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.chains.llm_math.base import LLMMathChain
from langchain.memory import ConversationBufferMemory
from langchain.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_openai import ChatOpenAI


def load_keys():
    """Load API keys from the .env file."""
    load_dotenv()
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY is missing in .env")
    return openai_key


def create_tools(llm):
    """Create the list of tools for the agent: Wikipedia and math calculator."""
    tools = []
    wiki = WikipediaAPIWrapper()
    tools.append(WikipediaQueryRun(api_wrapper=wiki))
    llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=False)
    math_tool = Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="Solve math questions using an LLM.",
    )
    tools.append(math_tool)
    return tools


def create_agent(llm, tools):
    """Initialize the agent with conversation memory."""
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
    )
    return agent


def main():
    openai_key = load_keys()
    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_key)
    tools = create_tools(llm)
    agent = create_agent(llm, tools)
    print("\nAgent ready. Type your question (Ctrl+C to exit):\n")
    try:
        while True:
            question = input("> ")
            answer = agent.run(question)
            print(f"\nAnswer: {answer}\n")
    except KeyboardInterrupt:
        print("\nProgram terminated.")


if __name__ == "__main__":
    main()
