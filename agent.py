from llm import llm
from graph import graph
from tools.vector import get_vector_info
from tools.cypher import cypher_qa
from utils import get_session_id
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.chat_message_histories import Neo4jChatMessageHistory

# Create a business chat chain
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a business expert providing information about companies."),
        ("human", "{input}"),
    ]
)

chat = chat_prompt | llm | StrOutputParser()

# Create a set of tools
from langchain.tools import Tool

tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat not covered by other tools",
        func=chat.invoke,
    ), 
    Tool.from_function(
        name="Mortage Calculator",
        description="Provide information about the mortgage calculator using Cypher",
        func = cypher_qa,
        return_direct=False
    ), 
    Tool.from_function(
        name="Banks",  
        description="For when you need to find information about Allied Irish Banks",
        func=get_vector_info, 
    )
]

# Create chat history callback
def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)


# Create the agent
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import PromptTemplate

agent_prompt = PromptTemplate.from_template("""
Be as helpful as possible and return as much information as possible.
Don't use your own knowledge.

TOOLS:
------

You have access to the following tools:
{tools}
To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")


agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
    )

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

# Create a handler to call the agent
def generate_response(user_input):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

    return response['output']