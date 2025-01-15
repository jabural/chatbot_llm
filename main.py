"""
This code provides an API for interacting with a chatbot built using a transformer model.
It uses OpenAI and LangGraph for processing and managing
conversational states.
"""
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from starlette import status
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_openai import ChatOpenAI

memory = MemorySaver() #Define memory

#Initialize the graph
graph_builder = StateGraph(MessagesState)

#Generate tool for being able to search the internet
tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatOpenAI(model="gpt-4o-mini")   #Load llm model
llm_with_tools = llm.bind_tools(tools)  #Bind tools to model

def chatbot(state: MessagesState):
    """
    Define chatbot node that calls the llm model.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

#Add nodes to graph
graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

app_graph = graph_builder.compile(checkpointer=memory)

app = FastAPI()

class Prompt(BaseModel):
    """
    Pydantic model representing the prompt to be sent to the chatbot and the user's thread.
    """
    prompt: str = Field(..., title="Prompt", description="The prompt to send")
    thread: str = Field("abc123", title="Thread", description="The thread of the user")


@app.get("/", status_code=status.HTTP_200_OK)
async def home():
    """
    A simple route to test if the API is running.
    """
    return {"message": "Hello world"}

@app.get("/healthy", status_code=status.HTTP_200_OK)
async def healthy():
    """
    A simple route to test if the API is running.
    """
    return {"status": "healthy"}


@app.post("/chatbot", status_code=status.HTTP_200_OK)
async def handle_prompt(data: Prompt):
    """
    Handle incoming requests with the user's prompt and return the AI's response.
    
    Args:
        data (Prompt): The prompt and thread information provided by the user.

    Returns:
        response (str): The AI's generated response.
    """
    user_input = data.prompt
    thread = data.thread

    config = {"configurable": {"thread_id": f"{thread}"}}

    try:
        # Attempt to stream events from the app graph
        events = app_graph.stream(
            {"messages": [("user", user_input)]}, config, stream_mode="values"
        )
    except Exception as e:
        # Handle errors during event streaming
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error occurred while processing the request: {str(e)}"
        )
    
    event = None
    try:
        # Extract the last event from the streamed events
        for event in events:
            pass

        # Check if the event is valid and contains the response content
        if event is None or not event.get("messages") or not event["messages"][-1].content:
            raise ValueError("Invalid event or missing content in the response.")
        
        # Extract the response content
        response = event["messages"][-1].content

    except ValueError as ve:
        # Handle specific value-related errors
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Value error: {str(ve)}"
        )
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error occurred: {str(e)}"
        )
    
    response = event["messages"][-1].content

    return {"response": response}
