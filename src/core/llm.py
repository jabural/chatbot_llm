from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Dict, Any
from ..models import Thread, Message
from sqlalchemy.orm import Session
from langchain.schema.runnable import RunnableConfig
from typing import cast
import asyncio

from openai import OpenAI

client = OpenAI()


async def get_transcription_audio_file(filename: str = "output.wav") -> str:
    """
    Transcribes audio data from a WAV file using the OpenAI Whisper API.

    Args:
        filename (str, optional): Path to the .wav file. Defaults to "output.wav".

    Returns:
        str: The transcribed text.
    """
    with open(filename, "rb") as audio_file:
        transcription = await asyncio.to_thread(lambda: client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en"
        ))
    return transcription.text


def add_message_sql(db: Session, thread_id: str, content: str, role: str):
    thread = db.query(Thread).filter_by(id=thread_id).first()
    if not thread:
        thread = Thread(id=thread_id, title="Conversation Title")
        db.add(thread)
        db.commit()

    user_message = Message(
        thread_id=thread.id,
        role=role,
        content=content
    )
    db.add(user_message)
    db.commit()


def get_thread_by_id(db: Session, thread_id: str):
    """
    Retrieves a thread and all its messages from the database.
    """
    # Actually query the database for the thread
    thread = db.query(Thread).filter_by(id=thread_id).first()
    if thread:
        print(f"Thread ID: {thread.id}, Title: {thread.title}")
        for message in thread.messages:
            print(f"[{message.role}] {message.content} (created at {message.created_at})")
    else:
        print("No thread found with ID:", thread_id)


# Initialize the state graph and chatbot
graph_builder = StateGraph(MessagesState)

tool = TavilySearchResults(max_results=2)
tools = [tool]
llm = ChatOpenAI(model="gpt-4o-mini")   # Load LLM
llm_with_tools = llm.bind_tools(tools)  # Bind tools to model


def chatbot(state: MessagesState) -> Dict[str, Any]:
    """
    Define chatbot node that calls the LLM model.

    Args:
        state (MessagesState): The current conversation state.

    Returns:
        Dict[str, Any]: A dictionary containing updated conversation messages.
    """
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

app_graph = graph_builder.compile()


async def get_response_llm(messages: list[tuple[Any, Any]], config: Dict[str, Any]) -> str:
    """
    Gets the response from the LLM via the compiled graph.

    Args:
        user_input (str): The text input from the user.
        config (Dict[str, Any]): Configuration for the graph execution (including thread ID).

    Returns:
        str: The final response from the LLM.
    """
    runnable_config = cast(RunnableConfig, config)

    events = await asyncio.to_thread(
        lambda: list(app_graph.stream({"messages": messages}, runnable_config, stream_mode="values"))
        )

    response = ""
    for event in events:
        # The last message in the conversation is the AI response
        response = event["messages"][-1].content

    return response
