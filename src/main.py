import os
import json
import base64
import numpy as np
import scipy.io.wavfile as wav

from typing import Dict, Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from starlette import status

from pydantic import BaseModel, Field

# langchain & langgraph imports
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

#sqlalchemy
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, ForeignKey, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
from datetime import datetime


# Your OpenAI client
from openai import OpenAI


client = OpenAI()

Base = declarative_base()

DATABASE_URL = "sqlite:///./conversations.db"

# 1. Create the engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Only needed for SQLite
    echo=True  # Optional; logs SQL to console, good for debugging
)

# 2. Create a Base class for your models
Base = declarative_base()

Base.metadata.create_all(engine)
# 3. Create a configured "SessionLocal" class
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

class Thread(Base):
    __tablename__ = "threads"

    id = Column(String, primary_key=True, index=True)
    title = Column(String, nullable=True)  # Optionally store a human-readable title
    created_at = Column(DateTime, default=datetime.utcnow)

    # One-to-many: A thread can contain multiple messages
    messages = relationship("Message", back_populates="thread", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Thread(id={self.id}, title={self.title})>"

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(Integer, ForeignKey("threads.id"), nullable=False)
    role = Column(String, nullable=False)   # e.g., "user", "assistant", "system"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationship back to Thread
    thread = relationship("Thread", back_populates="messages")

    def __repr__(self):
        return f"<Message(thread_id={self.thread_id}, role={self.role})>"



def decode_base64_to_array(base64_string: str, dtype: np.dtype = np.int16) -> np.ndarray:
    """
    Decodes a base64 string back into a NumPy array.

    Args:
        base64_string (str): The base64-encoded string containing audio data.
        dtype (np.dtype, optional): Data type of the returned array. Defaults to np.int16.

    Returns:
        np.ndarray: A numpy array containing the audio samples.
    """
    decoded_bytes = base64.b64decode(base64_string)
    return np.frombuffer(decoded_bytes, dtype=dtype)

def save_audio_to_wav(audio_array: np.ndarray, samplerate: int, filename: str = "output.wav") -> None:
    """
    Saves the decoded audio array to a .wav file using scipy.io.wavfile.write.

    Args:
        audio_array (np.ndarray): Numpy array of audio samples.
        samplerate (int): The sample rate of the audio data.
        filename (str, optional): The name of the .wav file to save. Defaults to "output.wav".
    """
    wav.write(filename, samplerate, audio_array)
    print(f"Audio saved to {filename}")

def get_transcription_audio_file(filename: str = "output.wav") -> str:
    """
    Transcribes audio data from a WAV file using the OpenAI Whisper API.

    Args:
        filename (str, optional): Path to the .wav file. Defaults to "output.wav".

    Returns:
        str: The transcribed text.
    """
    with open(filename, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="en"
        )
    return transcription.text

def add_message_sql(thread_id: str, content: str, role: str):
    session: Session = SessionLocal()
    thread = session.query(Thread).filter_by(id=thread_id).first()
    if not thread:
        thread = Thread(id=thread_id, title="Conversation Title")
        session.add(thread)
        session.commit()

    user_message = Message(
        thread_id=thread.id,
        role=role,
        content=content
    )
    session.add(user_message)
    session.commit()

    session.close()

def get_thread_by_id(thread_id):
    """
    Retrieves a thread and all its messages from the database.
    """
    session = SessionLocal()
    try:
        # Actually query the database for the thread
        thread = session.query(Thread).filter_by(id=thread_id).first()
        if thread:
            print(f"Thread ID: {thread.id}, Title: {thread.title}")
            for message in thread.messages:
                print(f"[{message.role}] {message.content} (created at {message.created_at})")
        else:
            print("No thread found with ID:", thread_id)
    finally:
        session.close()

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

def get_response_llm(messages: str, config: Dict[str, Any]) -> str:
    """
    Gets the response from the LLM via the compiled graph.

    Args:
        user_input (str): The text input from the user.
        config (Dict[str, Any]): Configuration for the graph execution (including thread ID).

    Returns:
        str: The final response from the LLM.
    """
    events = app_graph.stream(
        {"messages": messages}, config, stream_mode="values"
    )

    response = ""
    for event in events:
        # The last message in the conversation is the AI response
        response = event["messages"][-1].content

    return response

# FastAPI setup
app = FastAPI()

class Prompt(BaseModel):
    """
    Pydantic model representing the prompt to be sent to the chatbot and the user's thread.
    """
    prompt: str = Field(..., title="Prompt", description="The prompt to send")
    thread: str = Field("abc123", title="Thread", description="The thread of the user")

@app.get("/", status_code=status.HTTP_200_OK)
async def home() -> Dict[str, str]:
    """
    A simple route to test if the API is running.

    Returns:
        Dict[str, str]: A simple greeting message.
    """
    return {"message": "Hello world"}

@app.get("/healthy", status_code=status.HTTP_200_OK)
async def healthy() -> Dict[str, str]:
    """
    A simple route to check the health of the service.

    Returns:
        Dict[str, str]: A dictionary containing the health status.
    """
    return {"status": "healthy"}

@app.post("/chatbot", status_code=status.HTTP_200_OK)
async def handle_prompt(data: Prompt) -> Dict[str, str]:
    """
    Handle incoming requests with the user's prompt and return the AI's response.

    Args:
        data (Prompt): The prompt and thread information provided by the user.

    Returns:
        Dict[str, str]: A JSON containing the AI's response.
    """
    user_input: str = data.prompt
    thread: str = data.thread
    add_message_sql(thread, user_input, "user")

    config = {"configurable": {"thread_id": thread}}

    try:
        session = SessionLocal()
        messages = []
        db_thread = session.query(Thread).filter_by(id=thread).first()
        if db_thread:
            # Thread exists => load its prior messages in ascending creation time
            # (you may want to sort by Message.created_at for chronological order)
            for msg in db_thread.messages:
                messages.append((msg.role, msg.content))
        response = get_response_llm(messages, config)
        add_message_sql(thread, response, "assistant")
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

    return {"response": response}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket endpoint that handles real-time audio input for transcription and response.

    Args:
        websocket (WebSocket): The WebSocket connection instance.

    Returns:
        None
    """
    await websocket.accept()
    print("Client connected")

    try:
        while True:
            try:
                # Receive the JSON message from the client
                message: str = await websocket.receive_text()
                data: Dict[str, Any] = json.loads(message)

                # Extract audio and metadata
                thread: str = data.get("user_id", "abc123")
                base64_encoded_array: str = data.get("audio_data", "")
                samplerate: int = data.get("samplerate", 16000)
                channels: int = data.get("channels", 1)

                # Decode base64 audio and save it
                decoded_array: np.ndarray = decode_base64_to_array(base64_encoded_array)
                if channels > 1:
                    decoded_array = decoded_array.reshape(-1, channels)
                save_audio_to_wav(decoded_array, samplerate)

                # Transcribe audio
                user_input = get_transcription_audio_file()
                os.remove("output.wav")

                # Stream events from app_graph
                config = {"configurable": {"thread_id": thread}}
                response: str = get_response_llm(user_input, config)

                # Validate the response
                if not response:
                    raise ValueError("No response content generated.")

                # Send the response back to the WebSocket client
                await websocket.send_text(json.dumps({"status": "success", "text_generated": response}))

            except WebSocketDisconnect:
                # Client disconnected
                print("Client disconnected")
                break  # Exit the loop cleanly

            except ValueError as ve:
                # Handle specific validation errors
                print(f"Validation error: {ve}")
                await websocket.send_text(json.dumps({"status": "error", "message": str(ve)}))

            except Exception as e:
                # Handle other unexpected errors
                print(f"Unexpected error: {e}")
                await websocket.send_text(json.dumps({"status": "error", "message": "Internal server error."}))

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    finally:
        print("WebSocket connection closed")


if __name__ == "__main__":
    import uvicorn
    get_thread_by_id("abc123")
    uvicorn.run(app, host="127.0.0.1", port=8000)
