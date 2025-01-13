"""
This code provides an API for interacting with a chatbot built using a transformer model.
It uses HuggingFace's transformers pipeline and LangChain for processing and managing
conversational states.
"""
import os
from pydantic import BaseModel, Field
from fastapi import FastAPI
from starlette import status
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Get Hugging Face token from environment variables
hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

# Define the model constant in uppercase (as per PEP 8)
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=hf_token)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=hf_token)

# Create a pipeline for text generation
generation_pipeline = pipeline(
    "text-generation", model=model, tokenizer=tokenizer, max_new_tokens=20
)

# Wrap the pipeline in LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=generation_pipeline)

# Define the system message for the chatbot
prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            """asyncThe following is a friendly conversation between a human and an AI.
            The AI is talkative and provides lots of specific details from its context.
            If the AI does not know the answer to a question, it truthfully says it does not know.
            The human question or input shouldn't be generated, only the first AI response."""
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Initialize the workflow with a state graph
workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    """
    Calls the language model using the given state and returns a generated response.

    Args:
        state (MessagesState): The state that contains the conversation history.

    Returns:
        dict: The response from the language model.
    """
    print(f"State is: {state}")
    prompt = prompt_template.invoke(state)
    response = AIMessage(llm.invoke(prompt))
    return {"messages": response}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

# Initialize FastAPI app
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


@app.post("/chatbot", status_code=status.HTTP_200_OK)
async def handle_prompt(data: Prompt):
    """
    Handle incoming requests with the user's prompt and return the AI's response.
    
    Args:
        data (Prompt): The prompt and thread information provided by the user.

    Returns:
        response (str): The AI's generated response.
    """
    input_text = data.prompt + "\n"
    thread = data.thread

    config = {"configurable": {"thread_id": f"{thread}"}}

    input_messages = [HumanMessage(input_text)]
    output = app_graph.invoke({"messages": input_messages}, config)
    response = output["messages"][-1].content.split("AI: ")[-1]

    return {"response": response}
