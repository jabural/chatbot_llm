from typing import Annotated
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from starlette import status
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


app = FastAPI()

class Prompt(BaseModel):
    prompt: str = Field(..., title="Prompt", description="The prompt to send to the chatbot")
    thread: str = Field("abc123", title="Thread", description="The thread of the user that is speaking")

model_name = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
conversation_history = []
# Create a pipeline for text generation
generation_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Wrap the pipeline in LangChain's HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=generation_pipeline)

prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know. The human question or input shouldn't be generated, only the first AI response.",
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
)

workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    prompt = prompt_template.invoke(state)
    response = llm.invoke(prompt)
    return {"messages": response}


workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

@app.get('/', status_code=status.HTTP_200_OK)
async def home():
    return {"message": "Hello world"}

@app.post('/chatbot', status_code=status.HTTP_200_OK)
async def handle_prompt(data: Prompt):
    print(data)  # DEBUG
    input_text = data.prompt  + "\n" # Extract the 'prompt' key
    thread = data.thread

    config = {"configurable": {"thread_id": f"{thread}"}}

    input_messages = [HumanMessage(input_text)]
    output = app_graph.invoke({"messages": input_messages}, config)

    response = output["messages"][-1].content.split("AI: ")[-1]

    return {"response": response}
