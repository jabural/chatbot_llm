# Chatbot API with LangGraph and OpenAI

This project provides a FastAPI-based API for interacting with a chatbot built using the LangGraph library and OpenAI's GPT model. It includes functionality for processing conversational states and integrating internet search results into chatbot interactions via the Tavily search tool.

![screenshot](images/graph.png)


## Features
- FastAPI-based chatbot service.
- Conversational state management using LangGraph's `StateGraph`.
- Integration of OpenAI's GPT-4 mini model via the `langchain-openai` package.
- Ability to perform internet searches using the Tavily search tool.
- Memory-saving capabilities with SQLite.
- Conditional routing in the conversation flow based on tool usage.
- Graphical interface created with gradio.

## Prerequisites

Before running this project, ensure that you have the following installed:
- Python 3.10 or higher
- Docker (for containerization)

## Installation

To get started with the project, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/chatbot-api.git
cd chatbot-api
```
### 2. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
### 3. Install dependencies
```bash
pip install -r requirements.txt
```
### 4. Set environment variables
In order to be able to communicate with Tavily and openai the user needs first to obtain some api keys and set them as environment variables.
```bash
export OPENAI_API_KEY=xyz
export TAVILY_API_KEY=abc
```
### 5. Run the application
Once everything is installed the application can be ran.
```bash
uvicorn main:app --reload
```

## Run the dockerize application (Optional)
Instead of installing everything, another option is to use the compose file to run the dockerized image.
It is still needed to have the api keys set as environment values.
```bash
docker-compose up --build
```

## 6. Testing the API

The API has the following routes:

### 1. Home Route

**GET /**  
A simple route to test if the API is running.

**Response:**

```json
{
  "message": "Hello world"
}
```

### 2. Home Route

**GET /healthy**  
A simple route to test if the API is running.

**Response:**

```json
{
  "message": "Hello world"
}
```
### 3. History Route

**GET /history**  
Retrieves all messages from a specific thread based on the thread_id query parameter.

**Request:**

```
GET /history/?thread_id=abc123
```

**Response:**
If the thread exists:

```json
{
  "thread_id": "thread_id",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?",
      "created_at": "2024-06-01T12:00:00"
    },
    {
      "role": "assistant",
      "content": "I'm doing well, thank you!",
      "created_at": "2024-06-01T12:00:05"
    }
  ]
}
```

If the thread does not exist:

```json
{
  "detail": "No thread found with ID: <therad_id>"
}
```

### 4. Chatbot interaction

**POST /chatbot**  
Send a prompt to the chatbot along with a user thread to get a response. The prompt is sent to the AI model, which processes the input and returns a response.
The thread is optional, and one would be assigned if it is not chosen.

**Request body:**

```json
{
  "prompt": "Hello, how are you?",
  "thread": "abc123"
}
```


**Response:**

```json
{
  "response": "I'm doing well, thank you!"
}

```


### 5. Websocket Endpoint
The API also provides a Websocket endpoint for real-time interaction, such as audio processing and live chatbot responses.

**Enpoint:** ```ws```
This WebSocket endpoint allows clients to send audio data for transcription and receive chatbot-generated responses in real time.

**Workflow:**
1. The client connects to the WebSocket endpoint.

2. The client sends a JSON payload containing the audio data and metadata.

3. The server processes the audio, transcribes it, and generates a chatbot response.

4. The server sends the response back to the client in real time.

**Example JSON Payload:**
```json
{
  "user_id": "abc123",
  "audio_data": "<base64-encoded-audio>",
  "samplerate": 16000,
  "channels": 1
}
```

**Server Response:**

Upon a succesful processing:
```json
{
  "status": "success",
  "text_generated": "Hello! How can I assist you today?"
}
```

In case of errors:
```
{
  "status": "error",
  "message": "<Error description>"
}
```

## Graphical interface
In order to test it, a python code located in ```src/client_testing/client_interface.py``` is left. This code generates a graphical interface that the user
can apply to send audio and text to the model to see how well it responds.

![screenshot](images/gradio.png)
