"""
Code to generate an interface with gradio for our chatbot application
"""
import gradio as gr
import asyncio
import numpy as np
import websockets
import json
import base64
import requests

def encode_array_to_base64(array: np.ndarray) -> str:
    """
    Encodes a NumPy array to a base64 string.
    """
    array_bytes = array.tobytes()  # Convert the array to bytes
    base64_encoded = base64.b64encode(array_bytes).decode("utf-8")  # Encode to base64 string
    return base64_encoded


async def connect_to_server(audio_data=None, sample_rate=None, text_data=None, channels=1):
    """
    Connects to a WebSocket server, sends either audio or text input, and receives a response.
    """
    if audio_data is not None:
        uri = "ws://127.0.0.1:8000/ws"  # Change to your server's address
        async with websockets.connect(uri) as websocket:
            print("Connected to server")
            base64_encoded_audio = encode_array_to_base64(audio_data)
            message = {
                    "audio_data": base64_encoded_audio,
                    "samplerate": sample_rate,
                    "channels": channels,
                    "user_id": "example_user",
                    "input_type": "audio"
                }

            # Send the JSON message to the server
            json_data = json.dumps(message)
            await websocket.send(json_data)

            # Receive the JSON response
            response = await websocket.recv()
            data = json.loads(response)
            print(f"Response from server: {data}")
            return data.get("text_generated", "No text returned")

    elif text_data is not None:
        request_data = {
        'prompt': text_data,
        'thread': 'abc123'
        }
        response = requests.post("http://127.0.0.1:8000/chatbot", json=request_data, timeout=10)
        # Parse the response body as JSON
        response_data = response.json()
        # Extract the value of the 'response' key
        chatbot_response = response_data.get("response")
        print(f"Response from server: {chatbot_response}")
        return chatbot_response

    else:
        return "No data provided"

def process_input(audio_file, user_text, input_choice):
    """
    Processes the input based on the chosen type, sends it to the WebSocket server,
    and clears the input fields after submission.
    """
    # Audio branch
    if input_choice == "Audio":
        if audio_file is None:
            return "No audio received.", gr.update(value=None), gr.update(value="")
        sample_rate, audio_data = audio_file
        audio_data_flat = audio_data.flatten()  # Flatten in case the server expects a 1D array
        try:
            text_from_server = asyncio.run(
                connect_to_server(audio_data=audio_data_flat, sample_rate=sample_rate, channels=1)
            )
        except Exception as e:
            text_from_server = f"Error sending audio: {str(e)}"
        # Return the server response and clear both inputs
        return text_from_server, gr.update(value=None), gr.update(value="")

    # Text branch
    if input_choice == "Text":
        if not user_text:
            return "No text input provided.", gr.update(value=None), gr.update(value="")
        try:
            text_from_server = asyncio.run(
                connect_to_server(text_data=user_text)
            )
        except Exception as e:
            text_from_server = f"Error sending text: {str(e)}"
        return text_from_server, gr.update(value=None), gr.update(value="")

    else:
        return "Invalid input choice.", gr.update(value=None), gr.update(value="")


def toggle_input(choice):
    """
    Returns updates to set the visibility of the audio and text inputs.
    """
    if choice == "Audio":
        return gr.update(visible=True), gr.update(visible=False)
    return gr.update(visible=False), gr.update(visible=True)

with gr.Blocks() as demo:
    gr.Markdown("# Choose an Input Type")

    # Radio to choose input type: Audio or Text
    input_choice = gr.Radio(choices=["Audio", "Text"], value="Audio", label="Select Input Type")

    # Row holding both input components; only one will be visible at a time.
    with gr.Row():
        audio_input = gr.Audio(type="numpy", label="Record Audio")
        text_input = gr.Textbox(label="Enter Text", visible=False)

    submit_button = gr.Button("Submit")

    # Only one output: text returned from the server.
    output_text = gr.Textbox(label="Server Response", interactive=False)

    # When the radio selection changes, update the visibility of the input components.
    input_choice.change(fn=toggle_input, inputs=input_choice, outputs=[audio_input, text_input])

    # When submit is clicked, process the chosen input and clear the fields.
    # The processing function returns three values:
    # 1. The server response (to display in output_text)
    # 2. An update to reset the audio input field
    # 3. An update to reset the text input field
    submit_button.click(
        fn=process_input,
        inputs=[audio_input, text_input, input_choice],
        outputs=[output_text, audio_input, text_input]
    )

if __name__ == "__main__":
    demo.launch()
