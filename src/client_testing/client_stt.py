import os
import asyncio
import numpy as np
import sounddevice as sd
import websockets
import json
import base64
from gtts import gTTS
import soundfile as sf

async def record_audio(duration, samplerate=44100, channels=1):
    """
    Records audio for the specified duration and returns it as a NumPy array.
    """
    print(f"Recording for {duration} seconds...")
    # Create an empty buffer to hold the audio data
    buffer = np.zeros((int(samplerate * duration), channels), dtype=np.int16)

    loop = asyncio.get_event_loop()
    event = asyncio.Event()
    idx = 0

    def callback(indata, frame_count, time_info, status):
        nonlocal idx
        if status:
            print(f"Status: {status}")
        remainder = len(buffer) - idx
        if remainder <= 0:
            loop.call_soon_threadsafe(event.set)
            raise sd.CallbackStop
        indata = indata[:remainder]
        buffer[idx:idx + len(indata)] = indata
        idx += len(indata)

    # Use an input stream to capture audio
    stream = sd.InputStream(callback=callback, dtype=np.int16, channels=channels, samplerate=samplerate)
    with stream:
        await event.wait()

    print("Recording complete.")
    return buffer

async def play_buffer(buffer, **kwargs):
    loop = asyncio.get_event_loop()
    event = asyncio.Event()
    idx = 0

    def callback(outdata, frame_count, time_info, status):
        nonlocal idx
        if status:
            print(status)
        remainder = len(buffer) - idx
        if remainder == 0:
            loop.call_soon_threadsafe(event.set)
            raise sd.CallbackStop
        valid_frames = frame_count if remainder >= frame_count else remainder
        outdata[:valid_frames] = buffer[idx:idx + valid_frames]
        outdata[valid_frames:] = 0
        idx += valid_frames

    stream = sd.OutputStream(callback=callback, dtype=buffer.dtype,
                             channels=buffer.shape[1], **kwargs)
    with stream:
        await event.wait()


def encode_array_to_base64(array):
    """
    Encodes a NumPy array to a base64 string.
    """
    array_bytes = array.tobytes()  # Convert the array to bytes
    base64_encoded = base64.b64encode(array_bytes).decode("utf-8")  # Encode to base64 string
    return base64_encoded


async def connect_to_server(audio_data, samplerate, channels):
    """
    Connects to a WebSocket server, sends the audio data, and receives a response.
    """
    uri = "ws://127.0.0.1:8000/ws"
    async with websockets.connect(uri) as websocket:
        print("Connected to server")

        try:
            # Prepare metadata with audio data and sampling info
            base64_encoded_audio = encode_array_to_base64(audio_data)
            message = {
                "audio_data": base64_encoded_audio,
                "samplerate": samplerate,
                "channels": channels,
                "user_id": "abc"
            }

            # Send the JSON message to the server
            json_data = json.dumps(message)
            await websocket.send(json_data)
            # print(f"Sent to server: {json_data}")

            response = await websocket.recv()
            data = json.loads(response)
            print(f"Response from server: {data}")
            tts = gTTS(text=data["text_generated"], lang='en')

            # Save the audio file
            tts.save("audio_obtained.wav")
            data, rate = sf.read('audio_obtained.wav')

            # Play the audio file
            sd.play(data, rate)

            # Wait for the sound to finish playing
            sd.wait()

            os.remove("audio_obtained.wav")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Audio recording parameters
    DURATION = 3  # in seconds
    SAMPLERATE = 16000
    CHANNELS = 1

    # Record and send audio
    audio_data = asyncio.run(record_audio(DURATION, samplerate=SAMPLERATE, channels=CHANNELS))
    flattened_audio = audio_data.flatten()  # Flatten the array for transmission
    asyncio.run(connect_to_server(flattened_audio, SAMPLERATE, CHANNELS))
