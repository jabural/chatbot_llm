import base64
import numpy as np
import scipy.io.wavfile as wav


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
