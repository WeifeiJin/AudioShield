import os
import time
import warnings
from google.cloud import speech_v1 as speech
from google.api_core.exceptions import GoogleAPICallError, RetryError
import requests


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./api/google_token.json"
if not os.path.exists("./api/google_token.json"):
    print('api-key error!')

client = speech.SpeechClient()


config = speech.RecognitionConfig(
    encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
    sample_rate_hertz=16000,
    language_code="en-US",
)

def network_connection():
    try:
        response = requests.get('https://speech.googleapis.com/v1/speech:recognize')
        print(f"Network test response status: {response.status_code}")
    except Exception as e:
        print(f"Network connection error: {e}")

def google_ASR(audio_file):
    max_retries = 1
    retry_count = 0
    result = ''

    while retry_count < max_retries:
        try:
            with open(audio_file, 'rb') as f:
                byte_wave_f = f.read()
                audio_wav = speech.RecognitionAudio(content=byte_wave_f)
                # print('Audio file read successfully.')


                response = client.recognize(config=config, audio=audio_wav, timeout=90)
                # print('Received response from Google ASR.')


                if response.results:
                    for response_result in response.results:
                        result = response_result.alternatives[0].transcript.upper()
                    break
                else:
                    print(f"No recognition result from Google ASR (attempt {retry_count + 1})")

        except GoogleAPICallError as e:
            print(f"Google API call error in attempt {retry_count + 1}: {e}")
        except RetryError as e:
            print(f"Retry error in attempt {retry_count + 1}: {e}")
        except Exception as e:
            print(f"Error in attempt {retry_count + 1}: {e}")

        retry_count += 1
        time.sleep(5)

    if result == '':
        result = 'NA'

    # print(f'Google ASR Result after {retry_count} retries: {result}')
    return result

if __name__ == "__main__":
    network_connection()

    audio_file = "./benign_audio/p225_001.wav"
    result = google_ASR(audio_file)
    print(result)