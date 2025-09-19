# wyoming_openai_tts_gateway
Deal with dependency installation
Edit the Wyoming server host and port in code, then start the server. uvicorn main:app --host 0.0.0.0 --port 8555

- Еndpoint http://x.x.x.x:8555/v1/audio/speech
- The model field can have any value.
- View available voices http://x.x.x.x:8555/v1/voices

Example of use in ST

<img width="701" height="259" alt="image" src="https://github.com/user-attachments/assets/bb42e209-0a01-4e4a-b520-e8f34347ecb6" />
