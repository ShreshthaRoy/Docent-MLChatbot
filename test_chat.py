import requests

response = requests.post(
    "http://localhost:11434/api/chat",
    json={
        "model": "llama3.2",
        "messages": [
            {"role": "user", "content": "Say hello in one sentence."}
        ],
        "stream": False
    }
)

print(response.status_code)
print(response.json())