import requests

def chat_with_ollama(messages):
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.2",
            "messages": messages,
            "stream": False
        }
    )
    return response.json()["message"]["content"]


# This list holds the entire conversation history
conversation = []

print("Chat with your local AI (type 'quit' to exit)")
print("-" * 40)

while True:
    user_input = input("You: ")

    if user_input.lower() == "quit":
        break

    # Add the user's message to the conversation history
    conversation.append({"role": "user", "content": user_input})

    # Send the WHOLE conversation so far, not just the new message
    reply = chat_with_ollama(conversation)

    # Add the model's reply to the history too, so it's remembered next turn
    conversation.append({"role": "assistant", "content": reply})

    print(f"AI: {reply}")