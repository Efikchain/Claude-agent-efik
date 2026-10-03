"""
Simple personal AI assistant using the Claude API.

Setup:
  pip install anthropic
  export ANTHROPIC_API_KEY="your-key-here"   (Windows: set ANTHROPIC_API_KEY=your-key-here)
  python efik_assistant.py
"""

import anthropic

# Reads the key from the ANTHROPIC_API_KEY environment variable (safer than pasting it in code)
client = anthropic.Anthropic()

MODEL = "claude-sonnet-5-5"

SYSTEM_PROMPT = """You are my personal AI engineer and assistant.
I am a blockchain developer and the founder of Efikcoin.
Help me design, write, test, and review smart contracts and blockchain code,
plan product features, and write documentation.
Always point out security risks and tell me when code needs a professional audit."""

history = []

print("Efikcoin assistant ready. Type 'quit' to exit.\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() in ("quit", "exit"):
        break
    if not user_input:
        continue

    history.append({"role": "user", "content": user_input})

    print("\nAI: ", end="", flush=True)
    reply = ""
    with client.messages.stream(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=history,
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            reply += text
    print("\n")

    # Keep the conversation so the AI remembers earlier messages
    history.append({"role": "assistant", "content": reply})
