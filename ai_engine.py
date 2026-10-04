"""
PERSONAL AI ENGINE (agent) - built on the Claude API

What it does:
  You describe what you want. The AI plans, then uses tools to list/read/write
  files and run commands inside a safe "workspace" folder, checks its own work,
  and keeps going until the task is done.

Setup:
  1. pip install anthropic
  2. Get a key at https://platform.claude.com (Settings -> API keys)
  3. Set it:   export ANTHROPIC_API_KEY="sk-ant-..."     (Windows: set ANTHROPIC_API_KEY=sk-ant-...)
  4. Run:      python ai_engine.py

Safety built in:
  - The AI can only touch files inside ./workspace
  - Every command needs your approval (y/n) before it runs
  - Dangerous commands are blocked
  - NEVER put private keys or seed phrases in the workspace
"""

import subprocess
from pathlib import Path

import anthropic

# ---------------------------------------------------------------- settings
MODEL = "claude-sonnet-5-5"          # stronger: "claude-opus-5-5" | cheaper: "claude-haiku-4-5-20251001"
MAX_TOKENS = 4000
MAX_STEPS = 25                        # max tool steps per request (prevents endless loops)
WORKSPACE = Path("workspace").resolve()
WORKSPACE.mkdir(exist_ok=True)

BLOCKED = ["sudo", "rm -rf /", "mkfs", "shutdown", "reboot", ":(){", "dd if=",
           "seed phrase", "private key", "mnemonic"]

SYSTEM_PROMPT = """You are my personal AI engineer and technology assistant.
I am a blockchain developer and the founder of Efikcoin.

How you work:
- For any task: make a short plan, then use your tools to do it step by step.
- Write complete, working, well-commented code. Save it as files in the workspace.
- Run tests or the code to check it works, read any errors, and fix them.
- When finished, summarize what you built, where the files are, and how to run them.

Rules:
- Use testnets only. Never deploy to mainnet or move real funds.
- Never ask for, read, or write private keys, seed phrases, or API keys. Use a .env file placeholder instead.
- Always point out security risks in smart contracts and say when a professional audit is needed.
- If a request is unclear or risky, ask me before proceeding."""

# ---------------------------------------------------------------- tools
TOOLS = [
    {
        "name": "list_files",
        "description": "List all files and folders inside the workspace (or a subfolder).",
        "input_schema": {
            "type": "object",
            "properties": {"folder": {"type": "string", "description": "Subfolder path, '.' for the root"}},
            "required": [],
        },
    },
    {
        "name": "read_file",
        "description": "Read the text content of a file in the workspace.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path relative to the workspace"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a file in the workspace with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the workspace"},
                "content": {"type": "string", "description": "Full file content"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_command",
        "description": "Run a shell command inside the workspace (for example: python app.py, npm install, "
                       "npx hardhat test, forge test). The user must approve each command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
]


def safe_path(relative: str) -> Path:
    """Make sure the AI can never leave the workspace folder."""
    p = (WORKSPACE / relative).resolve()
    if p != WORKSPACE and WORKSPACE not in p.parents:
        raise ValueError("Path is outside the workspace.")
    return p


def clip(text: str, limit: int = 10000) -> str:
    return text if len(text) <= limit else text[:limit] + "\n...[output cut]..."


def run_tool(name: str, args: dict) -> str:
    try:
        if name == "list_files":
            base = safe_path(args.get("folder", "."))
            items = sorted(str(p.relative_to(WORKSPACE)) for p in base.rglob("*")
                           if "node_modules" not in p.parts and ".git" not in p.parts)
            return clip("\n".join(items)) or "(workspace is empty)"

        if name == "read_file":
            return clip(safe_path(args["path"]).read_text(encoding="utf-8"))

        if name == "write_file":
            p = safe_path(args["path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"], encoding="utf-8")
            print(f"  [wrote file] {p.relative_to(WORKSPACE)}")
            return f"Saved {p.relative_to(WORKSPACE)}"

        if name == "run_command":
            cmd = args["command"]
            if any(bad in cmd.lower() for bad in BLOCKED):
                return "BLOCKED: this command is not allowed."
            print(f"\n  [AI wants to run] {cmd}")
            if input("  Allow? (y/n): ").strip().lower() != "y":
                return "The user denied this command. Try a different approach or ask the user."
            result = subprocess.run(cmd, shell=True, cwd=WORKSPACE, capture_output=True,
                                    text=True, timeout=180)
            return clip(f"exit code: {result.returncode}\n{result.stdout}\n{result.stderr}")

        return f"Unknown tool: {name}"
    except subprocess.TimeoutExpired:
        return "Command timed out after 180 seconds."
    except Exception as e:  # return errors to the AI so it can fix them itself
        return f"Error: {e}"


# ---------------------------------------------------------------- the engine loop
def run_agent(client: anthropic.Anthropic, messages: list) -> None:
    """Keep calling Claude and running its tools until it has finished the task."""
    for _ in range(MAX_STEPS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        for block in response.content:
            if block.type == "text" and block.text.strip():
                print(f"\nAI: {block.text}")

        if response.stop_reason != "tool_use":
            return  # finished

        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  [tool] {block.name}")
                output = run_tool(block.name, block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})

    print("\n[Stopped: reached the step limit. Tell the AI to continue if needed.]")


def main() -> None:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from your environment
    messages: list = []

    print("=" * 60)
    print(" PERSONAL AI ENGINE")
    print(f" Workspace folder: {WORKSPACE}")
    print(" Commands: /reset = new conversation, /quit = exit")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user_input:
            continue
        if user_input == "/quit":
            break
        if user_input == "/reset":
            messages.clear()
            print("Conversation cleared.")
            continue

        messages.append({"role": "user", "content": user_input})
        try:
            run_agent(client, messages)
        except anthropic.APIError as e:
            print(f"\n[API error] {e}")
            messages.pop()  # remove the failed request so you can try again


if __name__ == "__main__":
    main()
