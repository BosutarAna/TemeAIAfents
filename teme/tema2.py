"""Tema 2 - Agent simplu pentru cautare in proiect"""

import os
import json
import subprocess
from dotenv import load_dotenv
from openai import OpenAI

# Incarca variabilele din .env
load_dotenv()

# Creeaza clientul pentru OpenRouter
client = OpenAI(
    base_url=os.getenv("OPEN_ROUTER_BASE_URL"),
    api_key=os.getenv("OPEN_ROUTER_API_KEY")
)

# Definirea uneltelor (tools) pentru agent
tools = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Lists all files in the project directory recursively"
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_keyword",
            "description": "Searches for a keyword in all project files",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "The keyword to search for"
                    }
                },
                "required": ["keyword"]
            }
        }
    }
]


def list_files():
    """Listeaza toate fisierele din proiect folosind dir /s /b."""
    result = subprocess.run(
        ["dir", "/s", "/b"],
        capture_output=True,
        text=True,
        shell=True
    )
    return result.stdout


def search_keyword(keyword: str):
    """Cauta un cuvant cheie in toate fisierele folosind findstr."""
    result = subprocess.run(
        ["findstr", "/s", "/i", keyword, "*.*"],
        capture_output=True,
        text=True,
        shell=True
    )
    return result.stdout


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Executa unealta ceruta de agent si returneaza rezultatul."""
    if tool_name == "list_files":
        return list_files()
    elif tool_name == "search_keyword":
        return search_keyword(tool_input["keyword"])
    else:
        return f"Unknown tool: {tool_name}"


def run_agent(user_message: str) -> str:
    """Ruleaza agentul: trimite mesajul, executa unelte, returneaza raspunsul."""
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the available tools to help the user."
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    while True:
        # Trimite mesajul la LLM
        response = client.chat.completions.create(
            model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
            messages=messages,
            tools=tools,
            tool_choice="auto"
        ).choices[0].message

        # Adauga raspunsul in conversatie
        messages.append(response.model_dump(exclude_none=True))

        # Daca LLM vrea sa foloseasca tools
        if response.tool_calls:
            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}

                print(f"Agent apeleaza: {tool_name}({args})")

                # Executa tool-ul
                output = execute_tool(tool_name, args)

                # Trimite rezultatul inapoi
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": output
                })

            # LLM proceseaza rezultatul
            response = client.chat.completions.create(
                model=os.getenv("OPEN_ROUTER_MODEL_NAME"),
                messages=messages,
                tools=tools
            ).choices[0].message
            messages.append(response.model_dump(exclude_none=True))

        # Daca nu mai sunt tool calls, returneaza raspunsul
        if not response.tool_calls:
            return response.content


if __name__ == "__main__":
    result = run_agent(
        "List all files in this project, then search for the keyword 'pytest'. "
        "Tell me which files contain this keyword."
    )
    print("\n=== RASPUNS FINAL ===")
    print(result)
