from dotenv import load_dotenv
from openai import OpenAI
import json as JSON
import requests as request
import os
load_dotenv()

def cli_agent(cmd:str):
    result = os.system(cmd)
    return result

def get_weather(city:str):

    url = f"https://wttr.in/{city}?format=%C+%T"
    response = request.get(url)
    if response.status_code == 200:
        return response.text.strip()
    else:
        return "Could not retrieve weather data."
    
Available_tools = {
        "get_weather": get_weather,
        "cli_agent": cli_agent
    }

SYSTEM_MESSASGE = """You are a AI agent that provides current weather information for a given city and also can execute command line instructions.
When asked about the weather you should call the get_weather function with the city name as an argument.
This will be your output format START(user input), THINK(your reasoning), TOOL(function name and arguments), OUTPUT(final answer).
 Finally return a JSON OBJECT with the following OUTPUT format.

 Allowed tools:
 - get_weather: retrieves the current weather for a specified city.
 - cli_agent: executes a command line instruction and returns the result.

 Rules:
 - Used allowed tools: get_weather, cli_agent
 - Strictly followed the given JSON OUTPUT format.
 - Only run one step at a time
 - The output sequence should be: START, THINK, TOOL, OUTPUT
 - Take City as Input for get_weather tool.

 Output Format:
   {"step": "START" or "THINK" or "TOOL" or "OUTPUT",
    "message": "your message here"
   }

 Example 1:
 START: What is the weather in London?
 THINK: I need to find the current weather in London.
 THINK: I will use the get_weather tool to retrieve the weather information.
 TOOL: {"tool":"get_weather", "input":"London"}
 OUTPUT: {"city": "London", "weather": "Partly cloudy, 15°C"}

 Example 2:
 START: Can you tell me the weather in New York?
 THINK: I need to find the current weather in New York.
 THINK: I will use the get_weather tool to retrieve the weather information.
 TOOL: {"tool":"get_weather", "input":"New York"}
 OUTPUT: {"city": "New York", "weather": "Sunny, 22°C"}

"""
USER_INPUT = input("Please type your query 👉🏻 ")

client = OpenAI()

mesages = [
    {"role": "system", "content": SYSTEM_MESSASGE},
    {"role": "user", "content": USER_INPUT}
]
while True:
    client_response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=mesages
    )

    raw_result = client_response.choices[0].message.content

    try:
        parsed_result = JSON.loads(raw_result)
        mesages.append({"role": "assistant", "content": raw_result})
    except JSON.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response: {raw_result}")
        break

    get_step = parsed_result.get("step","")

    if get_step == "START":
        print(f"User Input: {parsed_result.get('message')}")
    elif get_step == "THINK":
        print(f"Thinking: {parsed_result.get('message')}")
    elif get_step == "TOOL":
        message = parsed_result.get("message", "")
        
        # Parse the tool information - message should contain JSON with tool and input
        try:
            # The message itself might be a dict or a JSON string
            if isinstance(message, dict):
                tool_data = message
            elif isinstance(message, str):
                # Try to parse as JSON, if it fails, it might be a plain message
                try:
                    tool_data = JSON.loads(message)
                except JSON.JSONDecodeError:
                    # If message is just a description, get tool from parsed_result directly
                    tool_data = {"tool": parsed_result.get("tool"), "input": parsed_result.get("input")}
            
            tool_info = tool_data.get("tool")
            tool_input = tool_data.get("input", "")
            
            print(f"Using Tool: {tool_info} with input: {tool_input}")
            
            if tool_info and tool_info in Available_tools:
                # Call the tool with the city string directly
                tool_response = Available_tools[tool_info](tool_input)
                print(f"Tool Response: {tool_response}")
                mesages.append({"role": "developer", "content": f"Tool Response: {tool_response}"})
            else:
                print(f"Tool '{tool_info}' not found in available tools")
                break
        except Exception as e:
            print(f"Error executing tool: {e}")
            print(f"Parsed result: {parsed_result}")
            break

    elif get_step == "OUTPUT":
        print(f"Final Output: {parsed_result.get('message')}")
        break

