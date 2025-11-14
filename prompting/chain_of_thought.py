from dotenv import load_dotenv
from openai import OpenAI
import json as JSON
load_dotenv()

client = OpenAI()


# Chain of Thought prompting involves guiding the AI to think through problems step-by-step before arriving at a final answer.

SYSTEM_MESSAGE="""
You are an AI expert. Answer questions in a step-by-step manner to reach the final answer.

Rules:
- Return ONLY valid JSON format, no other text
- Use this exact JSON structure for each response
- Progress through steps: START -> PLAN -> OUTPUT
- Each response should be a single JSON object

Response Format:
{
  "step": "START" or "PLAN" or "OUTPUT",
  "message": "your message here",
  "is_answered": true or false
}

Example 1 - Math Problem:
User: "solve this particular problem 2+3*6+10/3"

Response 1:
{
  "step": "START",
  "message": "Solving the expression: 2+3*6+10/3",
  "is_answered": false
}

Response 2:
{
  "step": "PLAN",
  "message": "1. Apply order of operations (PEMDAS). ",
  "is_answered": false
}
Response 3:
{
"step::" PLAN",
"message": "2. 2+3*6+10/3 = 2+18+3.33 = 23.33",
"is_answered": false
}
Response 4:
{
"step":"PLAN",
"message":"3. Summarize the result.",
"is_answered": false
}
Response 5:
{
  "step": "OUTPUT",
  "message": "The result is 23.33 (2 + 18 + 3.33 = 23.33)",
  "is_answered": true
}
"""
print("\n\n\n")
user_input = input("Please enter your query: ")
message = [
    {"role":"system","content":SYSTEM_MESSAGE},
]
message.append({"role":"user","content":user_input})

while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=message,
        response_format={"type": "json_object"}
    )

    raw_result = response.choices[0].message.content
    message.append({"role":"assistant","content":raw_result})
    
    try:
        parsed_result = JSON.loads(raw_result)
    except JSON.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw response: {raw_result}")
        break

    step = parsed_result.get("step", "")
    
    if step == "START":
        print("Thinking...", parsed_result.get("message"))
        continue
    elif step == "PLAN":
        print("Planning...", parsed_result.get("message"))
        continue
    elif step == "OUTPUT":
        print("Final Answer:", parsed_result.get("message"))
        break
    else:
        print("Unknown step or completed:", parsed_result)
        break

print("\n\n\n")