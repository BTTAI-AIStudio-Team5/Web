from openai import OpenAI
import dotenv
import os
from .master_prompt import strips_prompt, final_prompt

dotenv.load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def world_to_strips(world):
    # world_input = input("Please input your world: ")
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": f"{strips_prompt}: {world}"}],
        max_tokens=1024
    )
    strips = response.choices[0].message.content
    print(strips)


def strips_to_NL(strips):
    response = client.chat.completions.create(
        model="gpt-4", 
        messages=[{"role": "user", "content": f"{final_prompt}: {strips}"}],
        max_tokens=2048
    )
    return response.choices[0].message.content
