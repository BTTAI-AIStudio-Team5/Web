from gpt4all import GPT4All
from .master_prompt import strips_prompt, final_prompt

model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf")  # downloads / loads a 4.66GB LLM


def world_to_strips(world):
    # world_input = input("Please input your world: ")
    with model.chat_session():
        strips = model.generate(
            f"{strips_prompt}: {world}",
            max_tokens=1024,
        )
        print(strips)


def strips_to_NL(strips):
    with model.chat_session():
        final = model.generate(
            f"{final_prompt}: {strips}",
            max_tokens=2048,
        )
        return final
