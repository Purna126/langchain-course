"""Minimal LangChain demo: load env, prompt template, ChatOpenAI, invoke chain."""

from pathlib import Path

from dotenv import dotenv_values, load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model 


import os

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)


def main() -> None:
    env_values = dotenv_values(ENV_PATH)
    api_key = env_values.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"OPENAI_API_KEY not found in {ENV_PATH}")
        return
    information = """
   Tell me something about the ongoing Data engineering trends.
    """
    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    """
    prompt = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )
    llm = ChatOpenAI(temperature=0, model="gpt-5")
    chain = prompt | llm
    response = chain.invoke(input={"information": information})
    print(response.content)
    
    """model = init_chat_model("openai:gpt-5.4")
    response = model.invoke("Hello, world!")
    print(response.content)"""


if __name__ == "__main__":
    main()
