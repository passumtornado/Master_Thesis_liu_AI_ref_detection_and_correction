from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage
import dotenv
import os
dotenv.load_dotenv()

llm = ChatOllama( model="qwen3-coder:480b-cloud",
            base_url="https://ollama.com",
            temperature=0.1,
            client_kwargs={
                "headers": {"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY')}"}
            },)
prompt = [HumanMessage(content="What is the capital of France?")]
llm.invoke(prompt)