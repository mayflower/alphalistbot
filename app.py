"""
A simple slack bot that does retrieval using a vector store \
and displays answers with sources.
"""
import os
from pathlib import Path
import langchain
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.cache import InMemoryCache
from langchain.vectorstores.faiss import FAISS

from dotenv import load_dotenv

load_dotenv()

langchain.llm_cache = InMemoryCache()


if not Path("store/index.pkl").exists():
    raise ValueError("store/index.pkl existiert nicht, "
                     "bitte führen Sie zunächst 'python import.py' aus")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", default="")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"

OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", default="gpt-3.5-turbo")
assert OPENAI_API_MODEL, "OPENAI_API_MODEL environment variable is missing from .env"

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", default="")
assert SLACK_BOT_TOKEN, "SLACK_BOT_TOKEN environment variable is missing from .env"

SLACK_BOT_KEYWORD = os.getenv("SLACK_BOT_KEYWORD", default="documentbot")
assert SLACK_BOT_KEYWORD, "SLACK_BOT_KEYWORD environment variable is missing from .env"

SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN", default="")
assert SLACK_APP_TOKEN, "SLACK_APP_TOKEN environment variable is missing from .env"


llm = ChatOpenAI(
    temperature=0.1,
    model_name=OPENAI_API_MODEL,
)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=OPENAI_API_MODEL)
vectorstore = FAISS.load_local('./store/', embeddings)
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm, chain_type="stuff",
    retriever=retriever)


history = []

app = App(
    token=SLACK_BOT_TOKEN,
)

@app.message(SLACK_BOT_KEYWORD)
def message_hello(message, say):
    """Main loop  - get questions as an slackbot event and reply."""
    question = message['text'].replace(SLACK_BOT_KEYWORD,'')
    print(f"Question {question}")
    answer = qa_chain({"question":question, "chat_history": history})["answer"]
    print(answer)
    say(
        text=f"<@{message['user']}>: {answer}"
    )
    history.append((question, answer))

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
