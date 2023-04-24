""" Import script to spider several data sources with urls and safe them to a faiss vectorstore. """
import os
from pathlib import Path
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import GoogleApiClient, GoogleApiYoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv
load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", default="")
assert OPENAI_API_KEY, "OPENAI_API_KEY environment variable is missing from .env"

OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", default="gpt-3.5-turbo")
assert OPENAI_API_MODEL, "OPENAI_API_MODEL environment variable is missing from .env"

GOOGLE_CREDENTIALS = os.getenv("GOOGLE_CREDENTIALS", default="")
assert GOOGLE_CREDENTIALS, "GOOGLE_CREDENTIALS environment variable is missing from .env"

YOUTUBE_CHANNEL = os.getenv("YOUTUBE_CHANNEL", default="alphalist1658")
assert YOUTUBE_CHANNEL, "YOUTUBE_CHANNEL environment variable is missing from .env"


def import_docs():
    """
    Imports the documents from a Youtube channel and saves them to a vector store. 
    """
    google_api_client = GoogleApiClient(credentials_path=Path(GOOGLE_CREDENTIALS))
    youtube_loader = GoogleApiYoutubeLoader(
        google_api_client=google_api_client,
        channel_name=YOUTUBE_CHANNEL,
        #video_ids=["EkkHBqm6oPg"],
        continue_on_failure=True,
        captions_language="en",
        add_video_info=True)

    documents = youtube_loader.load()

    for document in documents:
        video_id =  document.metadata["videoId"]
        document.metadata["source"] =f"https://www.youtube.com/watch?v={video_id}"
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 2000,
        chunk_overlap  = 50,
        length_function = len,
    )
    documents = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model=OPENAI_API_MODEL)
    vectorstore = FAISS.from_documents(documents=documents, embedding=embeddings)
    vectorstore.save_local('./store/')


if __name__ == "__main__":
    import_docs()
