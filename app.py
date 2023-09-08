import os
import tempfile
import streamlit as st
import pickle
from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter

st.set_page_config(page_title="Chatbot via PDF", page_icon="🦜")
st.title("Chatbot via PDF")


class StreamlitChatbotApp():
    def __init__(self,openai_api_key):
        self.openai_api_key = openai_api_key

    def run(self):

        openai_api_key = self.openai_api_key
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        uploaded_files = st.sidebar.file_uploader(
            label="Upload PDF files", type=["pdf"], accept_multiple_files=True
        )
        if not uploaded_files:
            st.info("Please upload PDF documents to continue.")
            st.stop()

        retriever = self.configure_retriever(uploaded_files)

        # Setup memory for contextual conversation
        msgs = StreamlitChatMessageHistory()
        memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

        # Setup LLM and QA chain
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, temperature=0, streaming=True
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm, retriever=retriever, memory=memory, verbose=True
        )

        if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
            msgs.clear()
            msgs.add_ai_message("How can I help you?")

        avatars = {"human": "user", "ai": "assistant"}
        for msg in msgs.messages:
            st.chat_message(avatars[msg.type]).write(msg.content)

        if user_query := st.chat_input(placeholder="Ask me anything!"):
            st.chat_message("user").write(user_query)

            with st.chat_message("assistant"):
                retrieval_handler = PrintRetrievalHandler(st.container())
                stream_handler = StreamHandler(st.empty())
                response = qa_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])

    @st.cache_resource(ttl="1h")
    def configure_retriever(self,uploaded_files):
        # Read documents
        docs = []

        for file in uploaded_files:
    
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=500,
                length_function=len
                )
            chunks = text_splitter.split_text(text=text)


            store_name = file.name[:-4]
            st.write(f'{store_name}')

            if os.path.exists(f"{store_name}.pkl"):
                with open(f"{store_name}.pkl", "rb") as f:
                    VectorStore = pickle.load(f)
            else:
                embeddings = OpenAIEmbeddings(openai_api_key = openai_api_key )
                VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
                with open(f"{store_name}.pkl", "wb") as f:
                    pickle.dump(VectorStore, f)

        # Define retriever
        retriever = VectorStore.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})

        return retriever


class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.write("**AI Answer**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")


if __name__ == "__main__":


    openai_api_key =  os.environ.get('OpenAI_key')
    app = StreamlitChatbotApp(openai_api_key=openai_api_key)
    app.run()
