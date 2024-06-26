import os
import streamlit as st
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Elasticsearch settings
ELASTICSEARCH_URL = "http://localhost:9200"
INDEX_NAME = "books"

# Initialize Elasticsearch client
es = Elasticsearch([ELASTICSEARCH_URL])

# App title
st.set_page_config(page_title="🤗💬 OpenAI Chat", layout="wide")
st.title("DHV AI Chatbot for AIA Claims Info")

# Sidebar language selection
with st.sidebar:
    st.title('🤗💬 OpenAI Chat')
    language = st.radio("Choose language", ("Thai", "English", "Japanese", "Chinese", "Arabic"))
    st.markdown('📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-an-llm-powered-chatbot-with-streamlit/)!')

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Load documents and index them into Elasticsearch
def index_documents():
    loader = DirectoryLoader("data/books", glob="*.md")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100, length_function=len, add_start_index=True)
    chunks = text_splitter.split_documents(documents)
    
    actions = [
        {
            "_index": INDEX_NAME,
            "_source": {
                "content": chunk.page_content,
                "metadata": chunk.metadata
            }
        }
        for chunk in chunks
    ]
    bulk(es, actions)
    print(f"Indexed {len(actions)} chunks into Elasticsearch.")

# Function to perform semantic search in Elasticsearch
def semantic_search(query):
    response = es.search(
        index=INDEX_NAME,
        body={
            "query": {
                "match": {
                    "content": query
                }
            }
        }
    )
    hits = response['hits']['hits']
    return hits

# Function for generating LLM response
def generate_response(prompt_input):
    try:
        # Perform semantic search
        search_results = semantic_search(prompt_input)
        if not search_results:
            return "Sorry, I couldn't find any relevant information."

        # Combine search results into context
        context = "\n\n".join(hit["_source"]["content"] for hit in search_results)
        
        # Generate response using context and prompt
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {prompt_input}"}
            ]
        )
        
        # Extract the generated response
        return response.choices[0].message["content"].strip()
    except Exception as e:
        st.error(f"Error: {e}")
        return "Sorry, I'm unable to generate a response at the moment."

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt)
                st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

# Index documents on startup
if __name__ == "__main__":
    index_documents()
