import os
import base64
import gc
import tempfile
import uuid

from IPython.display import display
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import streamlit as st

# Initialize session state for unique session identification and file caching
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4()
    st.session_state.document_cache = {}

session_id = st.session_state.session_id

# Load language model with a caching mechanism
@st.cache_resource
def initialize_llm():
    return Ollama(model="llama3.2:3b-instruct-q8_0", request_timeout=120.0)

# Reset the chat state and clear garbage
def reset_chat_state():
    st.session_state.chat_history = []
    gc.collect()

# Display PDF in a sidebar with embedded preview
def render_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_html = f"""
        <iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="100%" type="application/pdf"
                style="height:100vh;">
        </iframe>
    """
    st.markdown(pdf_html, unsafe_allow_html=True)

# Sidebar for document upload and processing
with st.sidebar:
    st.header("Upload a Document")
    uploaded_file = st.file_uploader("Select a PDF file", type="pdf")

    if uploaded_file:
        try:
            # Temporarily store uploaded file for processing
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                # Save uploaded file
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Create a unique key for the file to manage state
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Processing and indexing document...")

                # Process and cache the document if it hasn't been processed already
                if file_key not in st.session_state.document_cache:
                    if os.path.exists(temp_dir):
                        loader = SimpleDirectoryReader(
                            input_dir=temp_dir,
                            required_exts=[".pdf"],
                            recursive=True
                        )
                    else:
                        st.error('File not found. Please check and try again.')
                        st.stop()

                    # Load document data
                    documents = loader.load_data()

                    # Initialize LLM and embedding model
                    llm = initialize_llm()
                    embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-large-en-v1.5", trust_remote_code=True)

                    # Create an index over the loaded data
                    Settings.embed_model = embedding_model
                    document_index = VectorStoreIndex.from_documents(documents, show_progress=True)
                    
                    # Create a query engine using the LLM
                    Settings.llm = llm
                    query_engine = document_index.as_query_engine(streaming=True)

                    # Define custom prompt template for QA
                    qa_prompt_template_str = (
                        "Context information is provided below.\n"
                        "---------------------\n"
                        "{context_str}\n"
                        "---------------------\n"
                        "Based on the context, please respond concisely. "
                        "If the answer is unknown, respond with 'I don't know'.\n"
                        "Query: {query_str}\n"
                        "Answer: "
                    )
                    qa_prompt_template = PromptTemplate(qa_prompt_template_str)
                    query_engine.update_prompts(
                        {"response_synthesizer:text_qa_template": qa_prompt_template}
                    )

                    # Cache the query engine for future use
                    st.session_state.document_cache[file_key] = query_engine
                else:
                    # Load query engine from cache
                    query_engine = st.session_state.document_cache[file_key]

                # Confirm document is ready for interaction and display the PDF
                st.success("Document successfully indexed. Ready for queries!")
                render_pdf(uploaded_file)

        except Exception as e:
            st.error(f"Error processing document: {e}")
            st.stop()

# Main chat interface layout
col1, col2 = st.columns([6, 1])

with col1:
    st.header("Document Chat Interface")

with col2:
    st.button("Clear Chat", on_click=reset_chat_state)

# Initialize chat history
if "chat_history" not in st.session_state:
    reset_chat_state()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input for queries
if user_input := st.chat_input("Enter your question:"):
    # Append user's query to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process the query and display LLM response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Stream the LLM's response
        query_result = query_engine.query(user_input)
        for chunk in query_result.response_gen:
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    # Add the assistant's response to chat history
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
