import streamlit as st
import base64
import urllib.parse
import re
import logging
import os
import requests
from io import BytesIO
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
import streamlit.components.v1 as components
import time  # For simulating the progress bar
import asyncio

try:
    from pilot import chat_with_llm_stream, get_search_results_sync, reformulate_question,get_search_results_async
    from prompts import DOCSEARCH_PROMPT, QUESTION_GENERATOR_PROMPT
    from dotenv import load_dotenv
    load_dotenv("credentials.env")
    #for local testing print every credential

except Exception as e:
    print(e)
    # if not found in the current directory, try the parent directory
    import sys
    sys.path.append("../../common")
    from pilot import (chat_with_llm_stream, get_search_results_sync, reformulate_question,get_search_results_async)
    from prompts import DOCSEARCH_PROMPT, QUESTION_GENERATOR_PROMPT
    sys.path.append("../../")
    from dotenv import load_dotenv
    load_dotenv(r"C:\Users\bakklandmoil\GPT-Azure-Search-Engine\credentials.env")
    #for local testing print every credential




# Set page configuration
# Set page configuration
st.set_page_config(page_title="APL Smart Search", page_icon="📖", layout="wide", )
st.set_option("client.toolbarMode", "viewer")
st.logo("APLNOV1.png")

# Add custom CSS styles
st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 0rem;
    }
    .feedback-link {
        font-size: 12px;
        text-align: right;
        margin-top: 1.5rem;
    }
        .st-emotion-cache-4z1n4l.en6cib65 {
        visibility: hidden;
        height: 0px;
    }
        /* Hides the element with data-testid="InputInstructions" */
    [data-testid="InputInstructions"] {
        display: none;
    }
}
    </style>
    """, unsafe_allow_html=True)

# Header and Feedback Link
with st.container():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("APL Smart Search")
    with col2:
        st.markdown(
            """<div class="feedback-link"><a href="https://engage.cloud.microsoft/main/groups/eyJfdHlwZSI6Ikdyb3VwIiwiaWQiOiIyMDMwOTM1NTcyNDgifQ/all">Give Feedback</a></div>""",
            unsafe_allow_html=True
        )


# Button functions for improved clarity and code readability
def handle_search_button():
    st.session_state['is_running'] = True
    st.session_state["submitSearch"] = True
    st.session_state["doneStreaming"] = False

def handle_stop_button():
    st.session_state['is_running'] = False
    st.session_state["submitSearch"] = False

def handle_clear_button():
    st.session_state["doneStreaming"] = False
    st.session_state["submitSearch"] = False
    st.session_state.query = ""
    st.session_state.show_results = False

# Session State Variables
def clear_submit():
    st.session_state["submit"] = True

if "submitSearch" not in st.session_state:
    st.session_state["submitSearch"] = False
if 'is_running' not in st.session_state:
    st.session_state['is_running'] = False
if "reformulated_questions" not in st.session_state:
    st.session_state.reformulated_questions = []
    st.session_state.show_suggestions = False
if "query" not in st.session_state:
    st.session_state.query = ""
if "doneStreaming" not in st.session_state:
    st.session_state.doneStreaming = False
if "stored_answer" not in st.session_state:
    st.session_state.stored_answer = ""
if "stored_results" not in st.session_state:
    st.session_state.stored_results = []
if "Error" not in st.session_state:
    st.session_state.Error = None

# Sidebar instructions
with st.sidebar:
    st.markdown("# App Instructions")
    st.markdown("""
### How to Use APL Smart Search
The APL AI Smart Search tool provides answers exclusively from the uploaded documents, not from the internet or the chatbot’s internal knowledge. 
### Example Questions:
- Make an exhaustive requirement list on bolts.
- What are the inspection requirements for welded joints?
- Is there a 3.1 material certificate requirement for flexible hose end fittings?
    """)

# Main search area
col_input, col_select, col_button = st.columns([2, 1, 0.5], vertical_alignment="bottom")

def update_query():
    st.session_state.query = st.session_state.query_input

with col_input:
    query = st.text_input("Ask A Question About Your Selected Project", value=st.session_state.query, on_change=update_query, help="Enter your question below", key="query_input")

with col_select:
    index_mapping = {
        "2304 SLT Alto Tirreno": "srch-index-alto-tirreno",
        "2371 STP Bay du Nord": "srch-index-bay-du-nord",
        "2269 STP B29 Polok & Chinwol": "srch-index-polok-chinwol",
        "test": "srch-index-test1"
    }
    selected_label = st.selectbox("Choose What Project To Search In", list(index_mapping.keys()), help="Select the project to search in")
    selected_index = index_mapping[selected_label]

with col_button:
    if st.session_state['is_running']:
        st.button("Stop", on_click=handle_stop_button, help="Click to stop the search", type="primary")
    else:
        st.button("Search", on_click=handle_search_button, help="Click to start a search", type="secondary")


@st.fragment
def reformulate_fragment(llm):
    if st.button('Reformulate Question', help="Click to reformulate the question"):
        empty_comtext = []
        empty_value = {"chunk": "Empty", "score": 0.00, "location": "empty"}
        empty_comtext.append(
            Document(
                page_content=empty_value["chunk"], 
                metadata={"source": empty_value["location"], "score": empty_value["score"]}
            )
        )
        rfq = reformulate_question(llm, query, empty_comtext)
        if rfq:
            st.session_state.reformulated_questions = rfq
            st.session_state.show_suggestions = True
        else:
            st.session_state.reformulated_questions = ["Error, try again"] * 4
            st.session_state.show_suggestions = True

@st.fragment
def display_suggestions():
    if st.session_state.show_suggestions:
        row1_col1, row1_col2 = st.columns([1, 1])
        columns = [row1_col1, row1_col2]
        for i, suggestion in enumerate(st.session_state.reformulated_questions):
            with columns[i % len(columns)]:
                if st.button(f"{suggestion}", on_click=update_query_from_suggestion, args=(suggestion,)):
                    update_query_from_suggestion(suggestion)

@st.fragment
def search_fragment(llm, selected_index, selected_label, search_container):
    if st.session_state["submitSearch"]:
        try:
            if not st.session_state.query or st.session_state.query.strip() == "":
                st.error("Please enter a valid question!")
                st.session_state.is_running = False
                st.write(st.session_state.is_running)  # Dynamically show that the search is not running

            else:
                ordered_results = {}
                try:
                    k = 6
                    st.session_state['is_running'] = True  # Indicate the search is running
                    with st.spinner(f"Searching {selected_label}..."):
                        ordered_results = get_search_results_async(st.session_state.query, [selected_index], k=k, reranker_threshold=1, sas_token=os.environ['BLOB_SAS_TOKEN'])
                        
                        st.session_state["submitSearch"] = True
                        st.session_state["doneStreaming"] = False
                        st.session_state.stored_results = ordered_results
                        #st.write(st.session_state.stored_results)
                except Exception as e:
                    st.error("No data returned from Azure Search. Please check the connection.")
                    st.session_state['is_running'] = False
                    st.session_state["submitSearch"] = False
                    st.session_state["doneStreaming"] = True
                    st.session_state["Error"] = "No data returned. Refine query and try again."
                    logging.error(f"Search error: {e}")
                    st.write(f"Error during search: {st.session_state.Error}")  # Display error
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logging.error(f"Unexpected error: {e}")


@st.fragment
def answer_fragment(llm,answer_container):
    if st.session_state.stored_results and st.session_state["doneStreaming"] == False:
        top_docs = []
        for key, value in st.session_state.stored_results.items():
            location = value.get("location", "")
            top_docs.append(Document(page_content=value.get("chunk", ""), metadata={"source": location, "score": value.get("score", 0)}))
        
        if top_docs:
            with st.spinner("Generating answer..."):
                try:
                    # Create a placeholder for the streaming content
                    answer_placeholder = answer_container.empty()
                    
                    # Clear previous output
                    answer_placeholder.empty()

                    # Accumulate the result of the generator in a list
                    streamed_answer = []
                    
                    def stream_output(content):
                        # Accumulate streamed content and also display it
                        streamed_answer.append(content)
                        answer_placeholder.write(content)

                    # Stream the response
                    for chunk in chat_with_llm_stream(DOCSEARCH_PROMPT, llm, st.session_state.query, top_docs):
                        stream_output(chunk)

                    # Store the full answer after streaming completes
                    st.session_state.stored_answer = ''.join(streamed_answer)  # Join the accumulated output
                    st.session_state.doneStreaming = True
                except Exception as e:
                    st.error("Error generating answer.")
                    logging.error(f"LLM error: {e}")

def display_results(answer_container, search_container):
    # We want to handle cases where both answer and results exist
    if st.session_state.stored_answer and st.session_state.stored_results is not None:
        # If streaming is done and we are not running, or running status is True, display
        if st.session_state["doneStreaming"] and st.session_state["is_running"]:
            with answer_container:
                answer_container.empty()  # Clear previous answer
                if st.session_state.stored_answer:
                    st.markdown("#### Answer")
                    st.markdown("---")
                    st.markdown(st.session_state.stored_answer, unsafe_allow_html=True)

            if st.session_state.stored_results:
                with search_container:
                    st.markdown("#### Search Results")
                    for key, value in st.session_state.stored_results.items():
                        location = value.get("location", "")
                        title = str(value.get('title', value.get('name', 'Unnamed Document')))
                        score = str(round(value.get('score', 0) * 100 / 4, 2))
                        final_output = f"{location}"
                        text = f"{value.get('name', 'Unnamed Document')} {value.get('page', '')}"
                        st.markdown(f"**Document**: [{text}]({final_output})")
                        st.markdown(f"**Score**: {score}%")
                        st.markdown(value.get("caption", "No caption available"))
                        st.markdown("---")

            # Mark as completed, and reset flags
            st.session_state["is_running"] = False
            st.session_state["submitSearch"] = False
            st.session_state["doneStreaming"] = True
            # Trigger a rerun to ensure state updates and button resets after the display
            st.rerun()

    # Regardless, check to ensure the UI updates the button states on every run
        elif not st.session_state["is_running"]:
                with answer_container:
                    answer_container.empty()  # Clear previous answer
                    if st.session_state.stored_answer:
                        st.markdown("#### Answer")
                        st.markdown("---")
                        st.markdown(st.session_state.stored_answer, unsafe_allow_html=True)

                if st.session_state.stored_results:
                    with search_container:
                        st.markdown("#### Search Results")
                        for key, value in st.session_state.stored_results.items():
                            location = value.get("location", "")
                            title = str(value.get('title', value.get('name', 'Unnamed Document')))
                            score = str(round(value.get('score', 0) * 100 / 4, 2))
                            final_output = f"{location}"
                            text = f"{value.get('name', 'Unnamed Document')} {value.get('page', '')}"
                            st.markdown(f"**Document**: [{text}]({final_output})")
                            st.markdown(f"**Score**: {score}%")
                            st.markdown(value.get("caption", "No caption available"))
                            st.markdown("---")



# Initialize the LLM
# Check required environment variables
required_env_vars = [
    "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "BLOB_SAS_TOKEN"
]
missing_env_vars = [var for var in required_env_vars if not os.environ.get(var)]
answer_container = st.container()
search_container = st.container()
if missing_env_vars:
    st.error(f"Please set the following environment variables: {', '.join(missing_env_vars)}")
else:
    os.environ["OPENAI_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"]
    MODEL = os.environ.get("GPT4_DEPLOYMENT_NAME")
    llm = AzureChatOpenAI(deployment_name=MODEL, temperature=0.2, max_tokens=2500)

    # Run fragments, passing the `llm` instance

    print(selected_index, selected_label)
    reformulate_fragment(llm)
    display_suggestions()
    search_fragment(llm, selected_index, selected_label,search_container)
    answer_fragment(llm,answer_container)  # Fragment for LLM interaction and answer generation
    display_results(answer_container,search_container)
