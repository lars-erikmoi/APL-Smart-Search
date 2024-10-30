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
    from pilot import chat_with_llm_stream, get_search_results_sync, reformulate_question,get_search_results_async,chat_chain_with_llm, get_doc_from_blob
    from prompts import DOCSEARCH_PROMPT, QUESTION_GENERATOR_PROMPT
    from dotenv import load_dotenv
    load_dotenv("credentials.env")
    #for local testing print every credential

except Exception as e:
    print(e)
    # if not found in the current directory, try the parent directory
    import sys
    sys.path.append("../../common")
    from pilot import (chat_with_llm_stream, get_search_results_sync, reformulate_question,get_search_results_async,chat_chain_with_llm, get_doc_from_blob)
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
    .bordered-container {
        border: 1px solid #ccc;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
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



def process_documents(ordered_results):
    """
    Process the documents from the ordered results and return the top documents.
    """
    top_docs = []
    try:
        with st.spinner("Reading the source documents to provide the best answer... ⏳"):
            for key, value in ordered_results.items():
                if not st.session_state['is_running']:  # Check for stop
                    st.warning("Stopped processing documents.")
                    break
                if value.get("error"):
                    st.error(f"Error processing document: {value.get('error')}")
                    st.session_state['is_running'] = False
                    st.session_state["Error"] = f"Error processing document: {value.get('error')}"
                    break
                else:
                    location = value.get("location", "")
                    top_docs.append(Document(page_content=value.get("chunk", ""), metadata={"source": location, "score": value.get("score", 0)}))
    except Exception as e:
        st.error("Error processing documents.")
        logging.error(f"Document processing error: {e}")

    return top_docs


def process_documents_post(ordered_results):
    """
    Process the documents from the ordered results and return the top documents.
    """
    top_docs = []
    try:
        with st.spinner("Reading the source documents to provide the best answer... ⏳"):
            for key, value in ordered_results.items():
                location = value.get("location", "")
                top_docs.append(Document(page_content=value.get("chunk", ""), metadata={"source": location, "score": value.get("score", 0)}))
    except Exception as e:
        st.error("Error processing documents. could not fetch the documents")
        logging.error(f"Document processing error: {e}")

    return top_docs


def display_search_results(ordered_results):
    """
    Display the search results from the ordered results.
    """

    with search_container:
        if ordered_results is None:
            print("No search results to display.")
        else:
            
            st.markdown("---")
            st.markdown("#### Search Results")
            for key, value in ordered_results.items():
                location = value.get("location", "")
                title = str(value.get('title', value.get('name', 'Unnamed Document')))
                score = str(round(value.get('score', 0) * 100 / 4, 2))
                final_output = f"{location}"
                text = f"{value.get('name', 'Unnamed Document')} {value.get('page', '')}"
                st.markdown(f"**Document**: [{text}]({final_output})")
                st.markdown(f"**Score**: {score}%")
                st.markdown(value.get("caption", "No caption available"))
                st.markdown("---")


def display_answer(top_docs,spinner_placeholder):
    """
    Display the answer after processing the documents.
    """
    if len(top_docs) > 0:
        with answer_container:
            st.markdown("#### Answer")
            st.markdown("---")
            answer2 = chat_chain_with_llm(DOCSEARCH_PROMPT, llm, query, top_docs, 3, answer_container,spinner_placeholder)
            #answer2 = chat_with_llm_stream(DOCSEARCH_PROMPT, llm, query, top_docs, answer_container)
            st.session_state.stored_answer = answer2
    else:
        with answer_container:
            st.error("could not generate answer from the documents")

def main_search_flow(ordered_results, spinner_placeholder):
    """
    Encapsulates the main search flow, including processing documents and handling semantic search only.
    """
    if st.session_state['is_running'] and ordered_results:
        top_docs = process_documents(ordered_results)

        # Always display search results
        display_search_results(ordered_results)

        # Only display the answer if semantic search only is not enabled
        if not st.session_state['semantic_search_only']:
            display_answer(top_docs, spinner_placeholder)

    else:
        st.error("No valid results or search not started yet.")
# Button functions for improved clarity and code readability
def handle_search_button(semantic_only=False):
    # Start the search
    st.session_state['is_running'] = True
    st.session_state["submitSearch"] = True
    st.session_state["doneStreaming"] = False
    st.session_state.Error = None  # Clear error when new search starts¨
    st.session_state['semantic_search_only'] = semantic_only
    st.session_state['available_docs'] = []  # Clear available docs

    time.sleep(0.1)


def handle_stop_button():
    # Stop the search
    st.session_state['is_running'] = False
    st.session_state["submitSearch"] = False

def handle_clear_button():
    # Clear the results
    st.session_state["doneStreaming"] = False
    st.session_state["submitSearch"] = False
    st.session_state.query = ""
    st.session_state.show_results = False

# Buttons display logic


#### Session State Variables ####
def clear_submit():
    st.session_state["submit"] = True

if "submitSearch" not in st.session_state:
    st.session_state["submitSearch"] = False

if 'is_running' not in st.session_state:
    st.session_state['is_running'] = False

if 'available_docs' not in st.session_state:
    st.session_state['available_docs'] = []


if 'semantic_search_only' not in st.session_state:
    st.session_state['semantic_search_only'] = True
if "reformulated_questions" not in st.session_state:
    st.session_state.reformulated_questions = []
    st.session_state.show_suggestions = False 

# Function to update the main query input
def update_query_from_suggestion(suggestion):
    st.session_state.query = suggestion
    # Clear reformulated questions and hide suggestions
    st.session_state.reformulated_questions = []
    st.session_state.show_suggestions = False

# Function to toggle visibility of reformulated suggestions
def toggle_suggestions():
    st.session_state.show_suggestions = False

if "query" not in st.session_state:
    st.session_state.query = ""

if 'is_expanded' not in st.session_state:
    st.session_state['is_expanded'] = False

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
The APL AI Smart Search tool provides answers exclusively from the uploaded documents, not from the internet or the chatbot’s internal knowledge. If the system doesn’t find the information, it will simply say: "I don't know."
If the top answer isn't helpful, you can explore the additional search results below for more search hits.
                
### Example Questions:

- Make an exhaustive requirement list on bolts.
- Make an exhaustive requirement list on tubing.
- What are the inspection requirements for welded joints?
- Is there a 3.1 material certificate requirement for flexible hose end fittings?
- How should flexible hoses be marked?
- What bolt grade should I use for the piping system?
- What are the warranty terms in the client contract?
- What documents do clients request for review?
- What does the contract say about liquidated damages?
                
### Feedback
                
Your feedback is crucial for improvement. If the search didn't find information that you later discovered was actually there, please share the search query, the search results, and the information you expected to find using the feedback button in the top right corner. This helps us make necessary improvements.
Also, please share any success stories if this tool helped you in any way. It's the best way to demonstrate the value of investing in tools like this.

    """)

# Main search area
col_input, col_select = st.columns([2, 1],vertical_alignment="bottom")

# Text input field
with col_input:
    query = st.text_input("Ask A Question About Your Selected Project", value=st.session_state.query, on_change=clear_submit, help="Enter your question below")

# Selectbox for index selection
with col_select:
    index_mapping = {
        "2304 SLT Alto Tirreno": "srch-index-alto-tirreno",
        "2371 STP Bay du Nord": "srch-index-bay-du-nord",
        "2269 STP B29 Polok & Chinwol": "srch-index-polok-chinwol",
        "test": "srch-index-test1"
    }
    selected_label = st.selectbox("Choose What Project To Search In", list(index_mapping.keys()), help="Select the project to search in")
    selected_index = index_mapping[selected_label]


col_button1,col_button2 ,_, col_k = st.columns([2,2,2, 3],vertical_alignment="top",)

if st.session_state.get('is_running', False):

    # Display Stop button in the first column
    with col_button1:
        st.button("Stop", on_click=handle_stop_button, help="Stop the ongoing search", type="primary", key="stop_button")

else:
    # Display Regular Search and Semantic-only Search buttons in closer columns
    with col_button1:

        st.button("Search + Ai-answer", on_click=handle_search_button, 
                  args=(False,), help="Start a search with both semantic search and ChatGPT responses",
                  type="secondary", key="regular_search_button",)
    with col_button2:  # Directly adjacent column for Semantic-only
        st.button("Search", on_click=handle_search_button, 
                  args=(True,), help="Start a semantic search without ChatGPT responses",
                  type="secondary", key="semantic_only_search_button")

@st.cache_data(ttl=3600, show_spinner=False)
def get_cached_docs(index_name):
    return get_doc_from_blob(index_name)

# Check required environment variables
required_env_vars = [
    "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "BLOB_SAS_TOKEN"
]
missing_env_vars = [var for var in required_env_vars if not os.environ.get(var)]

if missing_env_vars:
    st.error(f"Please set the following environment variables: {', '.join(missing_env_vars)}")
else:
    os.environ["OPENAI_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"]
    MODEL = os.environ.get("GPT4_DEPLOYMENT_NAME")
    llm = AzureChatOpenAI(deployment_name=MODEL, temperature=0.2, max_tokens=4000, logprobs=True)

    numbers_list = list(range(1, 101))
                    # Display a selectbox with numbers 1 to 100, with 10 as the default value

    modified_value = selected_index.replace("srch-index-", "")
        # Split by hyphen and join without spaces
    modified_value = ''.join(modified_value.split('-'))
    available_docs = get_cached_docs(modified_value)
    st.session_state['available_docs'] = available_docs
    with col_k:

        selected_number = st.selectbox("Select the number of search results", numbers_list, index=9, help="Select the number of search results to retrieve for semantic search.")
        st.multiselect("Select the files in the project to search in", st.session_state['available_docs'], help="Files in the project to search in, not selecting anything will seach in all files", placeholder="Leave empty to search in all files")

                    #only sementic search
                      # index=9 because 10 is the 10th item in the list  
                      # 
                      # 
#                row1_col1, row1_col2, row1_col3 = st.columns([1, 1, 0.5], vertical_alignment="bottom")
#                row2_col1, row2_col2, _ = st.columns([1, 1,0.5], vertical_alignment="bottom")


                # Handle "Reformulate Question" button
#                with row1_col3:
 #                   if st.button('Reformulate Question', help="Click to reformulate the question"):
                        # Create empty context and append a placeholder document
#                        empty_comtext = []
#                        empty_value = {"chunk": "Empty", "score": 0.00, "location": "empty"}
#                        empty_comtext.append(
 #                           Document(
 #                               page_content=empty_value["chunk"], 
#                                metadata={"source": empty_value["location"], "score": empty_value["score"]}
#                            )
#                        )

                        # Call the function to reformulate the question
#                        rfq = reformulate_question(llm, query, empty_comtext)

                        # If reformulation is successful, store the reformulated questions in session state
#                        if rfq:
#                            st.session_state.reformulated_questions = rfq
#                            st.session_state.show_suggestions = True
#                        else:
                            # Handle error if the reformulation fails
#                            st.session_state.reformulated_questions = ["Error, try again", "Error, try again", "Error, try again", "Error, try again"]
#                            st.session_state.show_suggestions = True

                # Display reformulated suggestions only if the button has been pressed
#                if st.session_state.show_suggestions:
#                    row1 = [row1_col1, row1_col2]
#                    row2 = [row2_col1, row2_col2]
#                    columns = row1 + row2
#                    for i, suggestion in enumerate(st.session_state.reformulated_questions):
#                        with columns[i]:
#                            if st.button(f"{suggestion}", on_click=update_query_from_suggestion, args=(suggestion,)):
#                                update_query_from_suggestion(suggestion)







    spinner_placeholder = st.empty()
    error_container = st.container()
    answer_container = st.container()
    search_container = st.container()
    # Main search logic
    if st.session_state["submitSearch"]:
        try:
            if not query or query.strip() == "":
                st.error("Please enter a valid question!")
                st.session_state['is_running'] = False
                #st.session_state['is_expanded'] = False
                
                  # Reset running state
            else:
                # Azure Search
                ordered_results = {}  # Initialize ordered_results as an empty dictionary

                try:
                    k = 6
                    st.session_state['is_running'] = True
                    if st.session_state['is_running']:
                        with spinner_placeholder.container():
                            with st.spinner(f"Searching {selected_label}..."):
                                ordered_results = get_search_results_async(query, [selected_index], k=selected_number, reranker_threshold=1, sas_token=os.environ['BLOB_SAS_TOKEN'],spinner_placeholder=spinner_placeholder)
                                st.session_state["submitSearch"] = True
                                st.session_state["doneStreaming"] = False 
                                answer_placeholder = st.empty()  # Placeholder for the streaming response
                                results_placeholder = st.empty()  # Placeholder for search results
                                st.session_state.stored_results = ordered_results


                except Exception as e:
                    st.error("No data returned from Azure Search. Please check the connection.")
                    st.session_state['is_running'] = False
                    st.session_state["submitSearch"] = False
                    st.session_state["doneStreaming"] = True
                    st.session_state["Error"] = "No data returned from Azure Search. Please check the connection."
                    logging.error(f"Search error: {e}")
                    st.rerun()


            main_search_flow(ordered_results, spinner_placeholder)


            logging.info(f"submitSearch: {st.session_state['submitSearch']}")
            logging.info(f"doneStreaming: {st.session_state['doneStreaming']}")
            logging.info(f"is_running: {st.session_state['is_running']}")
            st.session_state["doneStreaming"] = True
  



        except Exception as e:
            st.error(f"An error occurred: {e}")
            logging.error(f"Unexpected error: {e}")


        finally:
                # Reset states after completion
                
                st.session_state["submitSearch"] = False
                st.session_state["doneStreaming"] = True
                st.session_state['is_running'] = False
                st.rerun()
                logging.info(f"submitSearch in finally: {st.session_state['submitSearch']}")
                logging.info(f"doneStreaming in finally: {st.session_state['doneStreaming']}")
                logging.info(f"is_running in finally: {st.session_state['is_running']}")



    if st.session_state.Error:
        with error_container:
            st.error(st.session_state.Error)
    if st.session_state.stored_answer or st.session_state.stored_results is not None:
        if st.session_state["doneStreaming"] and not st.session_state["submitSearch"]:
                if not st.session_state.semantic_search_only:
                    if st.session_state.stored_answer:
                        with answer_container:
                            st.markdown("#### Answer")
                            st.markdown("---")
                            st.write(st.session_state.stored_answer)
                    else:
                        top_docs = process_documents_post(st.session_state.stored_results)
                        display_answer(top_docs,spinner_placeholder)

                display_search_results(st.session_state.stored_results)
                        