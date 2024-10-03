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

try:
    from pilot import chat_with_llm_stream, get_search_results, reformulate_question
    from prompts import DOCSEARCH_PROMPT, QUESTION_GENERATOR_PROMPT
    from dotenv import load_dotenv
    load_dotenv("credentials.env")
    #for local testing print every credential

except Exception as e:
    print(e)
    # if not found in the current directory, try the parent directory
    import sys
    sys.path.append("../../common")
    from pilot import (chat_with_llm_stream, get_search_results, reformulate_question)
    from prompts import DOCSEARCH_PROMPT, QUESTION_GENERATOR_PROMPT
    sys.path.append("../../")
    from dotenv import load_dotenv
    load_dotenv(r"C:\Users\bakklandmoil\GPT-Azure-Search-Engine\credentials.env")
    #for local testing print every credential




# Set page configuration
# Set page configuration
st.set_page_config(page_title="APL Smart Search", page_icon="📖", layout="wide", )
st.set_option("client.toolbarMode", "viewer")
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


def handle_search_button():
    if st.session_state['is_running']:
        # Stop the search
        st.session_state['is_running'] = False
        st.session_state["submitSearch"] = False  # Reset the submit state
    else:
        # Start the search
        st.session_state['is_running'] = True
        st.session_state["submitSearch"] = True


#### Session State Variables ####
def clear_submit():
    st.session_state["submit"] = True

if "submitSearch" not in st.session_state:
    st.session_state["submitSearch"] = False

if 'is_running' not in st.session_state:
    st.session_state['is_running'] = False


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
col_input, col_select, col_button = st.columns([2, 1, 0.5],vertical_alignment="bottom")

# Text input field
with col_input:
    query = st.text_input("Ask A Question About Your Selected Project", value=st.session_state.query, on_change=clear_submit, help="Enter your question below")

# Selectbox for index selection
with col_select:
    index_mapping = {
        "2323 STP for Bay du Nord": "srch-index-bay-du-nord",
        "2304 SLT for Also Tirreno": "srch-index-altso-tirreno",
        "2269 STP for B29 Polok & Chinwol": "srch-index-polok-chinwol",
    }
    selected_label = st.selectbox("Choose What Project To Search In", list(index_mapping.keys()), help="Select the project to search in")
    selected_index = index_mapping[selected_label]

# Search button
with col_button:
    button_label = "Stop" if st.session_state['is_running'] else "Search"
    st.button(button_label,on_click=handle_search_button, help="Click to start or stop the search")


row1_col1, row1_col2, row1_col3 = st.columns([1, 1, 0.5], vertical_alignment="bottom")
row2_col1, row2_col2, _ = st.columns([1, 1, 0.5], vertical_alignment="bottom")



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
    llm = AzureChatOpenAI(deployment_name=MODEL, temperature=0.4, max_tokens=1000)

    # Handle "Reformulate Question" button
    with row1_col3:
        if st.button('Reformulate Question', help="Click to reformulate the question"):
            # Create empty context and append a placeholder document
            empty_comtext = []
            empty_value = {"chunk": "Empty", "score": 0.00, "location": "empty"}
            empty_comtext.append(
                Document(
                    page_content=empty_value["chunk"], 
                    metadata={"source": empty_value["location"], "score": empty_value["score"]}
                )
            )

            # Call the function to reformulate the question
            rfq = reformulate_question(llm, query, empty_comtext)

            # If reformulation is successful, store the reformulated questions in session state
            if rfq:
                st.session_state.reformulated_questions = rfq
                st.session_state.show_suggestions = True
            else:
                # Handle error if the reformulation fails
                st.session_state.reformulated_questions = ["Error, try again", "Error, try again", "Error, try again", "Error, try again"]
                st.session_state.show_suggestions = True

    # Display reformulated suggestions only if the button has been pressed
    if st.session_state.show_suggestions:
        row1 = [row1_col1, row1_col2]
        row2 = [row2_col1, row2_col2]
        columns = row1 + row2
        for i, suggestion in enumerate(st.session_state.reformulated_questions):
            with columns[i]:
                if st.button(f"{suggestion}", on_click=update_query_from_suggestion, args=(suggestion,)):
                    update_query_from_suggestion(suggestion)

    # Main search logic
    if st.session_state["submitSearch"]:
        try:
            if not query or query.strip() == "":
                st.error("Please enter a valid question!")
                st.session_state['is_running'] = False  # Reset running state
            else:
                # Azure Search
                ordered_results = {}  # Initialize ordered_results as an empty dictionary

                try:
                    k = 6
                    st.session_state['is_running'] = True

                    if st.session_state['is_running']:
                        with st.spinner(f"Searching {selected_label}..."):
                            ordered_results = get_search_results(query, [selected_index], k=k, reranker_threshold=1, sas_token=os.environ['BLOB_SAS_TOKEN'])
                            st.session_state["submitSearch"] = True
                            st.session_state["doneStreaming"] = False 
                            answer_placeholder = st.empty()  # Placeholder for the streaming response
                            results_placeholder = st.empty()  # Placeholder for search results

                except Exception as e:
                    st.error("No data returned from Azure Search. Please check the connection.")
                    logging.error(f"Search error: {e}")

            if st.session_state['is_running'] and ordered_results:
                answer = st.container()
                search = st.container()
                try:
                    top_docs = []
                    with st.spinner("Reading the source documents to provide the best answer... ⏳"):
                        for key, value in ordered_results.items():
                            if not st.session_state['is_running']:  # Check for stop
                                st.warning("Stopped processing documents.")
                                break
                            location = value.get("location", "")
                            top_docs.append(Document(page_content=value.get("chunk", ""), metadata={"source": location, "score": value.get("score", 0)}))

                    # Display the answer and search results
                    if top_docs:
                        with answer:
                            st.markdown("#### Answer")
                            st.markdown("---")
                            chat_with_llm_stream(DOCSEARCH_PROMPT, llm, query, top_docs)
                    else:
                        st.write(f"#### Answer\nNo results found.")

                    with search:
                        st.markdown("---")
                        st.markdown("#### Search Results")
                        if top_docs:
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
                        
                except Exception as e:
                    st.error("Error processing documents.")
                    logging.error(f"Document processing error: {e}")

            # Reset states after completion
            st.session_state["submitSearch"] = False
            st.session_state["doneStreaming"] = True
            st.session_state['is_running'] = False

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logging.error(f"Unexpected error: {e}")