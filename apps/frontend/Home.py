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
from pilot import chat_with_llm_stream, get_search_results
from prompts import DOCSEARCH_PROMPT, QUESTION_GENERATOR_PROMPT
import time  # For simulating the progress bar

# Set page configuration
# Set page configuration
st.set_page_config(page_title="APL Smart Search", page_icon="📖", layout="wide")

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



def clear_submit():
    st.session_state["submit"] = True

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
    query = st.text_input("Ask A Question About Your Selected Project", value="", on_change=clear_submit)

# Selectbox for index selection
with col_select:
    index_mapping = {
        "2323 STP for Bay du Nord": "srch-index-bay-du-nord",
        "2304 SLT for Also Tirreno": "srch-index-altso-tirreno",
    }
    selected_label = st.selectbox("Choose What Project To Search In", list(index_mapping.keys()))
    selected_index = index_mapping[selected_label]

# Search button
with col_button:
    search_button = st.button('Search')

# Check required environment variables
required_env_vars = [
    "AZURE_SEARCH_ENDPOINT", "AZURE_SEARCH_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "BLOB_SAS_TOKEN"
]
missing_env_vars = [var for var in required_env_vars if not os.environ.get(var)]

if missing_env_vars:
    st.error(f"Please set the following environment variables: {', '.join(missing_env_vars)}")
else:
    os.environ["OPENAI_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"]
    MODEL = os.environ.get("AZURE_OPENAI_MODEL_NAME")
    llm = AzureChatOpenAI(deployment_name=MODEL, temperature=0.4, max_tokens=1000)

    if search_button or st.session_state.get("submit"):
        try:
            if not query or query.strip() == "":
                st.error("Please enter a valid question!")
            else:
                # Azure Search
                try:
                    k = 6

                    with st.spinner(f"Searching {selected_label}..."):
                        ordered_results = get_search_results(query, [selected_index], k=k, reranker_threshold=1, sas_token=os.environ['BLOB_SAS_TOKEN'])
                        st.session_state["submit"] = True
                        st.session_state["doneStreaming"] = False  # Reset doneStreaming state
                        answer_placeholder = st.empty()  # Placeholder for the streaming response
                        results_placeholder = st.empty()  # Placeholder for search results

                except Exception as e:
                    st.error("No data returned from Azure Search. Please check the connection.")
                    logging.error(f"Search error: {e}")

            if "ordered_results" in locals():
                try:
                    top_docs = []
                    with st.spinner("Reading the source documents to provide the best answer... ⏳"):
                        for key, value in ordered_results.items():
                            location = value.get("location", "")
                            top_docs.append(Document(page_content=value["chunk"], metadata={"source": location, "score": value["score"]}))

                    if top_docs:
                        # Stream the response
                        st.markdown("#### Answer\n")
                        st.markdown(chat_with_llm_stream(DOCSEARCH_PROMPT, llm, query, top_docs))
                        st.session_state["doneStreaming"] = True  # Mark streaming as done
                    else:
                        st.write("#### Answer\nNo results found.")

                    # Display the search results after streaming is complete
                    with results_placeholder.container():
                        st.markdown("---")
                        st.markdown("#### Search Results")
                        for key, value in ordered_results.items():
                            location = value.get("location", "")
                            title = str(value.get('title', value['name']))
                            score = str(round(value['score'] * 100 / 4, 2))
                            st.markdown(f"**Location and Page:** [{location}]")
                            st.markdown(f"**Score:** {score}%")
                            st.markdown(value.get("caption", ""))
                            st.markdown("---")

                except Exception as e:
                    st.error("Error processing documents.")
                    logging.error(f"Document processing error: {e}")

            # Reset submit state
            st.session_state["submit"] = False

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logging.error(f"Unexpected error: {e}")