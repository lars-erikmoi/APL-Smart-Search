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
from utils import get_search_results, CustomAzureSearchQuestionReformulatorRetriever
from prompts import DOCSEARCH_PROMPT, QUESTION_GENERATOR_PROMPT
import time  # For simulating the progress bar

logging.basicConfig(filename='app.log', level=logging.INFO)

# Add custom CSS styles
st.markdown("""
    <style>
    /* Adjust the padding of the main block container */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
    }
    /* Style for the feedback link */
    .feedback-link {
        font-size: 12px;
        text-align: right;
    }
    </style>
    """, unsafe_allow_html=True)

# Create a container for the header
with st.container():
    # Create two columns for alignment
    col1, col2 = st.columns([9, 1])  # Adjust the ratio as needed

    # Title in the first (left) column
    with col1:
        st.title("GPT Smart Search Engine")

    # Feedback link in the second (right) column
    with col2:
        st.markdown(
            """
            <div class="feedback-link">
                <a href="https://your-feedback-link.com">Feedback</a>
            </div>
            """,
            unsafe_allow_html=True
        )

def clear_submit():
    st.session_state["submit"] = False

# Sidebar instructions
with st.sidebar:
    st.markdown("""# Instructions""")
    st.markdown("""
    Ask a question that you think can be answered with the information in about 10k Arxiv Computer Science publications from 2020-2021 or in 90k Medical Covid-19 Publications.
    """)

# Index selection dropdown
index_options = ["srch-index-books", "srch-index-pilot", "srch-index-csv"]

coli1, coli2 = st.columns([3, 1])
with coli1:
    query = st.text_input("Ask a question to your enterprise data lake", 
                          value="What are the main risk factors for Covid-19?", 
                          on_change=clear_submit)
    selected_index = st.selectbox("Choose Search Index", index_options)

search_button = st.button('Search')

# API and key checks
if not os.environ.get("AZURE_SEARCH_ENDPOINT"):
    st.error("Please set your AZURE_SEARCH_ENDPOINT in your Web App Settings")
elif not os.environ.get("AZURE_SEARCH_KEY"):
    st.error("Please set your AZURE_SEARCH_KEY in your Web App Settings")
elif not os.environ.get("AZURE_OPENAI_ENDPOINT"):
    st.error("Please set your AZURE_OPENAI_ENDPOINT in your Web App Settings")
elif not os.environ.get("AZURE_OPENAI_API_KEY"):
    st.error("Please set your AZURE_OPENAI_API_KEY in your Web App Settings")
elif not os.environ.get("BLOB_SAS_TOKEN"):
    st.error("Please set your BLOB_SAS_TOKEN in your Web App Settings")
else:
    os.environ["OPENAI_API_VERSION"] = os.environ["AZURE_OPENAI_API_VERSION"]
    MODEL = os.environ.get("AZURE_OPENAI_MODEL_NAME")
    llm = AzureChatOpenAI(deployment_name=MODEL, temperature=0.5, max_tokens=1000)

if search_button or st.session_state.get("submit"):
    if not query:
        st.error("Please enter a question!")
    else:
        try:
            # Progress bar initialization
            progress_text = "Operation in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)

            # Step 1: Instantiate the retriever
            retriever = CustomAzureSearchQuestionReformulatorRetriever(
                indexes=[selected_index],
                topK=10,
                reranker_threshold=1,
                sas_token=os.environ["BLOB_SAS_TOKEN"]
            )
            my_bar.progress(20, text="Retriever initialized...")

            # Step 2: Retrieve relevant documents
            with st.spinner("Retrieving relevant documents... ⏳"):
                retriever.get_relevant_documents(query)
            my_bar.progress(50, text="Documents retrieved...")

            # Check if documents were found
            if not retriever.context_docs:
                st.warning("No relevant documents found to answer the question.")
                my_bar.progress(100, text="Operation complete!")
            else:
                # Prepare a placeholder for streaming the answer
                response_placeholder = st.empty()

                # Initialize the LLM with streaming=True
                llm = AzureChatOpenAI(
                    deployment_name=MODEL,
                    temperature=0.5,
                    max_tokens=1000,
                    streaming=True
                )

                # Step 3: Get the chain and inputs for generating the answer
                chain, inputs = retriever.get_answer_chain(llm, query)

                if chain is None:
                    st.warning("Unable to generate the answer.")
                    my_bar.progress(100, text="Operation complete!")
                else:
                    my_bar.progress(60, text="Generating the answer...")

                    # Step 4: Run the chain with streaming using a generator
                    accumulated_answer = ""
                    for token in chain.stream(inputs):
                        accumulated_answer += token
                        response_placeholder.markdown(accumulated_answer)

                    my_bar.progress(80, text="Answer generated...")

                    st.session_state["submit"] = True
                    placeholder = st.empty()

                    # Display the answer
                    with placeholder.container():
                        st.markdown("#### Answer")
                        st.markdown(accumulated_answer, unsafe_allow_html=True)
                        st.markdown("---")

                        # Display the search results using retriever.context_docs
                        st.markdown("#### Search Results")
                        for doc in retriever.context_docs:
                            location = doc.metadata.get("source", "")
                            score = doc.metadata.get("score", 0)
                            score_percentage = str(round(float(score) * 100 / 4, 2))

                            st.markdown(f"**Score**: {score_percentage}%")
                            st.markdown(doc.page_content)
                            st.markdown("---")

                        # Update progress bar to complete
                        my_bar.progress(100, text="Operation complete!")

                    # Add a reformulate button
                    if "submit" in st.session_state and st.session_state.submit:
                        reformulate_button = st.button('Reformulate Question')
                        if reformulate_button:
                            # Prepare a placeholder for streaming the reformulated question
                            reformulate_placeholder = st.empty()

                            # Get the chain and inputs for reformulating the question
                            chain, inputs = retriever.get_new_question_chain(llm, query)

                            if chain is None:
                                st.warning("Unable to reformulate the question.")
                            else:
                                accumulated_question = ""
                                for token in chain.stream(inputs):
                                    accumulated_question += token
                                    reformulate_placeholder.markdown(accumulated_question)

                                st.markdown("#### Reformulated Question")
                                st.markdown(accumulated_question)

        except Exception as e:
            st.error(f"An error occurred: {e}")