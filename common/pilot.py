import linecache
import sys
import streamlit as st
import urllib
import os
import re
import time
import random
from operator import itemgetter
from collections import OrderedDict
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import fitz  # PyMuPDF
import requests
from io import BytesIO
import streamlit as st
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

import re
import os
import json
from io import BytesIO
from typing import Any, Dict, List, Optional, Awaitable, Callable, Tuple, Type, Union
import requests
import asyncio

from collections import OrderedDict
import base64
from bs4 import BeautifulSoup
import docx2txt
import tiktoken
import html
import time
from time import sleep
from typing import List, Tuple
from pypdf import PdfReader, PdfWriter
from dataclasses import dataclass
from sqlalchemy.engine.url import URL
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential



from langchain_core.callbacks import CallbackManagerForToolRun

from langchain_core.callbacks import AsyncCallbackManagerForToolRun
from pydantic import BaseModel, Field, Extra


from langchain_core.tools import BaseTool

from langchain_core.tools import StructuredTool

from langchain_core.tools import tool
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain_core.exceptions import OutputParserException

from langchain_core.output_parsers import BaseOutputParser
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.agents.agent_toolkits import create_csv_agent


from langchain_core.tools import BaseTool

from langchain_core.tools import StructuredTool

from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain.agents import AgentExecutor, initialize_agent, AgentType 

from langchain_core.tools import Tool
from langchain_community.utilities import BingSearchAPIWrapper
from langchain.agents import  create_openai_tools_agent

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.callbacks import BaseCallbackManager
from langchain_community.utilities import TextRequestsWrapper
from langchain.chains import APIChain
from langchain_community.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain_core.utils.json_schema import dereference_refs
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from operator import itemgetter
from typing import List
import urllib
from dotenv import load_dotenv
load_dotenv("credentials.env")



import os
import json
import random
import re
import urllib.parse
from collections import OrderedDict
from typing import List
import requests
import streamlit as st

import functools
import logging
import threading
# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def debug(func):
    """A decorator that logs the function signature and return value with a length limit"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        def truncate(value, length=100):
            """Truncates the representation of value if it exceeds the specified length"""
            value_str = repr(value)
            return value_str if len(value_str) <= length else value_str[:length] + '...'

        # Represent and truncate each argument
        args_repr = [truncate(a) for a in args]
        kwargs_repr = [f"{k}={truncate(v)}" for k, v in kwargs.items()]  # Represent each keyword argument
        signature = ", ".join(args_repr + kwargs_repr)
        logging.debug(f"Calling {func.__name__}({signature})")
        
        result = func(*args, **kwargs)
        
        # Truncate result if it's too long
        logging.debug(f"{func.__name__!r} returned {truncate(result)}")
        return result
    return wrapper_debug


import time

def timer(func):
    """A decorator that logs the runtime of the function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.time()  # Start the timer
        result = func(*args, **kwargs)
        end_time = time.time()  # End the timer
        runtime = end_time - start_time
        logging.info(f"{func.__name__!r} executed in {runtime:.4f} seconds")
        return result
    return wrapper_timer




import asyncio
import httpx
import concurrent.futures
import fitz  # PyMuPDF for PDF processing
import random
import re
import urllib
from io import BytesIO
from collections import OrderedDict
@timer
async def download_file_async(pdf_url, result_id, retries=3, delay=2):
    """Download a file asynchronously with retry logic, keeping track of the PDF URL."""
    attempt = 0
    while attempt < retries:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(pdf_url)
                if response.status_code == 200:
                    pdf_data = BytesIO(response.content)
                    # Check if the downloaded data has content
                    if pdf_data.getbuffer().nbytes > 0:
                        # Return the PDF URL along with the PDF data
                        return pdf_url, pdf_data  # Use pdf_url, not result_id
                    else:
                        print(f"Warning: Downloaded PDF from {pdf_url} (ID: {result_id}) is empty.")
                else:
                    print(f"Failed to download {pdf_url} (ID: {result_id}): {response.status_code}")
        except Exception as e:
            print(f"Error downloading {pdf_url}, attempt {attempt + 1}/{retries} (ID: {result_id}): {e}")
        
        # Wait before retrying
        attempt += 1
        await asyncio.sleep(delay)
    
    print(f"Failed to download {pdf_url} after {retries} attempts. (ID: {result_id})")
    return pdf_url, None  # Return pdf_url, not result_id


@timer
def create_location_to_result_map(search_results):
    """Create a dictionary mapping PDF location to a list of result IDs."""
    location_to_result_ids = {}

    # Extract the actual search results from the 'value' field
    if 'value' not in search_results:
        print("No search results found.")
        return location_to_result_ids

    # Iterate over actual results in the 'value' field
    for result in search_results['value']:
        # Check if the result contains the expected keys
        if isinstance(result, dict) and 'location' in result and 'id' in result:
            location = result['location']
            result_id = result['id']

            # If this location is already in the dictionary, append the result_id
            if location in location_to_result_ids:
                location_to_result_ids[location].append(result_id)
            else:
                # Otherwise, create a new entry with this location
                location_to_result_ids[location] = [result_id]
        else:
            print(f"Unexpected result format: {result}")  # Log unexpected result format for debugging

    return location_to_result_ids




def find_search_term_in_pdfAsync(pdf_stream, search_term):
    """Search for a term in a PDF and return the page number."""
    try:
        logging.info(f"Searching for term '{search_term}' in PDF.")

        pdf_doc = fitz.open(stream=pdf_stream, filetype="pdf")

        for page_num in range(pdf_doc.page_count):
            page = pdf_doc.load_page(page_num)
            text = page.get_text("text")
            if search_term.lower() in text.lower():
                pdf_doc.close()
                return [page_num + 1]  # Return the page number (1-based)
        pdf_doc.close()
    except Exception as e:
        print(f"Error processing PDF: {e}")
    return None

async def download_all_pdfs_concurrently(document_name_to_result_ids):
    """Download all unique PDFs concurrently and return a dictionary with document name as key and PDF stream as value."""
    downloaded_pdfs = {}
    
    download_tasks = []
    for document_name, result_ids in document_name_to_result_ids.items():
        # Use the first result_id for the download task (it will only download once per document)
        result_id = result_ids[0]
        download_tasks.append(download_file_async(document_name, result_id))  # Download using document name (URL) and result_id
    
    # Perform all downloads concurrently
    pdf_results = await asyncio.gather(*download_tasks)
    
    # Populate downloaded_pdfs with the results, mapping by document name (URL)
    for document_name, pdf_stream in pdf_results:
        if pdf_stream:  # Only add successful downloads
            downloaded_pdfs[document_name] = pdf_stream
            print(f"PDF downloaded successfully for document: {document_name}")
        else:
            print(f"PDF download failed for document: {document_name}")
    
    return downloaded_pdfs





@timer
def get_search_resultsAsync(query: str, indexes: list, k: int = 5, reranker_threshold: int = 1, use_captions: bool = True, sas_token: str = "") -> OrderedDict:
    """Performs search and processes multiple results concurrently, ensuring consistent linking with IDs."""
    headers = {'Content-Type': 'application/json', 'api-key': os.environ["AZURE_SEARCH_KEY"]}
    params = {'api-version': os.environ['AZURE_SEARCH_API_VERSION']}

    agg_search_results = dict()
    for index in indexes:
        search_payload = {
            "search": query,
            "select": "id, title, chunk, name, location",
            "queryType": "semantic",
            "vectorQueries": [{"text": query, "fields": "chunkVector", "kind": "text", "k": k}],
            "semanticConfiguration": "my-semantic-config",
            "captions": "extractive",
            "answers": "extractive",
            "count": "true",
            "top": k    
        }

        resp = requests.post(os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + index + "/docs/search",
                             data=json.dumps(search_payload), headers=headers, params=params)
        

        if resp.status_code == 200:
            search_results = resp.json()
            agg_search_results[index] = search_results
        else:
            print(f"Failed to retrieve search results for index {index}: {resp.status_code}")



    # Step 1: Create the mapping from location to result IDs
    mappings = create_location_to_result_map(agg_search_results[index])

    # Step 2: Download all PDFs concurrently (each location only once)
    downloaded_pdfs = asyncio.run(download_all_pdfs_concurrently(mappings))

    # Step 3: Process the PDFs and search for terms based on result IDs
    content = OrderedDict()
    # Iterate through the search results and handle both PDF and non-PDF files
    for index, search_results in agg_search_results.items():
        for result in search_results['value']:
            result_id = result['id']
            location = result['location']
            file_name = result['name']
            
            # Check if the file is a PDF and handle accordingly
            if file_name.endswith(".pdf") or file_name.endswith(".PDF"):
                # Handle the downloaded PDFs
                if location in downloaded_pdfs:
                    pdf_stream = downloaded_pdfs[location]
                    # Extract content to use (caption or highlights)
                    content_to_use = (
                        result.get('@search.captions', [{}])[0].get('highlights', "") if not use_captions
                        else result.get('@search.captions', [{}])[0].get('text', "")
                    )   

                    if content_to_use:
                        words = content_to_use.split()
                        word_groups = [' '.join(words[i:i + 4]) for i in range(len(words) - 3)]
                        filtered_groups = [group for group in word_groups if not re.search(r'[^a-zA-Z0-9.,:\s]', group)]

                        if not filtered_groups:
                            print(f"No valid filtered groups found for result ID {result_id}. Skipping.")
                            continue

                        # Search in the PDF for the filtered group terms
                        page_number = None
                        for loop in range(15):
                            search_term = random.choice(filtered_groups)
                            page_number = find_search_term_in_pdfAsync(pdf_stream, search_term)

                            if page_number:
                                break

                        if page_number is None:
                            page_number = 1  # Default to page 1 if no page is found

                        # Construct the page URL if the search term is found
                        page_url = f"#page={page_number[0]}" if isinstance(page_number, list) else f"#page={page_number}"
                        complete_location = f"{location}{page_url}"

                        # Add the result to content with cross-referenced metadata
                        content[result_id] = {
                            "title": result['title'],
                            "name": result['name'],
                            "chunk": result['chunk'],
                            "location": complete_location,
                            "caption": content_to_use,
                            "score": result['@search.rerankerScore'],
                            "index": result.get('index'),
                            "page": page_url
                        }

            # Handle non-PDF files (skip download but still include them in the results)
            elif file_name.endswith(".xlsx") or file_name.endswith(".xls") or file_name.endswith(".csv") or file_name.endswith(".docx"):
                print(f"Skipping download for non-PDF file: {file_name}")
                
                # Add the non-PDF file to the results with default page info
                content[result_id] = {
                    "title": result['title'],
                    "name": result['name'],
                    "chunk": result['chunk'],
                    "location": location,
                    "caption": result.get('@search.captions', [{}])[0].get('text', ""),
                    "score": result['@search.rerankerScore'],
                    "index": result.get('index'),
                    "page": ""  # No page number for non-PDF files
                }

    print(f"Processed {len(content)} results with search terms found.")
    
    # Order and return top K results
    try:
        ordered_content = OrderedDict(sorted(content.items(), key=lambda x: x[1]["score"], reverse=True)[:k])
    except TypeError as e:
        print(f"Error while creating OrderedDict: {e}")
        ordered_content = OrderedDict()

    return ordered_content




def extract_filtered_groups(content_part):
    """Extract word groups from content, filtering unwanted characters."""
    words = content_part.split()
    word_groups = [' '.join(words[i:i + 4]) for i in range(len(words) - 3)]
    filtered_groups = [group for group in word_groups if not re.search(r'[^a-zA-Z0-9.,:\s]', group)]
    return filtered_groups




def get_search_results(query: str, indexes: list, 
                       k: int = 5,
                       reranker_threshold: int = 1,
                       sas_token: str = "") -> List[dict]:
    """Performs multi-index hybrid search and returns ordered dictionary with the combined results"""
    


    headers = {'Content-Type': 'application/json','api-key': os.environ["AZURE_SEARCH_KEY"]}
    params = {'api-version': os.environ['AZURE_SEARCH_API_VERSION']}
    
    agg_search_results = dict()
    for index in indexes:
        search_payload = {
            "search": query,
            "select": "id, title, chunk, name, location",
            "queryType": "semantic",
            "vectorQueries": [{"text": query, "fields": "chunkVector", "kind": "text", "k": k}],
            "semanticConfiguration": "my-semantic-config",
            "captions": "extractive",
            "answers": "extractive",
            "count":"true",
            "top": k    
        }
        

        resp = requests.post(os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + index + "/docs/search",
                                     data=json.dumps(search_payload), headers=headers, params=params)
        search_results = resp.json()
        
        agg_search_results[index] = search_results

    content = dict()
    ordered_content = OrderedDict()
    for index,search_results in agg_search_results.items():
        for result in search_results['value']:      
            if result['@search.rerankerScore'] > reranker_threshold: # Show results that are at least N% of the max possible score=4             
                page_url = ""
                if result['name'].endswith(".pdf") or result['name'].endswith(".PDF"):
                    page_match = re.search(r"page_(\d+)", str(result['title']))
                    if page_match:
                        page_number = str(page_match.group(1))
                        page_url = "#page=" + str(page_number)
                    else:
                        search_term = ""
                        content_part = str(result['@search.captions'][0]['text'])
                        page_url = content_part
                        if content_part:
                            words = content_part.split()
                            word_groups = [' '.join(words[i:i+4]) for i in range(len(words)-3)]
                            filtered_groups = [group for group in word_groups if not re.search(r'[^a-zA-Z0-9.,:\s]', group)]

                                    # Join the filtered groups into a single sentence
                                    
                            for i in range(15):
                                result_sentence = random.choice(filtered_groups) 
                                try:
                                    result_search = find_search_term_in_pdf_from_url(str(result['location']), str(result_sentence))
                                except Exception as e:
                                    raise
                                        
                                if result_search:
                                    result_page = result_search
                                        
                                if len(result_search) == 1:  # Condition to break the loop
                                    result_page = result_search
                                    break
                                        
                            if result_page:  
                                if result_page[0]:
                                    page_url = "#page=" + str(result_page[0])
                                else:
                                    page_url = "#page=" + str(result_page)
                            else:
                                page_url = "#page=" + "1"
                        else:
                            page_url = "#page=" + "1"  # Default to page 1 if no page is found

                elif result['name'].endswith(".xlsx") or result['name'].endswith(".xls"):
                    page_url = ""
                        
                elif result['name'].endswith(".csv"):
                    page_url = ""

                if result['location']:
                    decoded_location = urllib.parse.unquote(result['location'])
                    encoded_location = urllib.parse.quote(decoded_location, safe='/:')
                content[result['id']]={
                                                "title": result['title'], 
                                                "name": result['name'], 
                                                "chunk": result['chunk'],
                                                "location": encoded_location + page_url if result['location'] else "", #changed was + sas_token before if
                                                "caption": result['@search.captions'][0]['text'],
                                                "score": result['@search.rerankerScore'],
                                                "index": index,
                                                "page": page_url
                                            } 

    topk = k
    
    
    count = 0  # To keep track of the number of results added
    for id in sorted(content, key=lambda x: content[x]["score"], reverse=True):
        ordered_content[id] = content[id]
        count += 1
        if count >= topk:  # Stop after adding topK results
            break
    
    return ordered_content

# Updated get_search_results function







def chat_with_llm(pre_prompt, llm, query, context):
    chain = (
        pre_prompt  # Passes the 4 variables above to the prompt template
        | llm   # Passes the finished prompt to the LLM
        | StrOutputParser()  # converts the output (Runnable object) to the desired output (string)
    )

    answer = chain.invoke({"question": query, "context":context})
    return answer

@timer
def chat_with_llm_stream(pre_prompt, llm, query, context):
    # Create a placeholder for streaming the response

    # Existing chain processing code

    # Chain processing
    chain = (
        pre_prompt  # Passes the variables to the prompt template
        | llm  # Use streaming feature of the LLM
        | StrOutputParser()  # Converts the output to a string
    )

    # Stream the response by invoking the chain
    ans = st.write_stream(chain.stream({"question": query, "context": context}))
    return ans



@timer
def chat_with_llm_stream2(pre_prompt, llm, query, context, batch_size=10):

    # Create a placeholder for streaming the response
    response_placeholder = st.empty()

    # Start an empty string to accumulate the answer as it streams
    accumulated_answer = ""

    # Chain processing
    chain = (
        pre_prompt  # Passes the variables to the prompt template
        | llm  # Use streaming feature of the LLM
        | StrOutputParser()  # Converts the output to a string
    )


    # Stream the response by invoking the chain
    for token in chain.stream({"question": query, "context": context}):
        accumulated_answer += token  # Accumulate the tokens
        response_placeholder.markdown(f"#### Answer\n---\n{accumulated_answer}")  # Update the placeholder progressively

    # Clear the placeholder
    # response_placeholder.empty()

    return accumulated_answer, response_placeholder



@timer
def chat_with_llm_stream3(pre_prompt, llm, query, context, batch_size=1):
    """
    Streams response from the LLM in batches, progressively displaying it in the UI.
    """

    # Create a placeholder for streaming the response
    response_placeholder = st.empty()

    # Use a generator to yield tokens as they are generated
    def token_generator(chain):
        for i, token in enumerate(chain.stream({"question": query, "context": context})):
            yield token

    # Chain processing
    chain = (
        pre_prompt  # Passes the variables to the prompt template
        | llm  # Use streaming feature of the LLM
        | StrOutputParser()  # Converts the output to a string
    )

    # Use a list to accumulate tokens (more efficient than string concatenation)
    accumulated_answer_list = []
    token_batch = []

    # Iterate over the generator and process tokens
    for i, token in enumerate(token_generator(chain)):
        token_batch.append(token)
        accumulated_answer_list.append(token)  # Append tokens to the list

        # Update the UI after every batch_size tokens
        if i % batch_size == 0:
            response_placeholder.markdown(f"#### Answer\n---\n{''.join(accumulated_answer_list)}", unsafe_allow_html=True)



    response_placeholder.markdown(f"#### Answer\n---\n{''.join(accumulated_answer_list)}", unsafe_allow_html=True)


    # Join the accumulated tokens into a final string
    accumulated_answer = ''.join(accumulated_answer_list)

    return accumulated_answer, response_placeholder


@functools.lru_cache(maxsize=32)
@timer
def download_pdf(pdf_url):
    response = requests.get(pdf_url)
    if response.status_code != 200:
        return None
    pdf_data = BytesIO(response.content)
    return fitz.open(stream=pdf_data, filetype="pdf")



@timer
def find_search_term_in_pdf_from_url(pdf_url, search_term):
    doc = download_pdf(pdf_url)
    if doc is None:
        return None

    # Make a copy of the document so the cached document is not modified
    doc_copy = fitz.open(stream=doc.write(), filetype="pdf")

    pages_with_term = []
    for page_num in range(len(doc_copy)):
        page = doc_copy.load_page(page_num)
        text = page.get_text("text")
        if search_term.lower() in text.lower():
            pages_with_term.append(page_num + 1)

    doc_copy.close()
    return pages_with_term




def reformulate_question(llm, query, context):
    PRE_PROMPT_TEXT = '''
Reformulate the question so it improves semantic search accuracy. Provide four different suggestions, and the answer should follow the exact format:
[reformulated question 1], [reformulated question 2], [reformulated question 3], [reformulated question 4]. Make sure the structure matches precisely.
    '''

    # Create the prompt template using ChatPromptTemplate and from_messages
    PRE_PROMPT = ChatPromptTemplate.from_messages(
        [
            ("system", PRE_PROMPT_TEXT + "\n\nCONTEXT:\n{context}\n\n"),  # Add context in the system message
            MessagesPlaceholder(variable_name="history", optional=True),  # Optional conversation history
            ("human", "{question}"),  # The query is injected here
        ]
    )
    #st.markdown("Reformulating...")
    questions_string = chat_with_llm(PRE_PROMPT, llm, query, context)
    #st.markdown("Done Reformulating...")
    questions_string = re.sub(r'\[reformulated.*?\]', '', questions_string)
    # Split the string by commas and remove parentheses and extra spaces
    questions_array = [question.strip("() ").strip() for question in questions_string.split(",")]
    cleaned_questions = [question.strip("[]") for question in questions_array]

    return cleaned_questions



# Helper function to search for a term in the PDF
@timer
def find_search_term_in_pdf(doc, search_term):
    pages_with_term = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if search_term.lower() in text.lower():
            pages_with_term.append(page_num + 1)  # Store page numbers starting from 1
    
    return pages_with_term



  


