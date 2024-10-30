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
import asyncio
import httpx
import concurrent.futures
import fitz  # PyMuPDF for PDF processing
import random
import re
import urllib
from io import BytesIO
from collections import OrderedDict
import time



# Set up basic configuration for logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def debug(func):
    """
    A decorator that logs the function's signature and its return value with a length limit for better readability in logs.

    - func (Callable): The function being decorated.

    Returns:
    - Callable: The decorated function with logging for both the input arguments and the return value, truncated if necessary.
    """

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





def timer(func):
    """
    A decorator that logs the runtime of the function, supporting both synchronous and asynchronous functions.

    - func (Callable): The function being decorated (sync or async).

    Returns:
    - Callable: The decorated function with added logging for the runtime, measured in seconds.
    """
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        if asyncio.iscoroutinefunction(func):  # Check if the function is asynchronous
            async def async_wrapper_timer(*args, **kwargs):
                start_time = time.time()  # Start the timer
                result = await func(*args, **kwargs)  # Await the async function
                end_time = time.time()  # End the timer
                runtime = end_time - start_time
                logging.info(f"{func.__name__!r} executed in {runtime:.4f} seconds (async)")
                return result
            return async_wrapper_timer(*args, **kwargs)
        else:  # If the function is synchronous
            start_time = time.time()  # Start the timer
            result = func(*args, **kwargs)
            end_time = time.time()  # End the timer
            runtime = end_time - start_time
            logging.info(f"{func.__name__!r} executed in {runtime:.4f} seconds (sync)")
            return result
    return wrapper_timer



@timer
@st.cache_resource(ttl=1500,  max_entries=20, show_spinner=False)
async def download_file_async(pdf_url, result_id, retries=3, delay=2):
    """
    Downloads a file asynchronously with retry logic, keeping track of the PDF URL.

    - pdf_url (str): The URL of the PDF to download.
    - result_id (str): An identifier for the download process (used for logging).
    - retries (int, optional): The number of retries allowed if the download fails (default: 3).
    - delay (int, optional): The delay between retries in seconds (default: 2).

    Returns:
    - Tuple[str, BytesIO | None]: Returns a tuple containing the PDF URL and the PDF data (BytesIO stream). If the download fails after retries, returns the URL and None.
    """

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
@timer
def create_location_to_result_map(search_results):
    """
    Creates a dictionary that maps PDF locations to a list of result IDs from search results.

    - search_results (dict): The search results containing 'location' and 'id' fields. The 'value' field should contain the actual results.

    Returns:
    - Dict[str, List[str]]: Returns a dictionary where the keys are document locations (URLs) and the values are lists of corresponding result IDs.
    """

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



@st.cache_resource(ttl=1500,  max_entries=10, show_spinner=False)
def find_search_term_in_pdfAsync(pdf_stream, search_term):
    """
    Searches for a specific term in a PDF and returns the page number where the term appears.

    - pdf_stream (BytesIO): A PDF document stream to search through.
    - search_term (str): The term to search for within the PDF.

    Returns:
    - List[int] | None: Returns a list with the page number (1-based) where the search term is found, or None if the term is not found.
    """

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
    """
    Downloads all unique PDFs concurrently and returns a dictionary of document names to PDF streams.

    - document_name_to_result_ids (Dict[str, List[str]]): A dictionary mapping document names (PDF URLs) to their corresponding result IDs.

    Returns:
    - Dict[str, BytesIO]: A dictionary where keys are document names (URLs) and values are the PDF streams (BytesIO).
    """

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

@st.cache_resource(ttl=3600,show_spinner=False)  # Cache the results for 1 hour
def get_cached_pdf_results(document_name_to_result_ids):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    pdf_results = loop.run_until_complete(download_all_pdfs_concurrently(document_name_to_result_ids))
    return pdf_results


@st.cache_data(ttl=1500, max_entries=10, show_spinner=False)
def get_documents(query, indexes, k, headers, params, title_prefixes=None):
    """
    Performs a search query across multiple indexes and returns the aggregated search results.
    
    Args:
    - query (str): The search query to execute.
    - indexes (list): A list of index names to search within.
    - k (int): The number of results to retrieve for each index.
    - headers (dict): HTTP headers containing API keys and other necessary information for the search request.
    - params (dict): HTTP parameters, such as the API version.
    - title_prefixes (list, optional): A list of document name prefixes to filter the search results (default: None).
    
    Returns:
    - dict: A dictionary where keys are index names and values are either the search results or an error message.
        - results: The search results for the index.
        - error: An error message if the search failed.
    - str: The index name that was used for the search.
    
    """
    agg_search_results = {}
    index = ""
    
    if title_prefixes:
        
        # Dynamically build the filter expression using the title_prefixes array
        filter_expression = " or ".join([f"docname eq '{title}'" for title in title_prefixes])
        
    try:
        for index in indexes:  
            if title_prefixes:
                search_payload = {
                    "search": query,
                    "select": "id, title, chunk, name, location",
                    "queryType": "semantic",
                    "vectorQueries": [{"text": query, "fields": "chunkVector", "kind": "text", "k": k}],
                    "semanticConfiguration": "my-semantic-config",
                    "captions": "extractive",
                    "answers": "extractive",
                    "count":"true",
                    "top": k,
                    "filter": f"({filter_expression})"   
                }
            else:
                search_payload = {
                    "search": query,
                    "select": "id, title, chunk, name, location",
                    "queryType": "semantic",
                    "vectorQueries": [{"text": query, "fields": "chunkVector", "kind": "text", "k": k}],
                    "semanticConfiguration": "my-semantic-config",
                    "captions": "extractive",
                    "answers": "extractive",
                    "count":"true",
                    "top": k,   
                }

                # Perform the search request
                resp = requests.post(os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + index + "/docs/search",
                                    data=json.dumps(search_payload), headers=headers, params=params)
                
                if resp.status_code == 200:
                    agg_search_results[index] = {"results": resp.json()}
                else:
                    error_message = f"Failed to retrieve search results for index {index} (Status Code: {resp.status_code})"
                    agg_search_results[index] = {"error": error_message}
                    logging.error(error_message)
        
    except Exception as e:
        error_message = f"An error occurred during search in index {index}: {str(e)}"
        agg_search_results[index] = {"error": error_message}
        logging.exception(error_message)
    
    return agg_search_results,index




@timer
@debug
def get_search_results_async(query: str, indexes: list, spinner_placeholder, k: int = 5, reranker_threshold: int = 1, sas_token: str = "",title_prefixes=None) -> OrderedDict:
    """
    Performs a search across multiple indexes, processes the results, and links them to their corresponding IDs, while handling PDF downloads and term search.
    Does downloading the correct pdf's asynchronously.

    - query (str): The search query.
    - indexes (list): A list of index names to search within.
    - k (int, optional): The maximum number of results to retrieve (default: 5).
    - reranker_threshold (int, optional): The minimum reranker score for filtering results (default: 1).
    - sas_token (str, optional): A SAS token for accessing resources (optional).
    - title_prefixes (list, optional): A list of document name prefixes to filter the search results (default: None).

    Returns:
    - OrderedDict: An ordered dictionary of processed search results, including document locations and extracted terms with metadata.
    """

    headers = {'Content-Type': 'application/json', 'api-key': os.environ["AZURE_SEARCH_KEY"]}
    params = {'api-version': os.environ['AZURE_SEARCH_API_VERSION']}

    # Step 0: Perform the search across multiple index(es)
    agg_search_results, index = get_documents(query, indexes, k, headers, params, title_prefixes)



    # Step 1: Create the mapping from location to result IDs
    print(f"Aggregated search results for index {index} with {len(agg_search_results[index]['results']['value'])} results.")
    #use the old mappings if they exist, serves as a cache

    mappings = create_location_to_result_map(agg_search_results[index]['results'])
    # Step 2: Filter out locations already in old_mappings


    with spinner_placeholder.container():
        with st.spinner("Downloading PDFs..."):
                # Download PDFs concurrently for the new locations
            downloaded_pdfs = get_cached_pdf_results(mappings)


# Now `old_mappings` contains both the old cached data and the newly downloaded data


    # Step 3: Process the PDFs and search for terms based on result IDs
    content = OrderedDict()
    # Iterate through the search results and handle both PDF and non-PDF files

    with spinner_placeholder.container():
        with st.spinner("Processing search results..."):
            for index, search_results in agg_search_results.items():
                search_results = search_results['results']

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
                                result.get('@search.captions', [{}])[0].get('highlights', "")
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
                                context = f"The text below is from {file_name}, located at {complete_location}. The text is extracted from the document for context. The chosen chunk is gather from Term Frequency-Inverse Document Frequency concept and embeddings. The chunk is: {result['chunk']}"                                    
                                # Add the result to content with cross-referenced metadata
                                content[result_id] = {
                                    "title": result['title'],
                                    "name": result['name'],
                                    "chunk": context,
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


def get_search_results_sync(query: str, indexes: list, 
                       k: int = 5,
                       reranker_threshold: int = 1,
                       sas_token: str = "", mappings=None) -> List[dict]:
    """
    Performs a search across multiple indexes, processes the results, and links them to their corresponding IDs, while handling PDF downloads and term search.
    Does NOT download the pdf's asynchronously.

    - query (str): The search query.
    - indexes (list): A list of index names to search within.
    - k (int, optional): The maximum number of results to retrieve (default: 5).
    - reranker_threshold (int, optional): The minimum reranker score for filtering results (default: 1).
    - use_captions (bool, optional): Flag to determine whether to use captions or highlights from search results (default: True).
    - sas_token (str, optional): A SAS token for accessing resources (optional).

    Returns:
    - OrderedDict: An ordered dictionary of processed search results, including document locations and extracted terms with metadata.
    """    


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




# Define functions to calculate log probabilities and find the best completion
def calculate_average_logprob(logprobs):
    """
    Calculate the average log probability for a given list of logprobs in the 'content' key.
    
    Args:
    - logprobs: A dictionary with a 'content' key containing log probability dictionaries for each token.

    Returns:
    - float: The average log probability for the tokens, or -inf if logprobs is unavailable.
    """
    # Ensure 'content' exists and contains the logprobs list
    if not logprobs or 'content' not in logprobs or not logprobs['content']:
        return float('-inf')
    
    # Extract logprob values from each token entry in 'content'
    try:
        logprob_values = [token.get("logprob", 0) for token in logprobs['content']]
        avg_logprob = sum(logprob_values) / len(logprob_values) if logprob_values else float('-inf')
    except (TypeError, KeyError) as e:
        print(f"Error extracting logprobs: {e}")
        avg_logprob = float('-inf')
    
    return avg_logprob


def find_best_completion(completions):
    """
    Finds the best completion based on the highest average log probability.
    """
    best_avg_logprob = float('-inf')
    best_completion = None
    for parsed_content, avg_logprob in completions:
        if avg_logprob > best_avg_logprob:
            best_avg_logprob = avg_logprob
            best_completion = parsed_content
    return best_completion

def chat_with_llm(pre_prompt, llm, query, context):
    """
    Executes a language model chain that takes a pre-built prompt template, processes it through the LLM, and parses the result.
    
    - pre_prompt (ChatPromptTemplate): The prompt template used to format the input for the LLM.
    - llm (AzureChatOpenAI | Any LLM): The language model instance that generates the response.
    - query (str): The user's question or input.
    - context (str): The additional context provided from the search results from Azure Search.

    Returns:
    - str: The parsed response from the language model.
    - logprobs: Log probability information for each token in the response.
    """
    
    # Build the chain without StrOutputParser to retain access to metadata
    chain = pre_prompt | llm

    # Invoke the model with the query and context
    response = chain.invoke({"question": query, "context": context})

    # Manually parse the content with StrOutputParser
    parser = StrOutputParser()
    parsed_content = parser.parse(response.content)

    # Retrieve the logprobs from response metadata, if available
    logprobs = response.response_metadata.get("logprobs", None)
    avg_logprob = calculate_average_logprob(logprobs)

    
    return parsed_content, avg_logprob

@timer
def chat_with_llm_stream(pre_prompt, llm, query, context,container):
    """
    Streams the repons from a language model chain that takes a pre-built prompt template, processes it through the LLM, and parses the result.

    - pre_prompt (ChatPromptTemplate): The prompt template used to format the input for the LLM.
    - llm (AzureChatOpenAI | Any LLM): The language model instance that generates the response.
    - query (str): The user's question or input.
    - context (str): The additional context provided from the search results from Azure Search.

    Returns:
    - Any: Streams and returns the LLM response using Streamlit's `write_stream` function.
    """    

    chain = (
        pre_prompt 
        | llm  
        | StrOutputParser()  
    )
    with container:
        ans = st.write_stream(chain.stream({"question": query, "context": context}))

    return ans

def chat_chain_with_llm2(pre_prompt, llm, query, context, n, container):
    PRE_PROMPT_TEXT = '''
    and Use the following previous ChatGPT responses to determine the correct answer to the questions. 
    Make sure to respond in the same manner as I previously outlined.
    
    Previous ChatGPT responses:
    '''
    chat_variables = {}  # Create a dictionary to store the variables

    for i in range(1,  n + 1):
        chat_variables[f'chat{i}'] = f'chat{i} = {chat_with_llm(pre_prompt, llm, query, context)}'  # Assign values

    pre_prompt_chain = pre_prompt + PRE_PROMPT_TEXT 
    # Now you can access the variables like this:
    for var_name, value in chat_variables.items():
        pre_prompt_chain + str(value)
        

    ans = chat_with_llm_stream(pre_prompt_chain, llm, query, context,container)
    return ans


def chat_chain_with_llm(pre_prompt, llm, query, context, n, container, spinner_placeholder):

    PRE_PROMPT_TEXT = '''
    and Use the following previous ChatGPT responses to determine the correct answer to the questions. 
    Make sure to respond in the same manner as I previously outlined.
    
    Previous ChatGPT responses:
    '''
    chat_variables = {}  # Create a dictionary to store the variables
    completions = []

    # Generate `n` completions and store their parsed content and average log probabilities
    with spinner_placeholder.container():
        with st.spinner("Generating completions..."):
                for i in range(1, n + 1):
                    parsed_content, avg_logprob = chat_with_llm(pre_prompt, llm, query, context)
                    completions.append((parsed_content, avg_logprob))
                    chat_variables[f'chat{i}'] = parsed_content

    with spinner_placeholder.container():
        with st.spinner("Finding best completion..."):
            best_completion = find_best_completion(completions)

        pre_prompt_chain = pre_prompt + PRE_PROMPT_TEXT 
        # Now you can access the variables like this:
        for var_name, value in chat_variables.items():
            pre_prompt_chain + str(value)
    


    print(f"Based on the avg logprob over the completion this is the best completion: {best_completion}")
    pre_prompt_chain + " based on the avg logprob over the completion this is the best completion " + best_completion

    # Final step: Generate the final answer using chat_with_llm_stream
    with spinner_placeholder.container():
        with st.spinner("Generating final answer..."):
            ans = chat_with_llm_stream(pre_prompt_chain, llm, query, context, container)

    return ans


 

@functools.lru_cache(maxsize=32)
@timer
def download_pdf(pdf_url):
    """
    Downloads a PDF from a URL and returns it as a PyMuPDF document.

    - pdf_url (str): The URL of the PDF to download.

    Returns:
    - fitz.Document | None: The PyMuPDF document object representing the PDF, or None if the download fails.
    """
    response = requests.get(pdf_url)
    if response.status_code != 200:
        return None
    pdf_data = BytesIO(response.content)
    return fitz.open(stream=pdf_data, filetype="pdf")



@timer
def find_search_term_in_pdf_from_url(pdf_url, search_term):
    """
    Searches for a term within a PDF retrieved from a URL and returns the pages where the term is found.

    - pdf_url (str): The URL of the PDF to search within.
    - search_term (str): The term to search for in the PDF.

    Returns:
    - List[int] | None: A list of page numbers (1-based) where the term is found, or None if not found.
    """
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
    """
    Reformulates a question into four different variations to improve search accuracy using a language model.

    - llm (AzureChatOpenAI | Any LLM): The language model instance used to reformulate the query.
    - query (str): The original query provided by the user.
    - context (str): The context given to the LLM to assist in the reformulation process.

    Returns:
    - List[str]: A list of four reformulated versions of the original query.
    """
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


  
from azure.storage.blob import BlobServiceClient

def get_doc_from_blob(blob_c_name):
    # env_path = os.path.expanduser('~/cloudfiles/code/GPT-Azure-Search-Engine/credentials.env')
    # load_dotenv(env_path)
    load_dotenv("credentials.env")

    # Replace with your connection string and container name
    connection_string = os.getenv("BLOB_CONNECTION_STRING")
    container_name = blob_c_name
   
    

    # Create a BlobServiceClient object to interact with the blob service
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)

    # Create a ContainerClient object to interact with the specific container
    container_client = blob_service_client.get_container_client(container_name)

    # List the blobs in the container
    blob_list = container_client.list_blobs()

    # Collect blob names in a list
    blob_names = []
    for blob in blob_list:
        blob_names.append(blob.name)
        
        
    # Check if any blobs were found and print a message
    if not blob_names:
        print("No blobs found in the container.")

    # Return the list of blob names
    return blob_names

