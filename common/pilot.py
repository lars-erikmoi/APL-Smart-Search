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

# Set up basic configuration for logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def debug(func):
    """A decorator that logs the function signature and return value"""
    @functools.wraps(func)
    def wrapper_debug(*args, **kwargs):
        args_repr = [repr(a) for a in args]  # Represent each argument
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]  # Represent each keyword argument
        signature = ", ".join(args_repr + kwargs_repr)
        logging.debug(f"Calling {func.__name__}({signature})")
        
        result = func(*args, **kwargs)
        
        logging.debug(f"{func.__name__!r} returned {result!r}")
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
        logging.debug(f"{func.__name__!r} executed in {runtime:.4f} seconds")
        return result
    return wrapper_timer




def extract_all_page_numbers(chunk_text):
    # Regular expression to match "Page X of Y" and extract all page numbers
    page_matches = list(re.finditer(r'Page (\d+) of \d+', chunk_text, re.IGNORECASE))
    # Create a list of tuples (page_number, position) for each page number found
    page_numbers = [(int(match.group(1)), match.start()) for match in page_matches]
    return page_numbers
def select_relevant_page_number(chunk_text, caption_text):
    # Extract all page numbers and their positions in the chunk
    page_numbers = extract_all_page_numbers(chunk_text)
    if not page_numbers:
        return None

    # Split caption text into word groups of 4 words each
    words = caption_text.split()
    word_groups = [' '.join(words[i:i + 4]) for i in range(len(words) - 3)]
    filtered_groups = [group for group in word_groups if not re.search(r'[^a-zA-Z0-9.,:\s]', group)]

    print(f"Filtered word groups: {filtered_groups}")

    # Find the position of the first match of the word group in the chunk
    for group in filtered_groups:
        match = re.search(re.escape(group), chunk_text, re.IGNORECASE)
        if match:
            group_position = match.start()
            # Determine the closest page number to this word group
            print(f"Matched group: '{group}' at position {group_position}")

            closest_page = None
            closest_distance = float('inf')
            for page_num, page_pos in page_numbers:
                distance = abs(group_position - page_pos)
                print(f"Page {page_num} at position {page_pos} has distance {distance}")
                if distance < closest_distance:
                    closest_distance = distance
                    closest_page = page_num
                    print(f"New closest page: {closest_page} with distance: {closest_distance}")

            print(f"Selected page number: {closest_page}")

            return closest_page +1

    return None

def find_page_number_in_captions(chunk_text, caption_text):
    # Select the most relevant page number based on proximity
    relevant_page_number = select_relevant_page_number(chunk_text, caption_text)
    if relevant_page_number == None:
        # If no relevant page number is found, return the first page number
        page_numbers = extract_all_page_numbers(chunk_text)
        if page_numbers:
            relevant_page_number = page_numbers[0][0] + 1
    return relevant_page_number

@timer
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

                #if result['location']:
                    #encoded_location = urllib.parse.quote(result['location'], safe='/:')
                content[result['id']]={
                                                "title": result['title'], 
                                                "name": result['name'], 
                                                "chunk": result['chunk'],
                                                "location":  result['location'] + page_url if result['location'] else "", #changed was + sas_token before if
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

def get_search_results_EXP(query: str, indexes: list, 
                       k: int = 5,
                       reranker_threshold: int = 1,
                       sas_token: str = "") -> List[dict]:
    """Performs multi-index hybrid search and returns ordered dictionary with the combined results"""

    headers = {'Content-Type': 'application/json', 'api-key': os.environ.get("AZURE_SEARCH_KEY", "")}
    params = {'api-version': os.environ.get('AZURE_SEARCH_API_VERSION', "")}

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



        resp = requests.post(os.environ.get('AZURE_SEARCH_ENDPOINT', '') + f"/indexes/{index}/docs/search",
                             data=json.dumps(search_payload), headers=headers, params=params)

        if resp.status_code != 200:
            st.error(f"Search request failed for index {index} with status code {resp.status_code}")
            continue

        #st.markdown(f"The response from the requests : {resp.text}")
    
        search_results = resp.json()
        if 'value' not in search_results:
            st.error(f"No search results found for index {index}")
            st.error(search_results.get('value', ''))
            st.error(search_results.get('error', ''))
            continue

        agg_search_results[index] = search_results

    content = dict()
    ordered_content = OrderedDict()

    for index,search_results in agg_search_results.items():
        for result in search_results['value']:
            if result['@search.rerankerScore'] > reranker_threshold: # Show results that are at least N% of the max possible score=4
                           # Try to extract caption text and search in the chunk
                content_part = str(result['@search.captions'][0]['text'])

                ###### EXPERIMENTAL: Try to extract page number from captions ######
                #page_number = find_page_number_in_captions(result['chunk'], content_part)
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
                        '''if filtered_groups:
                            search_term = urllib.parse.quote(random.choice(filtered_groups))
                            page_url = "#search=" + str(search_term)'''
                    else:
                        page_url = "#page=" + "1"  # Default to page 1 if no page is found


                        

                content[result['id']]={
                                        "title": result['title'], 
                                        "name": result['name'], 
                                        "chunk": result['chunk'],
                                        "location": result['location'] + page_url if result['location'] else "", #changed was + sas_token before if
                                        "caption": result['@search.captions'][0]['text'],
                                        "score": result['@search.rerankerScore'],
                                        "index": index,
                                    } 

    topk = k
    count = 0  # To keep track of the number of results added
    for id in sorted(content, key=lambda x: content[x]["score"], reverse=True):
        ordered_content[id] = content[id]
        count += 1
        if count >= topk:  # Stop after adding topK results
            break

    return ordered_content





def chat_with_llm(pre_prompt, llm, query, context):
    chain = (
        pre_prompt  # Passes the 4 variables above to the prompt template
        | llm   # Passes the finished prompt to the LLM
        | StrOutputParser()  # converts the output (Runnable object) to the desired output (string)
    )

    answer = chain.invoke({"question": query, "context":context})
    return answer
@timer
@debug
def chat_with_llm_stream(pre_prompt, llm, query, context):
    # Create a placeholder for streaming the response
    print("Chain processing...")
    print(f"Pre-prompt: {pre_prompt}")
    print(f"Query: {query}")
    print(f"Context: {context}")
    # Existing chain processing code

    # Chain processing
    chain = (
        pre_prompt  # Passes the variables to the prompt template
        | llm  # Use streaming feature of the LLM
        | StrOutputParser()  # Converts the output to a string
    )

    # Stream the response by invoking the chain
    st.write_stream(chain.stream({"question": query, "context": context}))




@functools.lru_cache(maxsize=32)
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
    I want you to reformulate the question so it will be formulated better when doing an semantic search. 
    I want you to give 4 diffrent suggestions and the answer i want back from you is in the format of 
    [reformulated question 1], [reformulated question 2], [reformulated question 3], [reformulated question 4]. 
    And it should be done exactly like that. 
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
