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

def get_search_results(query: str, indexes: list, 
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
                                        "index": index
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

def chat_with_llm_stream(pre_prompt, llm, query, context):
    # Create a placeholder for streaming the response


    # Chain processing
    chain = (
        pre_prompt  # Passes the variables to the prompt template
        | llm  # Use streaming feature of the LLM
        | StrOutputParser()  # Converts the output to a string
    )

    # Stream the response by invoking the chain
    st.write_stream(chain.stream({"question": query, "context": context}))






# Function to download the PDF from the URL, search for the term, and return the pages where it's found
def find_search_term_in_pdf_from_url(pdf_url, search_term):
    # Step 1: Download the PDF
    response = requests.get(pdf_url)
    if response.status_code != 200:
        return None  # Return None if download fails or handle error appropriately
    # Step 2: Open the downloaded PDF with PyMuPDF
    pdf_data = BytesIO(response.content)
    doc = fitz.open(stream=pdf_data, filetype="pdf")
    # Step 3: Search for the term in the PDF and store the page numbers where it's found
    
    pages_with_term = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")
        if search_term.lower() in text.lower():
            pages_with_term.append(page_num + 1)  # Store page numbers (1-based indexing)

    # Step 4: Close the PDF document
    doc.close()

    # Return the list of page numbers where the search term is found
    return pages_with_term