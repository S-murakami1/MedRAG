import re
import time
import urllib.error

from langchain.tools import tool
from langchain_community.tools import TavilySearchResults
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.utilities.pubmed import PubMedAPIWrapper
from loguru import logger

# Get paper information using ArxivAPI
arxiv_tool = ArxivAPIWrapper(top_k_results=2, load_all_available_meta=True)

@tool
def arxiv_tool_func(query: str) -> list[dict[str, str]]:
    """Search Arxiv with a query."""
    try:
        docs = arxiv_tool.load(query)
    except AttributeError as e:
        logger.warning(f"No results found. Modifying query.")
        query = query.rsplit(" ", 1)[0]
        logger.info(f"changed search query: {query}")
        docs = arxiv_tool.load(query)

    # Create data for each paper
    result = []
    for doc in docs:
        url = doc.metadata.get("entry_id", "URL not found")
        summary = doc.metadata.get("Summary", "Summary not found")
        result.append({"source": url, "content": summary})

    return result


# Get web information using TavilyAPI
tavily_tool = TavilySearchResults(max_results=1)


@tool
def web_tool_func(query: str) -> list[dict[str, str]]:
    """Search Tavily with a query."""
    try:
        data = tavily_tool.run(query, search_depth="depth")
    except AttributeError as e:
        logger.warning(f"No results found. Modifying query.")
        query = query.rsplit(" ", 1)[0]
        logger.info(f"changed search query: {query}")
    # print(data)
    contents = [{"source": item["url"], "content": item["content"]} for item in data]
    return contents


# Get paper information using PubMed
pubmed_tool = PubMedAPIWrapper(top_k_results=20)

@tool
def pubmed_tool_func(query: str) -> list[dict[str, str]]:
    """Search PubMed with a query."""
    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            docs = pubmed_tool.load(query)
            break
        except (urllib.error.URLError, AttributeError) as e:
            if attempt == max_retries - 1:  # Last attempt
                logger.error(f"PubMed search failed: {str(e)}")
                return []
            logger.warning(f"Retrying PubMed search. Attempt: {attempt + 1}")
            if isinstance(e, AttributeError):
                query = query.rsplit(" ", 1)[0]
                logger.info(f"Modified query: {query}")
            time.sleep(retry_delay)

    # Create data for each paper
    result = []
    for doc in docs:
        url = f"https://pubmed.ncbi.nlm.nih.gov/{doc['uid']}"
        summary = doc["Summary"]
        result.append({"source": url, "content": summary})

    return result


# Get patent information using SerpAPI
def patents_func(query, api_key) -> list[dict[str, str]]:
    params = {"engine": "google_patents", "q": query, "api_key": api_key,  "num": 1}
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
    except AttributeError as e:
        logger.warning(f"No results found. Modifying query.")
        params["query"] = query.rsplit(" ", 1)[0]
        logger.info(f"changed search query: {query}")
        search = GoogleSearch(params)
        results = search.get_dict()
    summarises = []

    # Process search results
    patents = results.get("organic_results", [])
    for patent in patents:
        title = patent.get("title", "No title")
        snippet = patent.get("snippet", "No description")
        pdf = patent.get("pdf", "No url")
        summarises.append({"source": pdf, "content": snippet})
    return summarises

if __name__ == "__main__":
    result = pubmed_tool_func.invoke("glomerular disease classification")
    print(result)