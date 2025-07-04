import csv
import os
import argparse
import re
import markdown
import pdfkit
from fpdf import FPDF

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from typing_extensions import TypedDict
from loguru import logger

from get_data import arxiv_tool_func, web_tool_func, pubmed_tool_func

openai_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_ENDPOINT")

all_summaries: str = ""

llm = ChatOpenAI(
    model="gpt-4o",
    api_key=openai_key,
    base_url=base_url,
)

output_dir = "./output" 

class State(TypedDict):
    input: str
    query: str
    arxiv_outputs: list[dict[str, str]]
    web_outputs: list[dict[str, str]]
    pubmed_outputs: list[dict[str, str]]
    report: str

make_query_prompt_template = """
# Instructions
- Generate appropriate search queries to meet the requirements in "# Input".

# Conditions
- Output should be 1-5 English words.
- Words should be separated by single spaces.
- Search queries should be minimal and suitable for patent/paper searches.
- Do not include words unrelated to technology like "research", "trends".

# Input
{context}
"""

make_report_prompt_template = """
# Instructions
- Organize the content from "# Search Results" by trends and create a response about "{input}".
- Only use the structure from # Output Example as reference.

# Conditions
- Output should be in English.
- Use appropriate technical terms rather than direct translations.
- Do not display ``` markers.
- Include references for all information in the main text.
- List **urls** of used references under **References**.
- Output in markdown format.
- The main text should be around 1000 characters, excluding References.
- Response should be in English.

# Output Example```
### Overview
XXX
### XXX
-XXX[1]
-XXX[2, 3]
...
---
### References
1. url: <https://XXX>
2. url: <https://XXX>
...
```

# Input
{context}
"""

def make_query(state: State):
    input = state["input"]
    documents = [Document(page_content=input)]
    prompt = ChatPromptTemplate.from_template(make_query_prompt_template)
    chain = create_stuff_documents_chain(llm, prompt)
    logger.info(f"make_query input: {input}")
    result = chain.invoke({"context": documents})
    logger.info(f"input: {input}")
    logger.info(f"search query: {result}")
    return{"query": result}

# Get paper information from Arxiv
def search_arxiv(state: State):
    query = state["query"]
    list_arxiv = arxiv_tool_func.invoke(query)
    return {"arxiv_outputs": list_arxiv}


# Get paper information from PubMed
def search_pubmed(state: State):
    query = state["query"]
    logger.info(f"search_pubmed query: {query}")
    list_pubmed = pubmed_tool_func.invoke(query)
    return {"pubmed_outputs": list_pubmed}


# Get web information from Web
def search_web(state: State):
    query = state["query"]
    logger.info(f"search_web query: {query}")
    list_web = web_tool_func.invoke(query)
    return {"web_outputs": list_web}



def format_output(doc_type, output):
    return f"## {doc_type}\n" + "\n".join([f"{doc['content']}(source: {doc['source']})" for doc in output])

def make_report(state: State):
    input = state["input"]
    report = state.get("report", "")

    arxiv_outputs = format_output("Arxiv", state.get("arxiv_outputs", []))
    web_outputs = format_output("Web", state.get("web_outputs", []))
    pubmed_outputs = format_output("PubMed", state.get("pubmed_outputs", []))

    logger.info(f"arxiv_outputs' length: {len(arxiv_outputs)}")
    logger.info(f"web_outputs' length: {len(web_outputs)}")
    logger.info(f"pubmed_outputs' length: {len(pubmed_outputs)}")

    documents = []
    if arxiv_outputs.strip():
        documents.append(Document(page_content=arxiv_outputs))
    if web_outputs.strip():
        documents.append(Document(page_content=web_outputs))
    if pubmed_outputs.strip():
        documents.append(Document(page_content=pubmed_outputs))

    if not documents:
        logger.error("No search results found!")
        return {"report": "No search results found."}

    prompt = ChatPromptTemplate.from_template(make_report_prompt_template)
    chain = create_stuff_documents_chain(llm, prompt)

    result = chain.invoke({"context": documents, "input": input})


    return {"report": result}


def save_report_as_pdf(report_text: str, input_query: str, output_dir: str):
    # GPT prompt
    simple_prompt = f"""
    # Instructions
    - Organize the content from "# Search Results" by trends and create a response about "{input_query}".
    - Only use the structure from # Output Example as reference.
    # Conditions
    - Output should be in English.
    - Use appropriate technical terms rather than direct translations.
    - Do not display ``` markers.
    - Output in markdown format.
    - The main text should be around 1000 characters.
    - Response should be in English.
    # Output Example
    ### Overview
    XXX
    ### XXX
    -XXX[1]
    -XXX[2, 3]
    ...
    ```
"""

    response = llm.invoke(simple_prompt).content
    md = markdown.Markdown()
    html_rag = md.convert(report_text)
    html_simple = md.convert(response)

    # HTML template
    html_template = """
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: "Arial";
            }}
            h1, h2, h3, h4, h5, h6 {{
                font-size: 1.5em;
            }}
            a {{
                color: #0066cc;
            }}
        </style>
    </head>
    <body>
    {content}
    </body>
    </html>
    """
    
    html_rag_with_css = html_template.format(content=html_rag)
    html_simple_with_css = html_template.format(content=html_simple)

    # PDF generation
    options = {
        'encoding': 'UTF-8'
    }
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # remove space from input_query
    input_query = input_query.replace(" ", "_")
    pdfkit.from_string(html_rag_with_css, f"{output_dir}/{input_query}_rag.pdf", options=options)
    pdfkit.from_string(html_simple_with_css, f"{output_dir}/{input_query}_gpt.pdf", options=options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query", type=str, help="Input query")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory path")
    args = parser.parse_args()

    # Set query and output directory
    input_query = args.query if args.query else medical_questions[0]
    output_dir = args.output_dir

    # Create graph
    graph_builder = StateGraph(State)
    graph_builder.add_node("Make Query Agent", make_query)
    graph_builder.add_node("Arxiv Search Agent", search_arxiv) 
    graph_builder.add_node("Web Search Agent", search_web)
    graph_builder.add_node("PubMed Search Agent", search_pubmed)
    graph_builder.add_node("Make Report Agent", make_report)

    graph_builder.set_entry_point("Make Query Agent")
    graph_builder.set_finish_point("Make Report Agent")
    graph_builder.add_edge("Make Query Agent", "Arxiv Search Agent")
    graph_builder.add_edge("Make Query Agent", "Web Search Agent")
    graph_builder.add_edge("Make Query Agent", "PubMed Search Agent")
    graph_builder.add_edge("Arxiv Search Agent", "Make Report Agent")
    graph_builder.add_edge("Web Search Agent", "Make Report Agent")
    graph_builder.add_edge("PubMed Search Agent", "Make Report Agent")

    graph = graph_builder.compile()
    graph.get_graph().print_ascii()
    result = graph.invoke({"input": input_query, "report": ""})
    save_report_as_pdf(result["report"], input_query, output_dir)

