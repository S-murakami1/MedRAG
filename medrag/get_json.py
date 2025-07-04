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

serp_api_key = os.getenv("SERPAPI_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")

all_summaries: str = ""

llm = ChatOpenAI(
    model="gpt-4o",
    api_key=openai_key,
)

# Create output directory if it doesn't exist
output_dir = "./output/eng"
os.makedirs(output_dir, exist_ok=True)

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
- Answer "{input}" accurately and precisely in 2-3 sentences max based on ## Input.


# Conditions
- Output should be in English.
- Provide URL citations to credible sources to substantiate every medical fact stated in your answer.
- Return a json where the key 'response' has the value of the text of the answer and the field 'citations' has the value of a list of citation URLs.

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

# Arxivデータの検索結果を取得
def search_arxiv(state: State):
    query = state["query"]
    #logger.info(f"search_arxiv query: {query}")
    list_arxiv = arxiv_tool_func.invoke(query)
    #logger.info(f"arxiv_outputs: {len(list_arxiv)} documents")
    #logger.debug(f"arxiv_outputs: {list_arxiv}")
    return {"arxiv_outputs": list_arxiv}


# PubMedデータの検索結果を取得
def search_pubmed(state: State):
    query = state["query"]
    logger.info(f"search_pubmed query: {query}")
    list_pubmed = pubmed_tool_func.invoke(query)
    #logger.info(f"pubmed_outputs: {len(list_pubmed)} documents")
    #logger.debug(f"pubmed_outputs: {list_pubmed}")
    return {"pubmed_outputs": list_pubmed}


# Webデータの検索結果を取得
def search_web(state: State):
    query = state["query"]
    logger.info(f"search_web query: {query}")
    list_web = web_tool_func.invoke(query)
    #logger.info(f"web_outputs: {len(list_web)} documents")
    #logger.debug(f"web_outputs: {list_web}")
    return {"web_outputs": list_web}



def format_output(doc_type, output):
    return f"## {doc_type}\n" + "\n".join([f"{doc['content']}(source: {doc['source']})" for doc in output])

def make_report(state: State):
    input = state["input"]
    report = state.get("report", "")

    arxiv_outputs = format_output("論文", state.get("arxiv_outputs", []))
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
        logger.error("検索結果がありません！")
        return {"report": "検索結果が見つかりませんでした。"}

    # 検索結果をCSVファイルに保存
    import pandas as pd
    from datetime import datetime
    
    # 入力クエリから有効なファイル名を生成
    filename = re.sub(r'[<>:"/\\|?*]', '_', input)[:50]  # 不正な文字を除去し長さを制限
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{output_dir}/search_results_{filename}_{timestamp}.csv"
    
    # 検索結果をデータフレームに変換
    data = []
    for doc in state.get("arxiv_outputs", []):
        data.append({"source": "Arxiv", "url": doc["source"], "content": doc["content"]})
    for doc in state.get("web_outputs", []):
        data.append({"source": "Web", "url": doc["source"], "content": doc["content"]})
    for doc in state.get("pubmed_outputs", []):
        data.append({"source": "PubMed", "url": doc["source"], "content": doc["content"]})
        
    df = pd.DataFrame(data)
    df.to_csv(csv_filename, index=False, encoding='utf-8')
    logger.info(f"Search results saved to {csv_filename}")

    prompt = ChatPromptTemplate.from_template(make_report_prompt_template)
    chain = create_stuff_documents_chain(llm, prompt)

    result = chain.invoke({"context": documents, "input": input})

    return {"report": result}


medical_questions = [
    "ACP（アドバンス・ケア・プランニング）とは何か？",
    "COPDと喘息の違いは？",
    "CRPが高値の場合、何が疑われる？",
    "CTとMRIの違いは？",
    "HbA1cの意味と基準値は？",
    "PTSDの診断基準を簡単に説明せよ。",
    "T細胞とB細胞の役割の違いは？",
    "うつ病の主な症状と診断基準は？",
    "がんの進行度（ステージ）とは何か？",
    "アセトアミノフェンとNSAIDsの違いは？",
    "アナフィラキシーの症状と治療は？",
    "アルコール依存症の特徴と治療は？",
    "インスリン製剤の種類と特徴は？",
    "インフォームド・コンセントとは何か？",
    "サルコペニアとは何か？",
    "ショックの分類と初期対応は？",
    "ステロイド薬の副作用を3つ挙げよ。",
    "チーム医療のメリットは？",
    "バイタルサインとは何か？",
    "パニック障害とはどのような病気か？",
    "フレイルとその予防法は？",
    "ホメオスタシスとは何か？",
    "ホルモンと酵素の違いは？",
    "ポリファーマシーとは何か？",
    "ワクチンが免疫を獲得する仕組みは？",
    "体温調節に関与する中枢はどこか？",
    "内視鏡検査で観察される主な消化管疾患は？",
    "副交感神経の主な作用は？",
    "動脈と静脈の構造的な違いは？",
    "医療安全で重要な「インシデント」とは？",
    "医療現場での標準予防策（スタンダードプリコーション）とは？",
    "医療行為における患者の自己決定権とは？",
    "向精神薬の分類と例を挙げよ。",
    "呼吸の一次的な調節中枢はどこにある？",
    "地域包括ケアシステムの概要は？",
    "多剤併用のリスクは何か？",
    "多職種連携で大切なことは？",
    "小腸で主に吸収される栄養素は？",
    "尿検査で蛋白尿が陽性となる原因は？",
    "心不全の左心系と右心系の症状は？",
    "心肺蘇生（CPR）の手順を説明せよ。",
    "心電図（ECG）のP波、QRS波、T波の意味は？",
    "感染対策で最も基本的な方法は？",
    "感染経路の種類と例を3つ挙げよ。",
    "抗がん剤の主な副作用は？",
    "抗原と抗体の関係を説明せよ。",
    "抗生物質耐性が起こるメカニズムは？",
    "敗血症とは何か？",
    "気道確保の基本手技は？",
    "狭心症と心筋梗塞の違いは？",
    "白血球数の増加が見られる病態は？",
    "睡眠障害の種類を3つ挙げよ。",
    "神経伝達物質にはどのような種類があるか？",
    "神経系は大きく分けて何系と何系に分けられるか？",
    "窒息時の応急処置は？",
    "糖尿病の種類と病態の違いは？",
    "細胞膜の構造と機能を説明せよ。",
    "終末期ケアで大切なことは？",
    "統合失調症の陽性症状と陰性症状の違いは？",
    "老年症候群とは？",
    "肝硬変の主な合併症は？",
    "肝臓の主な代謝機能は何か？",
    "肺で行われるガス交換のメカニズムは？",
    "肺炎と誤嚥性肺炎の違いは？",
    "胎児循環と出生後の循環の違いは？",
    "胸腔と腹腔を隔てている筋は？",
    "胸部X線写真で「すりガラス陰影」とは何か？",
    "脳梗塞と脳出血の鑑別点は？",
    "腎臓で尿が生成される過程を説明せよ。",
    "腎臓の主な機能は何か？",
    "腫瘍マーカーCEAとは何か？",
    "膠原病とは何か？",
    "自己免疫疾患が発症するメカニズムを説明せよ。",
    "自己免疫疾患とは何か？",
    "自殺リスクアセスメントで重要な視点は？",
    "自然免疫と獲得免疫の違いは？",
    "薬物の半減期とは何か？",
    "血圧を規定する要因を2つ挙げよ。",
    "血液の成分とその役割は？",
    "血液中の白血球数が上昇する主な原因は？",
    "血液凝固の基本的な仕組みは？",
    "血液検査でALTとASTが高値の場合、考えられる疾患は？",
    "血糖値を下げるホルモンは何か？",
    "褥瘡のリスク要因と予防策は？",
    "解剖学的正位とは何か？",
    "認知症患者へのコミュニケーションの工夫は？",
    "認知症高齢者のBPSDとは何か？",
    "認知行動療法（CBT）とは？",
    "誤嚥性肺炎の診断基準は？",
    "肝がんについて説明せよ。",
    "リンパ腫の病理所見について述べよ" ,
    "分子標的治療薬について説明せよ",
    "細胞の放射線感受性について説明せよ",
    "転移性脳腫瘍について説明せよ",
    "膵がんの予後は？",
]

if __name__ == "__main__":
    from tqdm import tqdm
    import random
    # ランダムに12個の質問を選択
    med_list = random.sample(medical_questions, 12)
    for index, input_query in tqdm(enumerate(med_list), total=len(med_list), desc="Processing..."):
        # Graphの作成
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
        #parser = argparse.ArgumentParser()
        #parser.add_argument("--input", "-i", type=str, required=True)
        #args = parser.parse_args()
        result = graph.invoke({"input": input_query, "report": ""})
        logger.info(f"result: {result["report"]}")

