import openai
import fitz  # PyMuPDF
import os
import pandas as pd
from itertools import combinations
from loguru import logger
import time

# OpenAI APIキー
openai.api_key = "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"

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_question(filename):
    """
    ファイル名が「番号_質問内容_種類.pdf」の形式の場合、
    2番目の部分を質問として抽出する
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    if len(parts) >= 2:
        return parts[1]
    return ""

def create_prompt(text1, text2, file1, file2):
    # 両ファイルの質問は共通と仮定し、file1の質問内容を使用
    common_q = extract_question(file1)
    
    prompt = f"""
以下は2つのレポート（{file1}と{file2}）です。
どちらのレポートも同じ質問「{common_q}」に対する回答となっています。
##指示
これらのレポートを以下の6項目で評価してください(5点が最も良い)。
- Completeness (1-5の整数)
- Lack_of_false_information (1-5の整数)
- Evidence (1-5の整数)
- Appropriateness (1-5の整数)
- Relevance (1-5の整数)
- CLEAR (上記5つの合計)
##制約条件
- 0点は付けないでください。最低でも1点をつけてください。

##【レポート1】
{text1}

##【レポート2】
{text2}

## 出力形式
レポート1の評価:
Completeness:数値
Lack_of_false_information:数値
Evidence:数値
Appropriateness:数値
Relevance:数値
CLEAR:数値

レポート2の評価:
Completeness:数値
Lack_of_false_information:数値
Evidence:数値
Appropriateness:数値
Relevance:数値
CLEAR:数値
"""
    return prompt

def evaluate_reports(file1, file2):
    text1 = extract_text_from_pdf(file1)
    text2 = extract_text_from_pdf(file2)
    prompt = create_prompt(text1, text2, os.path.basename(file1), os.path.basename(file2))
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "あなたはプロの教育者であり、文章評価の専門家です。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        output = response['choices'][0]['message']['content']
        return output
    except openai.error.RateLimitError as e:
        wait_time = float(str(e).split('in ')[1].split('s.')[0])
        logger.warning(f"レートリミットに達しました。{wait_time}秒待機します。")
        time.sleep(wait_time + 1)  # 余裕を持って1秒追加
        return evaluate_reports(file1, file2)  # 再帰的に再試行
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        raise

def parse_evaluation(gpt_output):
    scores = {}
    current_report = None
    
    for line in gpt_output.split('\n'):
        line = line.strip()
        if line.startswith('レポート'):
            current_report = line.split('の')[0]
            scores[current_report] = {}
        elif ':' in line and current_report:
            metric, value = line.split(':')
            try:
                scores[current_report][metric] = int(value)
            except ValueError:
                scores[current_report][metric] = value.strip()
    
    return scores

def main_evaluation(pdf_dir):
    # 既存の評価結果を読み込む
    existing_results = {}
    if os.path.exists("評価結果_gpt_clear.xlsx"):
        df = pd.read_excel("評価結果_gpt_clear.xlsx")
        existing_results = {row["ファイル名"]: dict(row) for _, row in df.iterrows()}

    # PDFファイルを質問内容でグループ化
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    question_groups = {}
    for pdf_file in pdf_files:
        question = extract_question(pdf_file)
        if question:
            if question not in question_groups:
                question_groups[question] = []
            question_groups[question].append(os.path.join(pdf_dir, pdf_file))

    results = existing_results.copy()
    total_comparisons = 0
    
    # 同じ質問内容のファイル同士を比較（必ず2つずつ）
    for question, files in question_groups.items():
        if len(files) != 2:  # 2つのファイルがない場合はスキップ
            logger.warning(f"質問「{question}」のファイル数が2つではありません。スキップします。")
            continue
            
        file1, file2 = files
        total_comparisons += 1
        
        file1_name = os.path.basename(file1)
        file2_name = os.path.basename(file2)
        
        # 両方のファイルが既に評価済みの場合はスキップ
        if file1_name in results and file2_name in results:
            continue
            
        logger.info(f"質問「{question}」の評価を実行中")
        logger.info(f"比較: {file1_name} vs {file2_name}")
        
        gpt_output = evaluate_reports(file1, file2)
        scores = parse_evaluation(gpt_output)
        
        # ファイル1の結果を追加/更新（未評価の場合のみ）
        if file1_name not in results:
            results[file1_name] = {
                "ファイル名": file1_name,
                "Completeness": scores["レポート1"]["Completeness"],
                "Lack_of_false_information": scores["レポート1"]["Lack_of_false_information"],
                "Evidence": scores["レポート1"]["Evidence"],
                "Appropriateness": scores["レポート1"]["Appropriateness"],
                "Relevance": scores["レポート1"]["Relevance"],
                "CLEAR": scores["レポート1"]["CLEAR"]
            }
            
        # ファイル2の結果を追加/更新（未評価の場合のみ）
        if file2_name not in results:
            results[file2_name] = {
                "ファイル名": file2_name,
                "Completeness": scores["レポート2"]["Completeness"],
                "Lack_of_false_information": scores["レポート2"]["Lack_of_false_information"],
                "Evidence": scores["レポート2"]["Evidence"],
                "Appropriateness": scores["レポート2"]["Appropriateness"],
                "Relevance": scores["レポート2"]["Relevance"],
                "CLEAR": scores["レポート2"]["CLEAR"]
            }
        
        # 途中経過を保存
        df = pd.DataFrame(list(results.values()))
        df.to_excel("評価結果_gpt_clear.xlsx", index=False)
    
    print(f"完了：評価結果_gpt_clear.xlsx に保存しました。合計 {total_comparisons} 組を評価しました。")

# 実行
if __name__ == "__main__":
    pdf_directory = "./output/pdf"  # PDFファイルがあるディレクトリを指定
    main_evaluation(pdf_directory)
