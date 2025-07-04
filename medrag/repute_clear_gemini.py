# -*- coding: utf-8 -*- # Added encoding declaration for clarity

import google.generativeai as genai
import google.api_core.exceptions # For Gemini specific exceptions
import fitz  # PyMuPDF
import os
import pandas as pd
from itertools import combinations
from loguru import logger
import time

# --- Configuration ---
# !!! IMPORTANT: Replace with your actual Google AI Gemini API Key !!!
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Choose the Gemini model to use
GEMINI_MODEL = "gemini-2.0-flash" # Or "gemini-1.5-pro-latest", etc.
# Directory containing PDF files
PDF_DIRECTORY = "./output/pdf"
# Output Excel file name
OUTPUT_EXCEL = "評価結果_gemini.xlsx" # Changed filename slightly
# Wait time in seconds when rate limit is hit (Gemini API might not specify exact time)
RATE_LIMIT_WAIT_SECONDS = 60
# --- End Configuration ---

# Configure the Gemini API Key
try:
    genai.configure(api_key=GEMINI_API_KEY)
    logger.info("Gemini API Key configured.")
except Exception as e:
    logger.error(f"Failed to configure Gemini API Key. Please check if the key is correct and valid. Error: {e}")
    # Optionally exit if the API key is critical for startup
    # exit()

def extract_text_from_pdf(file_path):
    """Extracts all text from a PDF file."""
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        logger.success(f"Successfully extracted text from: {os.path.basename(file_path)}")
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {e}")
    return text

def extract_question(filename):
    """
    Extracts the question part from a filename formatted as 'number_question_type.pdf'.
    Assumes the second part delimited by '_' is the question.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    if len(parts) >= 2:
        return parts[1]
    logger.warning(f"Could not extract question from filename format: {filename}")
    return "" # Return empty string if format is not matched

def create_prompt(text1, text2, file1, file2):
    """Creates the prompt for the Gemini API evaluation."""
    # Assume the question is common and use file1's extracted question
    common_q = extract_question(file1)
    if not common_q:
        common_q = "[質問内容不明]" # Fallback if question extraction failed

    prompt = f"""
あなたはプロの教育者であり、文章評価の専門家です。

以下は2つのレポート（{file1}と{file2}）です。
どちらのレポートも同じ質問「{common_q}」に対する回答となっています。

## 指示
これらのレポートを以下の5項目で評価してください (5点が最も良い)。
- Completeness (1-5の整数)
- Lack_of_false_information (1-5の整数)
- Evidence (1-5の整数)
- Appropriateness (1-5の整数)
- Relevance (1-5の整数)

## 制約条件
- 評価は必ず1から5の整数値で行ってください。0点は付けないでください。

## 【レポート1】
{text1}

## 【レポート2】
{text2}

## 出力形式 (この形式に厳密に従ってください):
レポート1の評価:
Completeness: 数値
Lack_of_false_information: 数値
Evidence: 数値
Appropriateness: 数値
Relevance: 数値

レポート2の評価:
Completeness: 数値
Lack_of_false_information: 数値
Evidence: 数値
Appropriateness: 数値
Relevance: 数値
"""
    # logger.info(f"Generated prompt for evaluating {file1} and {file2}")
    # Uncomment below if you want to log the full prompt (can be very long)
    # logger.debug(f"Prompt:\n{prompt}")
    return prompt

def evaluate_reports(file1, file2):
    """Evaluates two reports using the Gemini API."""
    logger.info(f"Starting evaluation for: {os.path.basename(file1)} vs {os.path.basename(file2)}")
    text1 = extract_text_from_pdf(file1)
    text2 = extract_text_from_pdf(file2)

    # Handle cases where text extraction might fail
    if not text1:
        logger.error(f"Could not extract text from {file1}. Skipping evaluation for this pair.")
        return None
    if not text2:
        logger.error(f"Could not extract text from {file2}. Skipping evaluation for this pair.")
        return None

    prompt = create_prompt(text1, text2, os.path.basename(file1), os.path.basename(file2))

    model = genai.GenerativeModel(GEMINI_MODEL)

    while True:
        try:
            logger.info(f"Sending request to Gemini model ({GEMINI_MODEL})...")
            # Configure safety settings to be less restrictive if needed,
            # but be aware of the implications. Default settings are generally recommended.
            # Example:
            # safety_settings = [
            #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            # ]
            # response = model.generate_content(prompt, safety_settings=safety_settings)

            response = model.generate_content(prompt)

            # Check if the response was blocked due to safety settings or other issues
            if not response.parts:
                 logger.warning(f"Gemini response was empty or blocked. Prompt text may have triggered safety filters. Blocked reason: {response.prompt_feedback.block_reason}")
                 # Decide how to handle blocked responses - here we return None to skip
                 return None # Or retry, or modify prompt, etc.

            output = response.text
            logger.success("Received response from Gemini.")
            logger.debug(f"Gemini Raw Output:\n{output}")

            # Basic check if the output seems valid (contains expected keywords)
            # This is a simple heuristic, might need refinement.
            if "レポート1の評価" not in output or "レポート2の評価" not in output or "Completeness" not in output:
                 logger.warning("Gemini output format might be incorrect. Retrying...")
                 time.sleep(5) # Short delay before retry
                 continue

            return output

        except google.api_core.exceptions.ResourceExhausted as e:
            # Handle rate limiting specifically
            logger.warning(f"Gemini API Rate limit reached. Waiting for {RATE_LIMIT_WAIT_SECONDS} seconds before retrying. Error: {e}")
            time.sleep(RATE_LIMIT_WAIT_SECONDS)
        except Exception as e:
            # Handle other potential API errors or issues
            logger.error(f"An error occurred during Gemini API call: {e}")
            logger.error("Skipping evaluation for this pair due to error.")
            return None # Skip this pair if a persistent error occurs


def parse_evaluation(gpt_output):
    """Parses the structured text output from the AI into a dictionary."""
    scores = {}
    current_report_key = None
    expected_metrics = ["Completeness", "Lack_of_false_information", "Evidence", "Appropriateness", "Relevance"]

    lines = gpt_output.strip().split('\n')
    line_iter = iter(lines)

    try:
        while True:
            line = next(line_iter).strip()

            if line.startswith('レポート1の評価:'):
                current_report_key = "レポート1"
                scores[current_report_key] = {}
            elif line.startswith('レポート2の評価:'):
                current_report_key = "レポート2"
                scores[current_report_key] = {}
            elif ':' in line and current_report_key:
                parts = line.split(':', 1) # Split only on the first colon
                metric = parts[0].strip()
                value_str = parts[1].strip()

                if metric in expected_metrics:
                    try:
                        # Attempt to convert to integer, ensure it's within 1-5
                        score_val = int(value_str)
                        if 1 <= score_val <= 5:
                            scores[current_report_key][metric] = score_val
                        else:
                            logger.warning(f"Parsed score '{score_val}' for '{metric}' in '{current_report_key}' is outside the valid range (1-5). Storing as is, but check output.")
                            scores[current_report_key][metric] = score_val # Store invalid number for review
                    except ValueError:
                        logger.warning(f"Could not parse '{value_str}' as an integer for '{metric}' in '{current_report_key}'. Storing raw value.")
                        scores[current_report_key][metric] = value_str # Store the raw string if not an int
                else:
                    logger.warning(f"Unexpected metric '{metric}' found in report '{current_report_key}'. Ignoring.")

            # Basic check to prevent infinite loops if parsing fails badly
            if current_report_key and len(scores.get(current_report_key, {})) >= len(expected_metrics):
                if current_report_key == "レポート1":
                     # Move to check for Report 2 header
                    pass
                elif current_report_key == "レポート2":
                     # Finished parsing Report 2, break loop
                     break

    except StopIteration:
        # Reached end of lines
        pass
    except Exception as e:
        logger.error(f"Error parsing evaluation output: {e}\nOutput was:\n{gpt_output}")
        return {} # Return empty dict on parsing error

    # Validation: Check if both reports and all metrics were found
    if "レポート1" not in scores or len(scores["レポート1"]) != len(expected_metrics):
        logger.error(f"Failed to parse all metrics for レポート1. Found: {scores.get('レポート1', {})}")
        # Optionally return {} or handle partial data
    if "レポート2" not in scores or len(scores["レポート2"]) != len(expected_metrics):
        logger.error(f"Failed to parse all metrics for レポート2. Found: {scores.get('レポート2', {})}")
        # Optionally return {} or handle partial data

    return scores


def main_evaluation(pdf_dir):
    """Main function to orchestrate the PDF evaluation process."""
    if not os.path.isdir(pdf_dir):
        logger.error(f"PDF directory not found: {pdf_dir}")
        return

    # Load existing results if the Excel file exists
    existing_results = {}
    if os.path.exists(OUTPUT_EXCEL):
        try:
            df_existing = pd.read_excel(OUTPUT_EXCEL)
            # Use filename as the key for easier lookup
            existing_results = {row["ファイル名"]: dict(row) for _, row in df_existing.iterrows() if pd.notna(row["ファイル名"])}
            logger.info(f"Loaded {len(existing_results)} existing results from {OUTPUT_EXCEL}")
        except Exception as e:
            logger.error(f"Error reading existing results file {OUTPUT_EXCEL}: {e}. Starting fresh.")
            existing_results = {} # Reset if file is corrupted

    # Group PDF files by extracted question
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    question_groups = {}
    for pdf_file in pdf_files:
        question = extract_question(pdf_file)
        if question:  # Only process files where a question could be extracted
            if question not in question_groups:
                question_groups[question] = []
            question_groups[question].append(os.path.join(pdf_dir, pdf_file))
        else:
             logger.warning(f"Skipping file due to inability to extract question: {pdf_file}")

    results = existing_results.copy()
    total_comparisons_attempted = 0
    new_evaluations_performed = 0

    # Compare files within each question group (expecting exactly 2 files per group)
    for question, files in question_groups.items():
        if len(files) != 2:
            logger.warning(f"Question '{question}' has {len(files)} files, expected 2. Skipping this group.")
            continue # Skip groups that don't have exactly two files

        file1, file2 = files
        total_comparisons_attempted += 1

        file1_name = os.path.basename(file1)
        file2_name = os.path.basename(file2)

        # Check if BOTH files already have results
        if file1_name in results and file2_name in results:
            logger.info(f"Skipping comparison for '{question}': Results for {file1_name} and {file2_name} already exist.")
            continue

        logger.info(f"--- Evaluating Question: '{question}' ---")
        logger.info(f"Comparing: {file1_name} vs {file2_name}")

        # Perform the evaluation via Gemini API
        gpt_output = evaluate_reports(file1, file2)

        if gpt_output:
            scores = parse_evaluation(gpt_output)

            # Check if parsing was successful and returned the expected structure
            if "レポート1" in scores and "レポート2" in scores:
                new_evaluations_performed += 1
                # Add/Update results for file 1 (only if not already present)
                if file1_name not in results:
                    results[file1_name] = {
                        "ファイル名": file1_name,
                        "質問": question, # Add question for context
                        "Completeness": scores["レポート1"].get("Completeness", "N/A"),
                        "Lack_of_false_information": scores["レポート1"].get("Lack_of_false_information", "N/A"),
                        "Evidence": scores["レポート1"].get("Evidence", "N/A"),
                        "Appropriateness": scores["レポート1"].get("Appropriateness", "N/A"),
                        "Relevance": scores["レポート1"].get("Relevance", "N/A")
                    }
                    logger.info(f"Added evaluation results for: {file1_name}")
                    logger.info(f"  Scores: {scores['レポート1']}")


                # Add/Update results for file 2 (only if not already present)
                if file2_name not in results:
                     results[file2_name] = {
                        "ファイル名": file2_name,
                        "質問": question, # Add question for context
                        "Completeness": scores["レポート2"].get("Completeness", "N/A"),
                        "Lack_of_false_information": scores["レポート2"].get("Lack_of_false_information", "N/A"),
                        "Evidence": scores["レポート2"].get("Evidence", "N/A"),
                        "Appropriateness": scores["レポート2"].get("Appropriateness", "N/A"),
                        "Relevance": scores["レポート2"].get("Relevance", "N/A")
                    }
                     logger.info(f"Added evaluation results for: {file2_name}")
                     logger.info(f"  Scores: {scores['レポート2']}")

                 # Save intermediate results after each successful comparison
                try:
                    df_results = pd.DataFrame(list(results.values()))
                    # Define column order for consistency
                    column_order = ["ファイル名", "質問", "Completeness", "Lack_of_false_information", "Evidence", "Appropriateness", "Relevance"]
                    # Reindex df to ensure consistent column order, adding missing cols if needed
                    df_results = df_results.reindex(columns=column_order)
                    df_results.to_excel(OUTPUT_EXCEL, index=False)
                    logger.success(f"Intermediate results saved to {OUTPUT_EXCEL}")
                except Exception as e:
                    logger.error(f"Failed to save intermediate results to Excel: {e}")
            else:
                logger.error(f"Failed to parse evaluation scores for pair: {file1_name}, {file2_name}. Skipping update for this pair.")
        else:
            logger.warning(f"Evaluation function returned no output for pair: {file1_name}, {file2_name}. Skipping.")

    # Final save after processing all groups
    try:
        df_final = pd.DataFrame(list(results.values()))
        # Define column order for consistency
        column_order = ["ファイル名", "質問", "Completeness", "Lack_of_false_information", "Evidence", "Appropriateness", "Relevance"]
        # Reindex df to ensure consistent column order, adding missing cols if needed
        df_final = df_final.reindex(columns=column_order)
        df_final.to_excel(OUTPUT_EXCEL, index=False)
        logger.success(f"Final results saved to {OUTPUT_EXCEL}")
    except Exception as e:
        logger.error(f"Failed to save final results to Excel: {e}")


    print(f"\n--- Evaluation Summary ---")
    print(f"Processed {total_comparisons_attempted} pairs of files.")
    print(f"Performed {new_evaluations_performed} new evaluations.")
    print(f"Total results count: {len(results)}")
    print(f"Results saved to: {OUTPUT_EXCEL}")
    print(f"--- Done ---")


# --- Main Execution ---
if __name__ == "__main__":
    # Configure logging
    log_file_path = "evaluation_log_gemini.log"
    logger.add(log_file_path, rotation="1 MB", level="INFO") # Log info and above to file
    logger.info("Starting PDF evaluation script using Google Gemini API.")

    # Basic check for API Key placeholder
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY" or not GEMINI_API_KEY :
         logger.error("Gemini API Key is not set. Please edit the script and set the 'GEMINI_API_KEY' variable.")
    else:
        main_evaluation(PDF_DIRECTORY)

    logger.info("Evaluation script finished.")