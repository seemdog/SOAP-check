import pandas as pd
import ast
from tqdm import tqdm
import os
import argparse

def percentage_n_or_error(value):
    # value가 "ERROR" 문자열인 경우 그대로 반환

    if value == "ERROR":
        return "ERROR"
    
    # 정상적으로 리스트가 들어온 경우
    try:
        value = ast.literal_eval(value)
        count = sum(1 for item in value if item.get("label") == "N")
        return count/len(value)
        
    except Exception:
        # 혹시 예기치 않은 포맷인 경우도 방어적으로 ERROR 처리
        return "ERROR"


def count_n_or_error(value):
    # value가 "ERROR" 문자열인 경우 그대로 반환

    if value == "ERROR":
        return "ERROR"
    
    # 정상적으로 리스트가 들어온 경우
    try:
        value = ast.literal_eval(value)
        count = sum(1 for item in value if item.get("label") == "N")
        return count
        
    except Exception:
        # 혹시 예기치 않은 포맷인 경우도 방어적으로 ERROR 처리
        return "ERROR"


def count(value):
    # value가 "ERROR" 문자열인 경우 그대로 반환

    if value == "ERROR":
        return "ERROR"
    
    # 정상적으로 리스트가 들어온 경우
    try:
        value = ast.literal_eval(value)
        count = len(value)
        return count
        
    except Exception:
        # 혹시 예기치 않은 포맷인 경우도 방어적으로 ERROR 처리
        return "ERROR"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='eval_gemini.csv')

    args = parser.parse_args()

    file = args.file

    data_dir = "./data/" + file
    
    df = pd.read_csv(data_dir)

    df["hallucination_percentage"] = df["hallucination"].apply(percentage_n_or_error)
    df["hallucination_count"] = df["hallucination"].apply(count_n_or_error)
    df["omission_percentage"] = df["omission"].apply(percentage_n_or_error)
    df["omission_count"] = df["omission"].apply(count_n_or_error)
    df["redundancy_count"] = df["redundancy"].apply(count)
    df["miscategorization_count"] = df["miscategorization"].apply(count)

    df.to_csv(data_dir, index = False)
    
    