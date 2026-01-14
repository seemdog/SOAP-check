import pandas as pd
import ast
from openai import OpenAI
from tqdm import tqdm
import os
import pickle
import argparse
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import re 

# === Utility Functions ===
def open_txt(file_path: str):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


def process_pkl(total_rows: int, new_dir: str, file: str, error_type: str):
    """
    pkl 로드 + 이어서 처리할 인덱스 범위 반환
    항상 (list, range, pkl_path) 를 리턴하도록 정리
    """
    pkl_path = f'{new_dir}/{file[:-4]}_{error_type}.pkl'

    # 기존 결과 로드 (있으면 이어서 진행)
    try:
        with open(pkl_path, 'rb') as f:
            error_type_list = pickle.load(f)
        print(f"[{error_type}] Loaded existing data. Current length: {len(error_type_list)}")
    except Exception:
        error_type_list = []
        print(f"[{error_type}] No existing data to load. Starting from scratch.")

    error_type_start_idx = len(error_type_list)

    # 이미 다 끝난 상태면 pkl 지우고, 빈 range 리턴
    if error_type_start_idx >= total_rows:
        print(f"[{error_type}] All rows are already processed.")

        if os.path.exists(pkl_path):
            os.remove(pkl_path)
            print(f"[{error_type}] Checkpoint file removed: {pkl_path}")
            
        return error_type_list, range(0), pkl_path # 빈 range 반환

    # 남은 인덱스 범위
    error_type_indices = range(error_type_start_idx, total_rows)
    return error_type_list, error_type_indices, pkl_path

def safe_literal_eval(result: str):
    """
    모델이 ```json ... ``` 같은 코드블럭으로 감싸서 줄 때도
    최대안 파싱해보는 helper. 실패하면 None 리턴.
    """
    # 1차 시도
    try:
        return ast.literal_eval(result)
    except Exception:
        pass

    # 코드블럭 제거 후 2차 시도
    try:
        text = result.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            # 첫 줄이 ``` 또는 ```json 이면 제거
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            # 마지막 줄이 ``` 이면 제거
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        return ast.literal_eval(text)
    except Exception:
        return None

# [수정] temperature, seed 인자 추가
def chat_hallucination_omission(system, user, reference_unit, soap, model, temperature, seed):
    user_prompt = user.replace("{ORIGINAL_TEXT}", reference_unit)
    user_prompt = user_prompt.replace("{SUMMARY_TEXT}", soap)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, # 추가
        seed=seed                # 추가
    )
    return completion.choices[0].message.content

# [수정] temperature, seed 인자 추가
def chat_redundancy_miscategorization(system, user, soap, model, temperature, seed):
    user_prompt = user.replace("{SUMMARY_TEXT}", soap)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt},
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, # 추가
        seed=seed                # 추가
    )
    return completion.choices[0].message.content


# ===== [핵심 수정] 전처리 로직 (Greedy Regex 적용) =====

def preprocess_soap_data(soap_text: str):
    """
    SOAP 텍스트를 줄 단위로 분리 및 정제
    """
    lines = soap_text.split('\n')
    processed_lines = []
    
    current_section = None 
    pending_subheader = ""
    
    # 1. 헤더 감지용 패턴 (줄 시작 부분의 S, O, A, P 확인)
    header_detect_pattern = re.compile(r'^[\#\*\-\s]*([SOAP])(?:[\s\:\)\.]|\s*\(.*?\))', re.IGNORECASE)
    
    # 2. 헤더 내용 제거용 패턴 (Greedy Cleaning)
    clean_header_garbage_pattern = re.compile(
        r'^[\#\*\-\s]*[SOAP](?:[\s\*\:\.\)\-]|\s*\(.*?\))*', 
        re.IGNORECASE
    )

    section_map = {
        'S': 'S (Subjective)', 
        'O': 'O (Objective)', 
        'A': 'A (Assessment)', 
        'P': 'P (Plan)'
    }

    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue
        
        # 1. 메인 헤더(S/O/A/P) 감지
        match = header_detect_pattern.match(clean_line)
        if match:
            key = match.group(1).upper()
            current_section = section_map.get(key, f"{key} (Section)")
            pending_subheader = "" # 섹션 변경 시 소제목 버퍼 초기화
            
            # 헤더 찌꺼기 제거
            content_part = clean_header_garbage_pattern.sub('', clean_line).strip()
            
            # 남은 문자열 앞뒤의 불필요한 기호(:, *, -) 한 번 더 정리
            content_part = content_part.lstrip(':*- ').strip()

            # 유효한 내용이 있는 경우에만 추가
            if content_part and any(c.isalnum() for c in content_part):
                if content_part.endswith(':') and len(content_part) < 20:
                    pending_subheader = content_part
                else:
                    processed_lines.append(f"[{current_section}] {content_part}")
            continue

        # 아직 유효 섹션 진입 전이면 스킵
        if current_section is None:
            continue

        # 2. 소제목(Dangling Subheader) 처리
        if clean_line.endswith(':') and len(clean_line) < 30 and not clean_line.startswith('-'):
            pending_subheader = clean_line
            continue 
        
        # 3. 내용 결합 및 저장
        final_text = clean_line
        
        if pending_subheader:
            final_text = f"{pending_subheader} {clean_line}"
            if not clean_line.startswith('-'):
                pending_subheader = ""
        
        processed_lines.append(f"[{current_section}] {final_text}")
        
    return processed_lines


# ===== 멀티스레드용 워커들 (Params 추가) =====

MAX_RETRY = 5

def hallucination_worker(i, df, system, user, model, temperature, seed):
    reference_unit = df["reference_unit"][i]
    soap = df["SOAP"][i]

    # 개선된 전처리 함수 사용
    soap_lines = preprocess_soap_data(soap)
    
    # 프롬프트에 넣을 때는 다시 하나의 문자열로 합침
    soap_clean_str = "\n".join(soap_lines)

    hallucination = "ERROR"  # 기본값

    for _ in range(MAX_RETRY):
        try:
            result = chat_hallucination_omission(
                system, user, reference_unit, soap_clean_str, model, temperature, seed
            )
            parsed = safe_literal_eval(result) 
        except Exception:
            parsed = None

        # 형식/길이 체크
        if not isinstance(parsed, list) or len(parsed) != len(soap_lines):
            continue  # 재시도

        # '정보 없음' 포함된 것의 label 을 X로 변경
        for item in parsed:
            if isinstance(item, dict):
                text = item.get("text", "")
                if "정보 없음" in text:
                    item["label"] = "X"

        hallucination = parsed
        break 

    return i, hallucination

def omission_worker(i, df, system, user, model, temperature, seed):
    reference_unit = df["reference_unit"][i]
    soap = df["SOAP"][i].strip()
    
    omission = "ERROR" 

    for _ in range(MAX_RETRY):
        try:
            result = chat_hallucination_omission(
                system, user, reference_unit, soap, model, temperature, seed
            )
            parsed = safe_literal_eval(result)
        except Exception:
            parsed = None

        # 형식/길이 체크
        if not isinstance(parsed, list) or len(parsed) != len(reference_unit.strip().split("\n")):
            continue 

        omission = parsed
        break 

    return i, omission

def redundancy_worker(i, df, system, user, model, temperature, seed):
    soap = df["SOAP"][i].strip()

    redundancy = "ERROR" 

    for _ in range(MAX_RETRY):
        try:
            result = chat_redundancy_miscategorization(system, user, soap, model, temperature, seed)
            parsed = safe_literal_eval(result)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            redundancy = parsed
            break   

    return i, redundancy


def miscategorization_worker(i, df, system, user, model, temperature, seed):
    soap = df["SOAP"][i].strip()

    miscategorization = "ERROR" 

    for _ in range(MAX_RETRY):
        try:
            result = chat_redundancy_miscategorization(system, user, soap, model, temperature, seed)
            parsed = safe_literal_eval(result)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            miscategorization = parsed
            break

    return i, miscategorization

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-5.1')
    parser.add_argument('--file', type=str, default='evaluation_sample.csv')
    parser.add_argument('--max_workers', type=int, default=5)
    
    # [추가] Variance 제어를 위한 인자
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()

    model = args.model
    file = args.file
    max_workers = args.max_workers
    temperature = args.temperature
    seed = args.seed

    load_dotenv("./key.env")
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), timeout=20)

    data_dir = "./data/" + file
    df = pd.read_csv(data_dir)

    # 프롬프트 로드
    system_hallucination = open_txt("./prompt/system_hallucination.txt")
    user_hallucination = open_txt("./prompt/user_hallucination.txt")
    system_omission = open_txt("./prompt/system_omission.txt")
    user_omission = open_txt("./prompt/user_omission.txt")
    system_redundancy = open_txt("./prompt/system_redundancy.txt")
    user_redundancy = open_txt("./prompt/user_redundancy.txt")
    system_miscategorization = open_txt("./prompt/system_miscategorization.txt")
    user_miscategorization = open_txt("./prompt/user_miscategorization.txt")

    # 결과 저장 디렉토리
    new_dir = "./data/pkl"
    os.makedirs(new_dir, exist_ok=True)

    total_rows = len(df)

    # ==== pkl 로드 (재시작 가능) ====
    hallucination_list, hallucination_indices, hallucination_pkl_path = process_pkl(
        total_rows=total_rows, new_dir=new_dir, file=file, error_type="hallucination"
    )
    omission_list, omission_indices, omission_pkl_path = process_pkl(
        total_rows=total_rows, new_dir=new_dir, file=file, error_type="omission"
    )
    redundancy_list, redundancy_indices, redundancy_pkl_path = process_pkl(
        total_rows=total_rows, new_dir=new_dir, file=file, error_type="redundancy"
    )
    miscategorization_list, miscategorization_indices, miscategorization_pkl_path = process_pkl(
        total_rows=total_rows, new_dir=new_dir, file=file, error_type="miscategorization"
    )

    # ====== hallucination 병렬 처리 ======
    if len(list(hallucination_indices)) > 0:
        pending = {}
        next_to_write = len(hallucination_list)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    hallucination_worker, i, df, system_hallucination, user_hallucination, model,
                    temperature, seed # [추가] Params 전달
                ): i for i in hallucination_indices
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="hallucination"):
                i, result = future.result()
                pending[i] = result

                updated = False
                while next_to_write in pending:
                    hallucination_list.append(pending.pop(next_to_write))
                    next_to_write += 1
                    updated = True

                if updated:
                    with open(hallucination_pkl_path, 'wb') as f:
                        pickle.dump(hallucination_list, f)

    # ====== omission 병렬 처리 ======
    if len(list(omission_indices)) > 0:
        pending = {}
        next_to_write = len(omission_list)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    omission_worker, i, df, system_omission, user_omission, model,
                    temperature, seed # [추가] Params 전달
                ): i for i in omission_indices
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="omission"):
                i, result = future.result()
                pending[i] = result

                updated = False
                while next_to_write in pending:
                    omission_list.append(pending.pop(next_to_write))
                    next_to_write += 1
                    updated = True

                if updated:
                    with open(omission_pkl_path, 'wb') as f:
                        pickle.dump(omission_list, f)

    # ====== redundancy 병렬 처리 ======
    if len(list(redundancy_indices)) > 0:
        pending = {}
        next_to_write = len(redundancy_list)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    redundancy_worker, i, df, system_redundancy, user_redundancy, model,
                    temperature, seed # [추가] Params 전달
                ): i for i in redundancy_indices
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="redundancy"):
                i, result = future.result()
                pending[i] = result

                updated = False
                while next_to_write in pending:
                    redundancy_list.append(pending.pop(next_to_write))
                    next_to_write += 1
                    updated = True

                if updated:
                    with open(redundancy_pkl_path, 'wb') as f:
                        pickle.dump(redundancy_list, f)

    # ====== miscategorization 병렬 처리 ======
    if len(list(miscategorization_indices)) > 0:
        pending = {}
        next_to_write = len(miscategorization_list)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    miscategorization_worker, i, df,
                    system_miscategorization, user_miscategorization, model,
                    temperature, seed # [추가] Params 전달
                ): i for i in miscategorization_indices
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="miscategorization"):
                i, result = future.result()
                pending[i] = result

                updated = False
                while next_to_write in pending:
                    miscategorization_list.append(pending.pop(next_to_write))
                    next_to_write += 1
                    updated = True

                if updated:
                    with open(miscategorization_pkl_path, 'wb') as f:
                        pickle.dump(miscategorization_list, f)

    # ====== 결과 정리 & 저장 ======
    if len(hallucination_list) != total_rows:
        print(f"[WARN] hallucination_list length ({len(hallucination_list)}) != df length ({total_rows})")
    if len(omission_list) != total_rows:
        print(f"[WARN] omission_list length ({len(omission_list)}) != df length ({total_rows})")
    if len(redundancy_list) != total_rows:
        print(f"[WARN] redundancy_list length ({len(redundancy_list)}) != df length ({total_rows})")
    if len(miscategorization_list) != total_rows:
        print(f"[WARN] miscategorization_list length ({len(miscategorization_list)}) != df length ({total_rows})")

    df["hallucination"] = hallucination_list
    df["omission"] = omission_list
    df["redundancy"] = redundancy_list
    df["miscategorization"] = miscategorization_list

    df.to_csv(data_dir, index=False)
    print("Processing done and CSV saved.")

    # ====== pkl 삭제 (전부 끝난 경우) ======
    if len(hallucination_list) == total_rows and os.path.exists(hallucination_pkl_path):
        os.remove(hallucination_pkl_path)
        print(f"Checkpoint file removed: {hallucination_pkl_path}")

    if len(omission_list) == total_rows and os.path.exists(omission_pkl_path):
        os.remove(omission_pkl_path)
        print(f"Checkpoint file removed: {omission_pkl_path}")

    if len(redundancy_list) == total_rows and os.path.exists(redundancy_pkl_path):
        os.remove(redundancy_pkl_path)
        print(f"Checkpoint file removed: {redundancy_pkl_path}")

    if len(miscategorization_list) == total_rows and os.path.exists(miscategorization_pkl_path):
        os.remove(miscategorization_pkl_path)
        print(f"Checkpoint file removed: {miscategorization_pkl_path}")