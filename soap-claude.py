import pandas as pd
import ast
import anthropic  # [변경] OpenAI 대신 anthropic 임포트
from tqdm import tqdm
import os
import pickle
import argparse
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed


# === Utility Functions ===
def open_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# [수정 1] OpenAI -> Anthropic API 호출로 변경
# 주의: Claude는 현재 'seed' 파라미터를 지원하지 않으므로 인자는 받되 실제 호출엔 쓰지 않습니다.
def chat(system, user, content, model, temperature=0, seed=42): 

    user = user.replace("{TRANSCRIPTION}", content)

    # Claude API는 System 프롬프트를 messages 리스트가 아닌 별도 파라미터로 받습니다.
    # User 메시지 구성
    messages = [
        {
            "role": "user",
            "content": user
        }
    ]

    try:
        response = client.messages.create(
            model=model,
            max_tokens=4096,  # [추가] Claude API는 max_tokens가 필수입니다.
            system=system,    # [변경] 시스템 프롬프트 별도 지정
            messages=messages,
            temperature=temperature
            # seed=seed       # [참고] Claude는 현재 seed 파라미터를 지원하지 않습니다.
        )
        # 응답 추출 방식 변경
        return response.content[0].text
        
    except Exception as e:
        print(f"Error during API call: {e}")
        return ""


def worker(i, text, system_soap, user_soap, model, temperature, seed):
    """한 row 처리용 워커 함수 (멀티스레드에서 사용)"""
    # chat 함수 내부 로직만 바뀌었으므로 worker는 그대로 유지해도 됩니다.
    soap = chat(system_soap, user_soap, content=text, model=model, temperature=temperature, seed=seed)
    return i, soap


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # [변경] 기본 모델을 Claude 모델명으로 변경 (예: claude-3-5-sonnet-20241022)
    parser.add_argument('--model', type=str, default='claude-3-5-sonnet-20241022')
    parser.add_argument('--file', type=str, default='test.csv')
    parser.add_argument('--max_workers', type=int, default=5)  # Claude API Rate Limit 고려하여 조절 필요
    
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()

    model = args.model
    file = args.file
    max_workers = args.max_workers
    temperature = args.temperature
    seed = args.seed

    # 키 로드 & 클라이언트 생성
    load_dotenv("./key.env")
    
    # [변경] Anthropic 클라이언트 생성
    # key.env 파일 안에 ANTHROPIC_API_KEY 가 있어야 합니다.
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in environment variables.")
        
    client = anthropic.Anthropic(api_key=api_key)

    # 데이터 로드
    data_dir = "./data/" + file
    data_dir_save = "./data/" + model + "_" + file
    df = pd.read_csv(data_dir)

    # 프롬프트 텍스트 로드
    system_soap = open_txt("./prompt/system_soap.txt")
    user_soap = open_txt("./prompt/user_soap.txt")

    # 결과 저장 디렉토리
    new_dir = "./data/pkl"
    os.makedirs(new_dir, exist_ok=True)

    pkl_path = f'{new_dir}/{model}_{file[:-4]}_soap.pkl'

    # 기존 결과 로드 (있으면 이어서 진행)
    try:
        with open(pkl_path, 'rb') as f:
            soap_list = pickle.load(f)
        print(f"Loaded existing data. Current length: {len(soap_list)}")
    except Exception:
        soap_list = []
        print("No existing data to load. Starting from scratch.")

    start_idx = len(soap_list)
    total_rows = len(df)

    # 이미 다 끝난 상태면 pkl 지우고 종료
    if start_idx >= total_rows:
        print("All rows are already processed.")

        if os.path.exists(pkl_path):
            os.remove(pkl_path)
            print(f"Checkpoint file removed: {pkl_path}")
    else:
        indices = list(range(start_idx, total_rows))

        # 멀티스레드 실행
        pending = {}              # 아직 soap_list에 안 붙인 결과들 (i -> soap)
        next_to_write = start_idx  # 다음에 soap_list에 붙여야 할 인덱스

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    worker,
                    i,
                    df["transcription"][i],
                    system_soap,
                    user_soap,
                    model,
                    temperature,
                    seed
                ): i for i in indices
            }

            for future in tqdm(as_completed(futures), total=len(futures)):
                i, soap = future.result()
                pending[i] = soap

                # 가능한 한 인덱스 순서대로 soap_list에 append
                updated = False
                while next_to_write in pending:
                    soap_list.append(pending.pop(next_to_write))
                    next_to_write += 1
                    updated = True

                # 순서대로 새로 붙은 게 있을 때마다 체크포인트 저장
                if updated:
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(soap_list, f)

        # 최종적으로 CSV 저장
        if len(soap_list) != total_rows:
            print(f"Warning: soap_list length ({len(soap_list)}) != df length ({total_rows})")
        else:
            print("All rows processed successfully.")

        df["SOAP"] = soap_list
        df.to_csv(data_dir_save, index=False) 
        print(f"Processing done and CSV saved as {data_dir_save}.")

        # 전체 처리 완료된 경우 pkl 삭제
        if len(soap_list) == total_rows and os.path.exists(pkl_path):
            os.remove(pkl_path)
            print(f"Checkpoint file removed: {pkl_path}")