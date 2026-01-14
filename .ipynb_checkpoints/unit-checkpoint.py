import pandas as pd
import ast
from openai import OpenAI
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


# [수정 1] temperature와 seed 인자 추가 및 API 호출에 적용
def chat(system, user, content, model, temperature=0, seed=42): 

    user = user.replace("{content}", content)

    messages = [
        {
            "role": "system",
            "content": system
        },
        {
            "role": "user",
            "content": user
        }
    ]

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, # [추가] 0에 가까울수록 결정적(Deterministic) 결과
        seed=seed                # [추가] 고정된 시드값 사용 (Reproducibility)
    )
    return completion.choices[0].message.content


# [수정 2] worker 함수에서도 temperature와 seed를 받아서 chat으로 전달
def worker(i, text, system_unit, user_unit, model, temperature, seed):
    """한 row 처리용 워커 함수 (멀티스레드에서 사용)"""
    unit = chat(system_unit, user_unit, content=text, model=model, temperature=temperature, seed=seed)
    return i, unit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt-5.1')
    parser.add_argument('--file', type=str, default='evaluation_sample.csv')
    parser.add_argument('--max_workers', type=int, default=5)  # 병렬 처리 스레드 수
    
    # [수정 3] 커맨드라인 인자로 조절 가능하도록 추가 (기본값 설정: temp=0, seed=42)
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
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), timeout=50)

    # 데이터 로드
    data_dir = "./data/" + file
    data_dir_save = "./data/" + model + "_" + file
    df = pd.read_csv(data_dir)

    # 프롬프트 텍스트 로드
    system_unit = open_txt("./prompt/system_unit_2.txt")
    user_unit = open_txt("./prompt/user_unit.txt")

    # 결과 저장 디렉토리
    new_dir = "./data/pkl"
    os.makedirs(new_dir, exist_ok=True)

    pkl_path = f'{new_dir}/{model}_{file[:-4]}_unit.pkl'

    # 기존 결과 로드 (있으면 이어서 진행)
    try:
        with open(pkl_path, 'rb') as f:
            unit_list = pickle.load(f)
        print(f"Loaded existing data. Current length: {len(unit_list)}")
    except Exception:
        unit_list = []
        print("No existing data to load. Starting from scratch.")

    start_idx = len(unit_list)
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
        pending = {}              # 아직 unit_list에 안 붙인 결과들 (i -> unit)
        next_to_write = start_idx  # 다음에 unit_list에 붙여야 할 인덱스

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    worker,
                    i,
                    df["transcription"][i],
                    system_unit,
                    user_unit,
                    model,
                    temperature, # [수정 4] worker에 인자 전달
                    seed         # [수정 4] worker에 인자 전달
                ): i for i in indices
            }

            for future in tqdm(as_completed(futures), total=len(futures)):
                i, unit = future.result()
                pending[i] = unit

                # 가능한 한 인덱스 순서대로 unit_list에 append
                updated = False
                while next_to_write in pending:
                    unit_list.append(pending.pop(next_to_write))
                    next_to_write += 1
                    updated = True

                # 순서대로 새로 붙은 게 있을 때마다 체크포인트 저장
                if updated:
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(unit_list, f)

        # 최종적으로 CSV 저장
        if len(unit_list) != total_rows:
            print(f"Warning: unit_list length ({len(unit_list)}) != df length ({total_rows})")
        else:
            print("All rows processed successfully.")

        df["reference_unit"] = unit_list
        df.to_csv(data_dir_save, index=False) 
        print(f"Processing done and CSV saved as {data_dir_save}.")

        # 전체 처리 완료된 경우 pkl 삭제
        if len(unit_list) == total_rows and os.path.exists(pkl_path):
            os.remove(pkl_path)
            print(f"Checkpoint file removed: {pkl_path}")