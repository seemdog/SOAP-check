# SOAP-check

## Usage

1. Prerequisite  
   set your openai, anthropic(optional) api key in `key.env`
2. SOAP note generation  
   for openai models: run `python soap.py`  
   for anthropic models: run `python soap-claude.py`
3. Evaluataion  
   (1) run `python unit.py --model llm-judge --file soap-note`  
   (2) run `python eval.py --model llm-judge --file soap-note`  
   (3) run `python score.py --file soap-note`  

## ETC.
- soap-note must be a `.csv` file
- soap-note must be under the `dataset` folder
- soap-note must have the following columns: `transcription`, `SOAP`
