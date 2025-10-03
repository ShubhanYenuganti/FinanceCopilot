## How to Run

# 1. Install dependencies
```bash
pip install -r requirements.txt
```

# 2. Preprocess data (optimize CSVs for querying metrics)
```bash
cd agent
python preprocess.py
cd ..
```

# 3. Set your OpenAI API key
# macOS/Linux
```bash
export OPENAI_API_KEY="your-api-key"
```
# Windows (PowerShell)
```bash
setx OPENAI_API_KEY "your-api-key"
```

# 4. Start the Streamlit app
```bash
streamlit run app.py
```
# 5 Running pytests
From the main directory of the project
```bash
export PYTHONPATH=.
pytest tests/test.py -v
```
