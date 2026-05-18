# Liquid Handler Assistant

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about liquid handler manuals using AI. Built with Streamlit, ChromaDB, OpenAI, and Gemini.

## Manuals Included

- Hamilton Microlab STAR V with VENUS 6.3
- Agilent Bravo Platform
- Beckman Coulter Biomek i-Series
- Hamilton VENUS Four

## Features

- Ask questions in plain English about any of the liquid handler manuals
- Answers are grounded in the manual content with page number citations
- Switch between OpenAI (gpt-4o) and Gemini (gemini-1.5-flash) in the sidebar
- Multi-turn conversation memory

## Requirements

- Python 3.10+
- OpenAI API key and/or Gemini API key

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/billiontoone/se_chatbot.git
cd se_chatbot
```

**2. Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add API keys**
```bash
cp .env.example .env
nano .env
```

Fill in your API keys:
```
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=AIza-...
```

- OpenAI API key: https://platform.openai.com/api-keys
- Gemini API key: https://aistudio.google.com/app/apikey

**5. Ingest the manuals (one-time)**
```bash
# OpenAI embeddings
python ingest.py --provider openai

# Gemini embeddings
python ingest.py --provider gemini

# Both
python ingest.py
```

**6. Run the app**
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

## Usage

- Select your AI provider in the sidebar (OpenAI or Gemini)
- Type your question in the chat box
- The assistant will search the manuals and answer with page number citations
- If the answer is not found in the manuals, the assistant will say so

## Adding New Manuals

1. Drop the PDF into the `data/` folder
2. Re-run `python ingest.py --provider openai` (or `gemini`)
3. Restart the app

## Project Structure

```
se_chatbot/
├── app.py              # Streamlit UI
├── ingest.py           # PDF → ChromaDB ingestion pipeline
├── rag.py              # Retrieval + AI response
├── data/               # PDF manuals
├── vectorstore/        # ChromaDB embeddings (auto-generated)
├── requirements.txt
├── .env.example
└── .gitignore
```
