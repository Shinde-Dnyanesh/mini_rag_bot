# Mini RAG Telegram Bot
A lightweight RAG Telegram bot built with Python,SentenceTransformers, SQLite, and a local HuggingFace T5 model.
The bot retrieves the most relevant document chunks from the data folder and generates answers directly inside Telegram.

##  Project Structure
mini_rag_bot

- bot.py              - Telegram bot logic & handlers
- retriever.py        - Loads DB and retrieves best chunks
- llm_query.py        - Builds prompt + generates answer with HF model
- embed_build.py      - Builds embeddings and stores them into SQLite
- utils.py            - Helper functions (load docs, chunk text)
- requirements.txt    - Python dependencies
- data                - Knowledge base (.txt /.md files)
    about.txt
    policy.txt
    faqs.txt

## Architecture (RAG Pipeline)
data.txt
- utils.py - loads files & chunks text
- embed_build.py  -  memory.db
- retriever.py  -  finds best chunks for a query
- llm_query.py  -  generates answer using HF model
- bot.py  -  sends answer to Telegram user

##  Installation
Clone the repo:
- git clone https://github.com/Shinde-Dnyanesh/mini_rag_bot.git
- cd mini_rag_bot

Create a virtual environment:
- python3 -m venv venv
- source venv/bin/activate

Install dependencies:
- pip install --upgrade pip
- pip install -r requirements.txt

##  Build Embeddings 
This reads files in data, splits them into chunks, builds embeddings, and saves them into memory.db.
python embed_build.py

Expected output:
Saved N chunks to memory.db

## Test retriever:
python retriever.py

##  Run the Telegram Bot
- Set your bot token: export TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"

- Run the bot: python bot.py
- you should see = Bot running... open Telegram and send /start

## Usage (Telegram Commands)
- /start → Welcome message
- /help → Help menu
- /ask (question) → Ask about your documents
- Or simply type any message → bot will answer normally