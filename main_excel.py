import os
import re
import duckdb
import pandas as pd
import asyncio
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

# LangChain Imports
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 1. LOAD CONFIGURATION
load_dotenv()
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
DINOIKI_API_KEY = os.getenv("DINOIKI_API_KEY")

# Path ke file excel (Gunakan environment variable agar fleksibel)
EXCEL_PATH = os.getenv("EXCEL_PATH", "data_kpi.xlsx")

# Whitelist User ID dari Railway
ALLOWED_USERS_RAW = os.getenv("ALLOWED_USERS", "")
ALLOWED_USERS = [int(i.strip()) for i in ALLOWED_USERS_RAW.split(",") if i.strip()]

# 2. SETUP AI ENGINE (DUCKDB PROMPT)
TELEGRAM_EXCEL_PROMPT = """You are a DuckDB expert and a helpful AI Assistant for Pak Adnan.
Given an input question, create a syntactically correct DuckDB SQL query to run.
HANYA BERIKAN QUERY SQL MURNI, TANPA MARKDOWN ATAU BACKTICK.

Informasi Tabel:
Nama tabel adalah 'kpi_data'. Kolomnya mengikuti header yang ada di file Excel.

ATURAN SQL DUCKDB:
- Gunakan NULLIF(pembagi, 0) untuk menghindari division by zero.
- DuckDB bisa langsung menggunakan ROUND(angka, 2).
- Pastikan query kompatibel dengan sintaks DuckDB.

ATURAN FORMAT JAWABAN:
- Berikan jawaban akhir dalam Bahasa Indonesia yang profesional.
- JAWABAN HARUS FULL NARASI (Gunakan poin • jika data lebih dari satu).
- JANGAN gunakan tabel HTML (<table>) atau [CHART].
- Gunakan *teks* untuk menebalkan angka atau poin penting.

Table structure: {table_info}
Question: {input}"""

def get_duckdb_chain():
    try:
        df = pd.read_excel(EXCEL_PATH)
        conn = duckdb.connect(database=':memory:')
        conn.register('kpi_data', df)
        db_duck = SQLDatabase.from_uri("duckdb:///:memory:")
        llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=DINOIKI_API_KEY,
            base_url="https://ai.dinoiki.com/v1",
            temperature=0
        )
        PROMPT = PromptTemplate(input_variables=["input", "table_info"], template=TELEGRAM_EXCEL_PROMPT)
        return SQLDatabaseChain.from_llm(llm, db_duck, prompt=PROMPT, verbose=True, return_direct=False)
    except Exception as e:
        raise Exception(f"Gagal memproses file Excel: {str(e)}")

def clean_response(text):
    text = re.sub(r'\[CHART\].*?\\[/CHART\]', '', text, flags=re.DOTALL)
    text = re.sub(r'<table.*?>.*?</table>', '', text, flags=re.DOTALL)
    text = text.replace("```sql", "").replace("```", "").strip()
    return text

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ALLOWED_USERS:
        return
    await update.message.reply_text(
        "👋 *Halo Pak Adnan! (Versi Excel-DuckDB)*\n\nBot terhubung langsung ke Excel Sharing Folder. Data dibaca ulang setiap kali Bapak bertanya.", 
        parse_mode='Markdown'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ALLOWED_USERS:
        return
    user_question = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    try:
        chain = get_duckdb_chain()
        response = chain.invoke({"query": user_question})
        raw_answer = response.get("result", response)
        final_answer = clean_response(raw_answer)
        await update.message.reply_text(final_answer, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"⚠️ Maaf Pak, ada kendala: `{str(e)}`", parse_mode='Markdown')

if __name__ == '__main__':
    if not TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN belum diset!")
    else:
        print("🚀 Bot Telegram (Excel Mode) sedang berjalan...")
        application = ApplicationBuilder().token(TOKEN).build()
        application.add_handler(CommandHandler('start', start))
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
        application.run_polling()
