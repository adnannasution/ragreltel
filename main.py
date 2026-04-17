import os
import re
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
DATABASE_URL = os.getenv("DATABASE_URL")
# Menggunakan DINOIKI_API_KEY sesuai dengan dashboard Web Bapak
DINOIKI_API_KEY = os.getenv("DINOIKI_API_KEY")

TELEGRAM_CUSTOM_PROMPT = """You are a PostgreSQL expert and a helpful AI Assistant for Pak Adnan.
Given an input question, create a syntactically correct PostgreSQL query to run.
HANYA BERIKAN QUERY SQL MURNI, TANPA MARKDOWN ATAU BACKTICK.

Setelah mendapatkan hasil dari database, berikan jawaban akhir dalam Bahasa Indonesia yang profesional.

ATURAN SQL (SAMA DENGAN WEB):
- Selalu gunakan NULLIF(pembagi, 0) pada posisi penyebut untuk menghindari division by zero.
- WAJIB gunakan casting ::numeric untuk fungsi ROUND. 
  Contoh: ROUND((hasil_perhitungan)::numeric, 2)
- Pastikan query kompatibel dengan PostgreSQL.

ATURAN FORMAT JAWABAN (KHUSUS TELEGRAM):
1. JAWABAN HARUS FULL NARASI: JANGAN gunakan tabel HTML (<table>), JANGAN gunakan format [CHART].
2. Gunakan poin-poin (•) jika data lebih dari satu agar tetap rapi di layar HP.
3. Menebalkan poin penting dengan *teks*.
4. Tambahkan emoticon yang relevan (📊, ✅, ⚠️) agar interaktif.

Table structure: {table_info}
Question: {input}"""

# Inisialisasi Database
db = SQLDatabase.from_uri(DATABASE_URL)

# Inisialisasi LLM via Dinoiki (Disamakan dengan Dashboard Web)
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=DINOIKI_API_KEY,
    base_url="https://ai.dinoiki.com/v1",
    temperature=0.7
)

PROMPT = PromptTemplate(
    input_variables=["input", "table_info"], 
    template=TELEGRAM_CUSTOM_PROMPT
)

# Chain utama
db_chain = SQLDatabaseChain.from_llm(
    llm, 
    db, 
    prompt=PROMPT, 
    verbose=True,
    return_direct=False
)

# 3. FUNGSI MEMBERSIHKAN JAWABAN
def clean_response(text):
    """Pembersihan ekstra untuk Telegram"""
    # Hapus tag HTML sisa jika AI tidak sengaja mengirim tabel/chart
    text = re.sub(r'\[CHART\].*?\[/CHART\]', '', text, flags=re.DOTALL)
    text = re.sub(r'<table.*?>.*?</table>', '', text, flags=re.DOTALL)
    # Bersihkan sisa-sisa backtick SQL
    text = text.replace("```sql", "").replace("```", "").strip()
    return text

# 4. TELEGRAM BOT HANDLERS
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 *Halo Pak Adnan!*\n\nSaya Bot Khusus Telegram untuk monitoring KPI Maintenance.\nSilakan tanya apa saja, saya akan memberikan ringkasan narasinya.",
        parse_mode='Markdown'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_question = update.message.text
    
    # Animasi 'typing...'
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        # Panggil AI
        response = db_chain.invoke({"query": user_question})
        raw_answer = response.get("result", response)
        
        # Bersihkan jawaban agar rapi di HP
        final_answer = clean_response(raw_answer)
        
        await update.message.reply_text(final_answer, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"⚠️ Maaf Pak, ada kendala teknis: `{str(e)}`", parse_mode='Markdown')

# 5. RUN THE BOT
if __name__ == '__main__':
    if not TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN belum diset di environment variables!")
    else:
        print("🚀 Bot Telegram Khusus KPI sedang berjalan...")
        application = ApplicationBuilder().token(TOKEN).build()
        
        application.add_handler(CommandHandler('start', start))
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
        
        application.run_polling()