import os
import re
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, filters

from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.runnables import RunnablePassthrough

# 1. LOAD CONFIGURATION
load_dotenv()
TOKEN           = os.getenv("TELEGRAM_BOT_TOKEN")
DATABASE_URL    = os.getenv("DATABASE_URL")
DINOIKI_API_KEY = os.getenv("DINOIKI_API_KEY")

ALLOWED_USERS_RAW = os.getenv("ALLOWED_USERS", "")
ALLOWED_USERS = [int(i.strip()) for i in ALLOWED_USERS_RAW.split(",") if i.strip()]

# 2. SETUP AI ENGINE
db = SQLDatabase.from_uri(DATABASE_URL, sample_rows_in_table_info=0)

llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=DINOIKI_API_KEY,
    base_url="https://ai.dinoiki.com/v1",
    temperature=0.7
)

# Memory per user: simpan max 10 pesan terakhir (5 pasang tanya-jawab)
MAX_HISTORY = 10
user_histories: dict[int, list] = {}

def get_history(user_id: int) -> list:
    return user_histories.get(user_id, [])

def add_to_history(user_id: int, question: str, answer: str):
    history = user_histories.get(user_id, [])
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))
    # Batasi ke MAX_HISTORY pesan terakhir
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
    user_histories[user_id] = history

def clear_history(user_id: int):
    user_histories.pop(user_id, None)

# System prompt
SYSTEM_PROMPT = """You are a PostgreSQL expert and a helpful AI Assistant for a refinery company.
Given an input question, create a syntactically correct PostgreSQL query to run, then answer based on the result.
HANYA BERIKAN QUERY SQL MURNI SAAT GENERATE SQL, TANPA MARKDOWN ATAU BACKTICK.

Kamu memiliki memori percakapan — gunakan konteks dari pesan sebelumnya jika relevan.
Misalnya jika user bertanya "berapa yang expired?" setelah bertanya tentang ATG, kamu tahu konteksnya adalah ATG.

DATABASE SCHEMA:
{table_info}

ATURAN QUERY SQL:
- Pilih tabel yang paling relevan berdasarkan nama tabel dan kolom yang tersedia.
- Jika tabel relevan kosong, jawab: "Data belum tersedia."
- Kolom RU antar tabel mungkin berbeda format, gunakan ILIKE '%RU II%' saat JOIN.
- Selalu gunakan NULLIF(kolom_penyebut, 0) untuk menghindari division by zero.
- Gunakan ROUND(nilai::numeric, 2) untuk pembulatan.
- JANGAN query SELECT * tanpa LIMIT. Selalu agregasi, filter, atau LIMIT 20.
- Untuk bad_actor_monitoring: kolom utama adalah ru, tag_number, status, problem, action_plan, progress, target_date.
- Untuk icu_monitoring: kolom utama adalah ru, icu_status (Medium/High/Critical/Low), tag_no, issue, mitigation, permanent_solution, progress, target_closed, report_date.
- Untuk program_kerja_atg: kolom utama adalah refinery_unit, type, atg_eksisting, program_2024, prokja, action_plan_category, target, month_update.
- Untuk paf: Plant Availability Factor — kolom type, ru, target_realisasi, value, plan_unplan, month.
- Untuk zero_clamp: kolom ru, area, unit, tag_no_ln, type_damage, type_perbaikan, status, tanggal_dipasang, tanggal_rencana_perbaikan.
- Untuk issue_paf: kolom type (Primary/Secondary Unit), ru, date, issue.
- Untuk power_stream: kolom refinery_unit, type_equipment, equipment, status_operation, desain, kapasitas_max, average_actual.
- Untuk jumlah_eqp_utl: kolom refinery_unit, type_equipment, status_equipment, jumlah.
- Untuk critical_eqp_utl: kolom refinery_unit, type_equipment, highlight_issue, corrective_action, mitigasi_action, target_corrective.
- Untuk critical_eqp_prim_sec: kolom refinery_unit, unit_proses, equipment, highlight_issue, corrective_action, mitigasi_action.
- Untuk monitoring_operasi: kolom refinery_unit, unit_proses, unit, design, minimal_capacity, plant_readiness, actual, target_sts.
- Untuk inspection_plan: kolom refinery_unit, area, tag_no_ln, type_equipment, type_inspection, due_date, plan_date, actual_date, result_remaining_life, grand_result.

ATURAN FORMAT JAWABAN (KHUSUS TELEGRAM):
1. JAWABAN FULL NARASI — JANGAN gunakan tabel HTML, JANGAN format [CHART].
2. Gunakan poin-poin (•) jika data lebih dari satu.
3. Tebalkan poin penting dengan *teks*.
4. Tambahkan emoticon relevan (🏭, 💰, 📊, ✅, ⚠️, 🔧, 🛢️, 🚨).
5. Maksimal 5 poin per jawaban agar tidak terlalu panjang di layar HP.
6. Jika hasil lebih dari 10 item, tampilkan ringkasan/highlight saja dan sarankan user untuk mempersempit."""

OUT_OF_SCOPE = [
    "cuaca", "berita", "news", "coding", "resep", "masak", "film", "musik",
    "olahraga", "politik", "saham", "crypto", "bitcoin", "translate", "terjemahkan",
    "siapa presiden", "capital of", "ibukota",
]
DUMP_KEYWORDS = [
    "tampilkan semua", "lihat semua", "show all", "list semua", "dump",
    "seluruh isi", "semua baris", "semua data", "semua isi", "semua record",
    "export semua", "ceritakan semua", "semua action plan", "semua progress",
    "semua issue", "semua mitigasi", "semua prokja",
]

def clean_response(text):
    text = re.sub(r'\[CHART\].*?\[/CHART\]', '', text, flags=re.DOTALL)
    text = re.sub(r'\[DOWNLOAD:\w+\]', '', text)
    text = re.sub(r'<table.*?>.*?</table>', '', text, flags=re.DOTALL)
    text = text.replace("```sql", "").replace("```", "").strip()
    return text

async def run_query_with_memory(user_id: int, question: str) -> str:
    """Jalankan query AI dengan konteks history percakapan."""
    history = get_history(user_id)
    table_info = db.get_table_info()

    # Build prompt dengan history
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(table_info=table_info)}]
    for msg in history:
        if isinstance(msg, HumanMessage):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            messages.append({"role": "assistant", "content": msg.content})
    messages.append({"role": "user", "content": question})

    # Step 1: Generate SQL
    sql_response = llm.invoke(messages + [
        {"role": "user", "content": f"Berikan HANYA query SQL PostgreSQL untuk menjawab: {question}. Tanpa penjelasan."}
    ])
    sql_query = sql_response.content.replace("```sql", "").replace("```", "").strip()

    # Step 2: Execute SQL
    try:
        db_result = db.run(sql_query)
    except Exception as e:
        db_result = f"Error query: {str(e)}"

    # Step 3: Generate answer dengan hasil query + history
    messages.append({
        "role": "user",
        "content": f"Hasil query SQL:\n{db_result}\n\nBerikan jawaban dalam Bahasa Indonesia sesuai aturan format."
    })
    final_response = llm.invoke(messages)
    return final_response.content

# 3. TELEGRAM HANDLERS
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ALLOWED_USERS:
        await update.message.reply_text("⛔ *Akses Ditolak*\nMaaf, Anda tidak diizinkan menggunakan bot ini.", parse_mode='Markdown')
        return
    clear_history(user_id)
    await update.message.reply_text(
        "👋 *Halo!*\n\nBot siap digunakan dengan memori percakapan.\n"
        "Saya akan mengingat konteks pertanyaan sebelumnya dalam sesi ini.\n\n"
        "💡 *Tips:* Tanyakan analisis, perbandingan, atau insight — bukan tampilkan semua data.\n\n"
        "Ketik /reset untuk memulai percakapan baru.",
        parse_mode='Markdown'
    )

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ALLOWED_USERS:
        return
    clear_history(user_id)
    await update.message.reply_text("🔄 *Percakapan direset.* Memori sesi sebelumnya dihapus.", parse_mode='Markdown')

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ALLOWED_USERS:
        return

    user_question = update.message.text
    q_lower = user_question.lower()
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    # Pre-filter: out of scope
    if any(k in q_lower for k in OUT_OF_SCOPE):
        await update.message.reply_text(
            "⚠️ Maaf, saya hanya dapat membantu *analisis data maintenance kilang*.\n"
            "Silakan ajukan pertanyaan yang berkaitan dengan data yang tersedia.",
            parse_mode='Markdown'
        )
        return

    # Pre-filter: dump request
    if any(k in q_lower for k in DUMP_KEYWORDS):
        await update.message.reply_text(
            "📊 Menampilkan semua data dalam chat kurang efisien.\n\n"
            "Coba persempit pertanyaan, misalnya:\n"
            "• Berapa jumlah per RU?\n"
            "• Mana yang statusnya bermasalah?\n"
            "• Mana yang sudah melewati target date?\n\n"
            "💡 Untuk data lengkap, minta admin download via web.",
            parse_mode='Markdown'
        )
        return

    try:
        raw_answer = await run_query_with_memory(user_id, user_question)
        final_answer = clean_response(raw_answer)
        # Simpan ke history
        add_to_history(user_id, user_question, final_answer)
        await update.message.reply_text(final_answer, parse_mode='Markdown')
    except Exception as e:
        err = str(e)
        if "context_length_exceeded" in err:
            await update.message.reply_text(
                "⚠️ *Pertanyaan terlalu luas!*\n\n"
                "Coba persempit:\n"
                "• Sebutkan RU tertentu (misal: *RU II*)\n"
                "• Minta ringkasan, bukan daftar lengkap\n"
                "• Tambahkan filter waktu atau status",
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text(f"⚠️ Kendala teknis: `{err}`", parse_mode='Markdown')

# 4. RUN
if __name__ == '__main__':
    if not TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN belum diset!")
    else:
        print("🚀 Bot Telegram dengan memory percakapan berjalan...")
        application = ApplicationBuilder().token(TOKEN).build()
        application.add_handler(CommandHandler('start', start))
        application.add_handler(CommandHandler('reset', reset))
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
        application.run_polling()