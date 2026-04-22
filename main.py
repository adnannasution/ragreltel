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
DINOIKI_API_KEY = os.getenv("DINOIKI_API_KEY")

# Mengambil daftar ID dari Railway Variables (dipisahkan koma)
# Contoh isi di Railway: 12345678,87654321
ALLOWED_USERS_RAW = os.getenv("ALLOWED_USERS", "")
ALLOWED_USERS = [int(i.strip()) for i in ALLOWED_USERS_RAW.split(",") if i.strip()]

# 2. SETUP AI ENGINE
TELEGRAM_CUSTOM_PROMPT = """You are a PostgreSQL expert and a helpful AI Assistant for a refinery company.
Given an input question, create a syntactically correct PostgreSQL query to run.
HANYA BERIKAN QUERY SQL MURNI, TANPA MARKDOWN ATAU BACKTICK.

Setelah mendapatkan hasil dari database, berikan jawaban akhir dalam Bahasa Indonesia yang profesional.

STRUKTUR TABEL TERSEDIA:
{table_info}

ATURAN QUERY SQL:
- Pilih tabel yang paling relevan berdasarkan nama tabel dan kolom yang tersedia.
- Jika tabel relevan kosong, jawab: "Data belum tersedia, silakan upload datanya terlebih dahulu."
- Jangan query tabel yang tidak relevan dengan pertanyaan.
- Kolom RU antar tabel mungkin berbeda format, gunakan ILIKE '%RU II%' saat JOIN.
- Selalu gunakan NULLIF(kolom_penyebut, 0) untuk menghindari division by zero.
- Gunakan ROUND(nilai::numeric, 2) untuk pembulatan.
- Jika pertanyaan melibatkan lebih dari satu tabel, gunakan JOIN yang sesuai.
- Untuk bad_actor_monitoring: kolom utama adalah ru, tag_number, status, problem, action_plan, progress, target_date.
- Untuk icu_monitoring: kolom utama adalah ru, icu_status (Medium/High/Critical/Low), tag_no, issue, mitigation, permanent_solution, progress, target_closed, report_date.
- Untuk program_kerja_atg: kolom utama adalah refinery_unit, type, atg_eksisting, program_2024, prokja (progress), action_plan_category, target, month_update.
- Untuk paf: Plant Availability Factor — kolom type, ru, target_realisasi, value (angka PAF), plan_unplan, month.
- Untuk zero_clamp: monitoring temporary repair zero clamp — kolom ru, area, unit, tag_no_ln, type_damage, type_perbaikan, status, tanggal_dipasang, tanggal_rencana_perbaikan.
- Untuk issue_paf: daftar issue yang mempengaruhi PAF — kolom type (Primary/Secondary Unit), ru, date, issue.
- Untuk power_stream: status operasi equipment power & steam — kolom refinery_unit, type_equipment, equipment, status_operation, desain, kapasitas_max, average_actual.
- Untuk jumlah_eqp_utl: jumlah equipment utility per status — kolom refinery_unit, type_equipment, status_equipment, jumlah.
- Untuk critical_eqp_utl: critical equipment utility — kolom refinery_unit, type_equipment, highlight_issue, corrective_action, mitigasi_action, target_corrective.
- Untuk critical_eqp_prim_sec: critical equipment primary & secondary — kolom refinery_unit, unit_proses, equipment, highlight_issue, corrective_action, mitigasi_action.
- Untuk monitoring_operasi: monitoring kapasitas operasi unit proses — kolom refinery_unit, unit_proses, unit, design, minimal_capacity, plant_readiness, actual, target_sts.
- Untuk inspection_plan: rencana & realisasi inspeksi equipment — kolom refinery_unit, area, tag_no_ln, type_equipment, type_inspection, due_date, plan_date, actual_date, result_remaining_life, grand_result.

ATURAN FORMAT JAWABAN (KHUSUS TELEGRAM — NARASI SAJA):
1. JAWABAN HARUS FULL NARASI — JANGAN gunakan tabel HTML, JANGAN gunakan format [CHART].
2. Gunakan poin-poin (•) jika data lebih dari satu agar tetap rapi di layar HP.
3. Tebalkan poin penting dengan *teks*.
4. Tambahkan emoticon relevan (🏭, 💰, 📊, ✅, ⚠️, 📈, 📉, 🔧, 🛢️, 🚨, 🔴).
5. Maksimal 3-4 kalimat per poin agar tidak terlalu panjang di layar HP.

Question: {input}"""

db = SQLDatabase.from_uri(DATABASE_URL)
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=DINOIKI_API_KEY,
    base_url="https://ai.dinoiki.com/v1",
    temperature=0.7
)

PROMPT = PromptTemplate(input_variables=["input", "table_info"], template=TELEGRAM_CUSTOM_PROMPT)
db_chain = SQLDatabaseChain.from_llm(llm, db, prompt=PROMPT, verbose=True, return_direct=False)

def clean_response(text):
    text = re.sub(r'\[CHART\].*?\[/CHART\]', '', text, flags=re.DOTALL)
    text = re.sub(r'<table.*?>.*?</table>', '', text, flags=re.DOTALL)
    text = text.replace("```sql", "").replace("```", "").strip()
    return text

# 3. TELEGRAM BOT HANDLERS DENGAN WHITELIST
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ALLOWED_USERS:
        print(f"Akses ditolak untuk ID: {user_id}")
        await update.message.reply_text("⛔ *Akses Ditolak*\nMaaf Pak, Anda tidak diizinkan menggunakan bot ini.")
        return

    await update.message.reply_text(
        "👋 *Halo Pak Adnan!*\n\nBot sudah aman dan hanya bisa diakses oleh tim terdaftar. Silakan tanya data KPI.",
        parse_mode='Markdown'
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    # PROTEKSI: Cek apakah user ada di whitelist
    if user_id not in ALLOWED_USERS:
        return # Abaikan jika bukan user terdaftar

    user_question = update.message.text
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    
    try:
        response = db_chain.invoke({"query": user_question})
        raw_answer = response.get("result", response)
        final_answer = clean_response(raw_answer)
        await update.message.reply_text(final_answer, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"⚠️ Kendala teknis: `{str(e)}`", parse_mode='Markdown')

# 4. RUN THE BOT
if __name__ == '__main__':
    if not TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN belum diset!")
    else:
        print("🚀 Bot Telegram dengan Whitelist sedang berjalan...")
        application = ApplicationBuilder().token(TOKEN).build()
        application.add_handler(CommandHandler('start', start))
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
        application.run_polling()