import asyncio
import os
import threading
import uuid

import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, jsonify
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
    raise SystemExit(
        "Отсутствуют ключи. Скопируй .env.example в .env и заполни "
        "TELEGRAM_BOT_TOKEN и GEMINI_API_KEY."
    )

genai.configure(api_key=GEMINI_API_KEY)
MODEL = genai.GenerativeModel("gemini-1.5-flash")

MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_VOICE_SECONDS = 300
MAX_MESSAGE_LEN = 4000

FRIENDLY_ERROR = (
    "Ой, голосовое слишком длинное или не получилось разобрать. "
    "Попробуй записать короче."
)

PROMPT = (
    "Расшифруй голосовое сообщение в чистый текст на русском языке. "
    "Убери слова-паразиты («ну», «эээ», «типа», «как бы»), исправь явные "
    "оговорки, но сохрани смысл сказанного. Разбей текст на абзацы, чтобы "
    "его было легко читать. В конце выдели суть задачи или просьбы, если "
    "она есть. Верни только расшифрованный текст без комментариев."
)

app = Flask(__name__)


@app.route("/")
def health() -> str:
    return "Papa bot is alive"


@app.route("/health")
def health_check():
    return jsonify(status="ok")


def run_flask() -> None:
    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)


def transcribe_audio(audio_path: str, mime_type: str) -> str:
    uploaded_file = genai.upload_file(audio_path, mime_type=mime_type)
    response = MODEL.generate_content([uploaded_file, PROMPT])
    return response.text.strip()


def split_into_messages(text: str) -> list[str]:
    parts = []
    while len(text) > MAX_MESSAGE_LEN:
        cut = text.rfind("\n\n", 0, MAX_MESSAGE_LEN)
        if cut == -1:
            cut = text.rfind(" ", 0, MAX_MESSAGE_LEN)
        if cut == -1:
            cut = MAX_MESSAGE_LEN
        parts.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        parts.append(text)
    return parts


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message.voice is not None:
        media, ext = message.voice, "ogg"
    elif message.audio is not None:
        media, ext = message.audio, "audio"
    else:
        return

    if getattr(media, "duration", None) and media.duration > MAX_VOICE_SECONDS:
        await message.reply_text(FRIENDLY_ERROR)
        return

    if getattr(media, "file_size", None) and media.file_size > MAX_FILE_SIZE:
        await message.reply_text(FRIENDLY_ERROR)
        return

    mime_map = {
        "audio/mpeg": "mp3",
        "audio/mp4": "m4a",
        "audio/x-m4a": "m4a",
        "audio/ogg": "ogg",
        "audio/wav": "wav",
    }
    if ext == "audio" and media.mime_type:
        ext = mime_map.get(media.mime_type, "bin")

    ext_to_mime = {
        "ogg": "audio/ogg",
        "mp3": "audio/mpeg",
        "m4a": "audio/mp4",
        "wav": "audio/wav",
    }
    mime_type = ext_to_mime.get(ext, "audio/ogg")

    audio_path = f"/tmp/voice_{uuid.uuid4().hex}.{ext}"
    try:
        file = await context.bot.get_file(media.file_id)
        await file.download_to_drive(audio_path)
    except Exception as exc:
        print(f"[ошибка скачивания] {exc}")
        await message.reply_text(FRIENDLY_ERROR)
        return

    try:
        await message.reply_text("Распознаю голосовое сообщение…")
        text = await asyncio.to_thread(transcribe_audio, audio_path, mime_type)
    except Exception as exc:
        print(f"[ошибка распознавания] {exc}")
        await message.reply_text(FRIENDLY_ERROR)
        return
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    if not text:
        await message.reply_text(FRIENDLY_ERROR)
        return

    for part in split_into_messages(text):
        await message.reply_text(part)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Привет! Пришли мне голосовое сообщение — я расшифрую его в текст."
    )


def main() -> None:
    threading.Thread(target=run_flask, daemon=True).start()

    app_bot = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app_bot.add_handler(CommandHandler("start", start))
    app_bot.add_handler(
        MessageHandler(filters.VOICE | filters.AUDIO, handle_voice)
    )
    print("Бот запущен. Нажми Ctrl+C для остановки.")
    app_bot.run_polling()


if __name__ == "__main__":
    main()
