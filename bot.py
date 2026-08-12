import asyncio
import os
import threading
import time
import uuid

import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, jsonify, request
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

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
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


def transcribe_audio(audio_path: str, mime_type: str) -> str:
    file_ref = genai.upload_file(audio_path, mime_type=mime_type)
    while file_ref.state.name == "PROCESSING":
        time.sleep(1)
        file_ref = genai.get_file(file_ref.name)
    if file_ref.state.name == "FAILED":
        raise RuntimeError("Gemini не смог обработать файл")
    response = MODEL.generate_content([file_ref, PROMPT])
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


app_bot = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
app_bot.add_handler(CommandHandler("start", start))
app_bot.add_handler(
    MessageHandler(filters.VOICE | filters.AUDIO, handle_voice)
)


@app.route("/webhook", methods=["POST"])
def webhook() -> tuple[str, int]:
    data = request.get_json(silent=True)
    if data is None:
        return "Invalid JSON", 400

    loop = event_loop
    if loop is None:
        return "Bot not ready", 503

    update = Update.de_json(data, app_bot.bot)
    future = asyncio.run_coroutine_threadsafe(
        app_bot.process_update(update), loop
    )

    def _log_update_error(fut: asyncio.Future) -> None:
        if not fut.cancelled() and fut.exception():
            print(f"[ошибка обработки update] {fut.exception()}")

    future.add_done_callback(_log_update_error)
    return "OK", 200


event_loop: asyncio.AbstractEventLoop | None = None


def main() -> None:
    global event_loop
    event_loop = asyncio.new_event_loop()
    threading.Thread(target=event_loop.run_forever, daemon=True).start()

    asyncio.run_coroutine_threadsafe(app_bot.initialize(), event_loop).result()
    asyncio.run_coroutine_threadsafe(app_bot.start(), event_loop).result()

    webhook_base = os.getenv("WEBHOOK_URL")
    if webhook_base:
        webhook_url = webhook_base.rstrip("/") + "/webhook"
        asyncio.run_coroutine_threadsafe(
            app_bot.bot.set_webhook(webhook_url), event_loop
        ).result()
        print(f"Вебхук установлен: {webhook_url}")
    else:
        print("WEBHOOK_URL не задан — вебхук не установлен")

    port = int(os.getenv("PORT", "10000"))
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
