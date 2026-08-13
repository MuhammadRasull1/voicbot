import asyncio
import logging
import os
import tempfile
import threading

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()
logging.basicConfig(level=logging.INFO)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
    raise SystemExit(
        "Отсутствуют ключи. Скопируй .env.example в .env и заполни "
        "TELEGRAM_BOT_TOKEN и GEMINI_API_KEY."
    )

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=60000),
)
MODEL = "gemini-2.5-flash"

logger = logging.getLogger(__name__)

gemini_semaphore = asyncio.Semaphore(2)

MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_VOICE_SECONDS = 300

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


def _should_retry(exception):
    if isinstance(exception, ServerError):
        return True
    if isinstance(exception, ClientError):
        if exception.code in [400, 401, 403]:
            return False
        if exception.code in [429, 408, 503] or exception.code >= 500:
            return True
    return False


AUDIO_MIME_TO_SUFFIX = {
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
}


async def _process_voice(
    message, bot: Bot, file_id: str, suffix: str = ".ogg"
) -> None:
    audio_path = None
    uploaded_name = None

    async with gemini_semaphore:
        try:
            file_obj = await bot.get_file(file_id)

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                audio_path = tmp.name

            await file_obj.download_to_drive(custom_path=audio_path)

            await message.reply_text("🔄 Обрабатываю голосовое сообщение, подождите...")

            def run_gemini_with_retry():
                @retry(
                    stop=stop_after_attempt(4),
                    wait=wait_exponential(multiplier=2, min=2, max=15),
                    retry=retry_if_exception_type((ServerError, ClientError)),
                    reraise=True,
                )
                def execute():
                    nonlocal uploaded_name
                    uploaded = client.files.upload(file=audio_path)
                    uploaded_name = uploaded.name

                    response = client.models.generate_content(
                        model=MODEL,
                        contents=[PROMPT, uploaded],
                    )
                    return response.text

                return execute()

            response_text = await asyncio.to_thread(run_gemini_with_retry)

            if response_text:
                await send_long_message(message, response_text)
            else:
                await message.reply_text(FRIENDLY_ERROR)

        except Exception as e:
            logger.error(
                "Критическая ошибка при обработке голосового сообщения: %s", e,
                exc_info=True,
            )
            await message.reply_text(
                "❌ Извините, не удалось обработать голосовое сообщение из-за "
                "временной перегрузки серверов. Попробуйте позже."
            )

        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except Exception as ex:
                    logger.warning("Не удалось удалить локальный файл: %s", ex)

            if uploaded_name:
                try:
                    await asyncio.to_thread(client.files.delete, name=uploaded_name)
                except Exception as ex:
                    logger.warning("Не удалось удалить файл из Gemini API: %s", ex)


async def send_long_message(message, text: str) -> None:
    max_length = 4096
    for i in range(0, len(text), max_length):
        chunk = text[i : i + max_length]
        await message.reply_text(chunk)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.effective_message
    if message.voice is not None:
        media, suffix = message.voice, ".ogg"
    elif message.audio is not None:
        media = message.audio
        suffix = AUDIO_MIME_TO_SUFFIX.get(media.mime_type, ".audio")
    else:
        return

    if getattr(media, "duration", None) and media.duration > MAX_VOICE_SECONDS:
        await message.reply_text(FRIENDLY_ERROR)
        return

    if getattr(media, "file_size", None) and media.file_size > MAX_FILE_SIZE:
        await message.reply_text(FRIENDLY_ERROR)
        return

    asyncio.create_task(
        _process_voice(
            message=message,
            bot=context.bot,
            file_id=media.file_id,
            suffix=suffix,
        )
    )


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
