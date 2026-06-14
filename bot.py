#!/usr/bin/env python3
"""Task Assistant Bot — Telegram → OpenAI (текстовый чат с контекстом)"""

import asyncio
import logging
import os
import sys
from collections import defaultdict

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from openai import AsyncOpenAI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
ALLOWED_IDS = [int(x.strip()) for x in os.environ.get("ALLOWED_IDS", "7653823001").split(",")]

MAX_HISTORY = 20
conversations: dict[int, list[dict[str, str]]] = defaultdict(list)

ai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

SYSTEM_PROMPT = (
    "Ты — ассистент, который превращает сырые описания задач в формализованные задачи для разработчика.\n\n"
    "Твоя задача — общаться с пользователем, уточнять детали, а когда задача сформулирована — выдавать её в формате.\n\n"
    "Формат готовой задачи (только когда пользователь подтвердил, что всё чётко):\n\n"
    "IT02-08:00/8/6-{Task name}\n{Описание задачи одной строкой}\n\n"
    "1. {Шаг 1}\n2. {Шаг 2}\n3. {Шаг 3}\n\n"
    "ВСЁ на русском языке. Без Markdown, только plain text.\n\n"
    "Если сообщение не про задачу — поддерживай диалог, но мягко возвращай к теме.\n"
    "Помни контекст: что уже обсудили, какие детали выяснили."
)


def trim_history(history: list[dict]) -> list[dict]:
    return history[-MAX_HISTORY:] if len(history) > MAX_HISTORY else history


@dp.message(Command("start"))
async def cmd_start(msg: types.Message):
    if msg.from_user.id not in ALLOWED_IDS:
        await msg.reply("⛔ Нет доступа")
        return
    conversations[msg.from_user.id] = []
    await msg.reply(
        "👋 Привет! Я помню контекст разговора.\n\n"
        "Просто опиши задачу — помогу её формализовать.\n"
        "/start — сбросить контекст\n"
        "/clear — очистить историю"
    )


@dp.message(Command("clear"))
async def cmd_clear(msg: types.Message):
    if msg.from_user.id not in ALLOWED_IDS:
        return
    conversations[msg.from_user.id] = []
    await msg.reply("🧹 Контекст очищен")


@dp.message()
async def handle_message(msg: types.Message):
    if msg.from_user.id not in ALLOWED_IDS:
        await msg.reply("⛔ Нет доступа")
        return

    if not msg.text or msg.text.startswith("/"):
        return

    user_id = msg.from_user.id
    user_input = msg.text.strip()

    await bot.send_chat_action(msg.chat.id, "typing")

    try:
        conversations[user_id].append({"role": "user", "content": user_input})

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(trim_history(conversations[user_id]))

        response = await ai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=800,
        )
        reply = response.choices[0].message.content.strip()
        conversations[user_id].append({"role": "assistant", "content": reply})
        await msg.reply(reply)

        logger.info("Контекст %s: %d сообщений", user_id, len(conversations[user_id]))

    except Exception as e:
        logger.exception("API error")
        await msg.reply(f"❌ Ошибка: {e}")


async def main():
    logger.info("Бот запущен, ALLOWED_IDS=%s", ALLOWED_IDS)
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Остановлен")
        sys.exit(0)
