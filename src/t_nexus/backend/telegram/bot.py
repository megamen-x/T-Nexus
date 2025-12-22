import os
import io
import asyncio
import traceback
import tempfile
import logging
from typing import Any, List, Optional, Dict

import httpx
import pandas as pd

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message, InlineKeyboardButton, CallbackQuery, FSInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder

from src.t_nexus.backend.telegram.tg_database import TelegramDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TNexusBot:

    DEFAULT_ANSWER_MODE = "full"
    SUPPORTED_MODES = {"full", "short"}
    CSV_REQUIRED_COLUMN = "question"

    GREETING_MESSAGE = "Hello! I'm T-Nexus Support Bot. How can I help you?"
    ABOUT_MESSAGE = "T-Nexus - an intelligent support assistant"
    PROCESSING_MESSAGE = "💬"
    GENERAL_ERROR_MESSAGE = "Something went wrong. Please try again later."
    UNSUPPORTED_TYPE_MESSAGE = "Error! This message type is not supported!"
    API_RESPONSE_ERROR_MESSAGE = "Unable to receive a valid response from the server."
    
    FEEDBACK_CALLBACK_PREFIX = "feedback:"
    LIKE_CALLBACK_DATA = f"{FEEDBACK_CALLBACK_PREFIX}like"
    DISLIKE_CALLBACK_DATA = f"{FEEDBACK_CALLBACK_PREFIX}dislike"
    ANSWER_MODE_CALLBACK_PREFIX = "ansmode:"
    FULL_MODE_CALLBACK = f"{ANSWER_MODE_CALLBACK_PREFIX}full"
    SHORT_MODE_CALLBACK = f"{ANSWER_MODE_CALLBACK_PREFIX}short"
    

    def __init__(self, bot_token: str, db_path: str, rag_url: str, transcription_url: str):
        logger.info("Initializing T-Nexus bot")
        self.db = TelegramDatabase(db_path)
        self.rag_url = rag_url
        self.transcription_url = transcription_url
        
        self.bot = Bot(token=bot_token, default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN))
        self.dp = Dispatcher()
        
        self._likes_kb = self._build_feedback_keyboard()
        self._answer_mode_kb = self._build_answer_mode_keyboard()
        self._register_handlers()

        self.user_answer_mod: Dict[int, str] = {}
        self.user_states: Dict[int, str] = {}
        logger.info("Bot initialized successfully")

    def _build_feedback_keyboard(self) -> InlineKeyboardBuilder:
        builder = InlineKeyboardBuilder()
        builder.add(InlineKeyboardButton(text="👍", callback_data=self.LIKE_CALLBACK_DATA))
        builder.add(InlineKeyboardButton(text="👎", callback_data=self.DISLIKE_CALLBACK_DATA))
        return builder
    
    def _build_answer_mode_keyboard(self) -> InlineKeyboardBuilder:
        builder = InlineKeyboardBuilder()
        builder.add(
            InlineKeyboardButton(text="📜 Full answer",  callback_data=self.FULL_MODE_CALLBACK),
            InlineKeyboardButton(text="✂️ Short answer", callback_data=self.SHORT_MODE_CALLBACK),
        )
        return builder

    def _register_handlers(self):
        self.dp.message.register(self.start_command, Command("start"))
        self.dp.message.register(self.help_command, Command("help"))
        self.dp.message.register(self.answer_mode_command, Command("answer_mode"))
        self.dp.message.register(self.indexing_command, Command("indexing"))
        
        self.dp.message.register(self._handle_text, F.text)
        self.dp.message.register(self._handle_voice, F.voice)
        self.dp.message.register(self._handle_document, F.document)
        
        self.dp.callback_query.register(self._handle_feedback_callback, F.data.startswith(self.FEEDBACK_CALLBACK_PREFIX))
        self.dp.callback_query.register(self._handle_answer_mode_callback, F.data.startswith(self.ANSWER_MODE_CALLBACK_PREFIX))

    async def start_polling(self):
        await self.dp.start_polling(self.bot)

    async def start_command(self, message: Message):
        self.db.create_conv_id(message.chat.id)
        await message.reply(self.GREETING_MESSAGE)

    async def help_command(self, message: Message):
        help_text = (
            "🤖 T-Nexus Support Bot\n\n"
            "Available commands:\n"
            "/start - Start a new conversation\n"
            "/help - Show this help message\n"
            "/answer_mode - Choose answer mode (full/short)\n"
            "/indexing - Upload documents for indexing\n\n"
            "You can also:\n"
            "• Send text messages for questions\n"
            "• Send voice messages\n"
            "• Upload CSV/Excel files for batch processing"
        )
        await message.reply(help_text, parse_mode=None)
    
    async def answer_mode_command(self, message: Message):
        markup = self._answer_mode_kb.as_markup()
        await message.answer("Choose answer mode:", reply_markup=markup)

    async def indexing_command(self, message: Message):
        user_id = message.from_user.id if message.from_user else None
        if not user_id:
            return
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(self.rag_url + 'get_database_name/')
                response.raise_for_status()
                data = response.json()
                db_name = data.get("message", "Unknown")
        except Exception as e:
            logger.error(f"Error getting database name: {e}")
            db_name = "Could not retrieve database name"
        
        self.user_states[user_id] = "waiting_for_zip"

        await message.reply(
            text=f"Current database: {db_name}\n\n"
            "Please send a ZIP file with documents for indexing.\n"
            "If the filename matches the current database name, it will be updated.\n"
            "Otherwise, a new database will be created.",
            parse_mode=None
        )
        
    def _get_user_full_name(self, message: Message) -> Optional[str]:
        if not message.from_user:
            return None
        return message.from_user.full_name or message.from_user.username

    async def _handle_text(self, message: Message):
        if not message.text:
            return
        await self._process_content(message, message.text)
    
    async def _handle_voice(self, message: Message):
        user_id = message.from_user.id if message.from_user else None

        placeholder = await message.answer(f"🎤 Recognizing voice message...")
        try:
            transcribed_text = await self._transcribe_voice(message)
            await placeholder.delete()
            await self._process_content(message, transcribed_text)
        except Exception as e:
            logger.error(f"Error processing voice message for user {user_id}: {e}")
            traceback.print_exc()
            await placeholder.edit_text(f"Could not recognize speech: {e}")

    async def _handle_document(self, message: Message):
        user_id = message.from_user.id if message.from_user else None

        if self.user_states.get(user_id) == "waiting_for_zip":
            await self._handle_zip_for_indexing(message)
            self.user_states.pop(user_id, None)
            return

        content = await self._parse_document(message)
        if content is None:
            await message.answer(self.UNSUPPORTED_TYPE_MESSAGE)
            return

        placeholder = await message.answer(f"Processing file...")
        try:
            logger.debug(f"Sending batch processing request to API for user {user_id}")
            full_ans, short_ans, docs = await self._query_api(content)
            df = pd.DataFrame({'Question': content, 'Short Answer': short_ans, 'Full Answer': full_ans, 'Documents': docs})
            
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.csv', encoding='utf-8') as tmp_file:
                df.to_csv(tmp_file.name, index=False)
                await self.bot.send_document(
                    document=FSInputFile(tmp_file.name, filename="results.csv"), 
                    chat_id=message.chat.id
                )
            os.remove(tmp_file.name)
            await placeholder.delete()

        except Exception as e:
            logger.error(f"Error during batch document processing for user {user_id}: {e}")
            traceback.print_exc()
            await placeholder.edit_text(self.GENERAL_ERROR_MESSAGE)

    async def _process_content(self, message: Message, content: Any):
        user_id = message.from_user.id
        user_name = self._get_user_full_name(message)
        conv_id = self.db.get_current_conv_id(user_id)

        self.db.save_user_message(content, conv_id=conv_id, user_id=user_id, user_name=user_name)
        placeholder = await message.answer(self.PROCESSING_MESSAGE)

        try:
            full_ans, short_ans, docs = await self._query_api([content])

            mode = self.user_answer_mod.get(user_id, self.DEFAULT_ANSWER_MODE)
            answer_text = full_ans[0] if mode == "full" else short_ans[0]
            docs_text = ", ".join(docs[0]) if docs and docs[0] else "No sources available"
            final_answer = (
                f"{answer_text}\n"
                f"===========================\n"
                f"Documents:\n{docs_text}"
            )

            markup = self._likes_kb.as_markup()
            new_message = await placeholder.edit_text(final_answer, parse_mode=None, reply_markup=markup)

            self.db.save_assistant_message(
                content=final_answer,
                conv_id=conv_id,
                message_id=new_message.message_id,
            )
        except httpx.RequestError as exc:
            logger.error(f"Network error during API request for user {user_id}: {exc}")
            await placeholder.edit_text(f"Network error. Cannot connect to server. {exc}")
        except Exception:
            logger.error(f"General error processing content for user {user_id}")
            traceback.print_exc()
            await placeholder.edit_text(self.GENERAL_ERROR_MESSAGE)

    async def _handle_feedback_callback(self, callback: CallbackQuery):
        if not callback.data:
            return
        user_id = callback.from_user.id
        message_id = callback.message.message_id
        feedback = callback.data.replace(self.FEEDBACK_CALLBACK_PREFIX, "")
        is_correct = 1 if feedback == 'dislike' else 0

        self.db.save_feedback(feedback, user_id=user_id, message_id=message_id, is_correct=is_correct)
        await self.bot.edit_message_reply_markup(
            chat_id=callback.message.chat.id,
            message_id=message_id,
            reply_markup=None
        )
        await callback.answer(f"Thank you for your feedback: {feedback}!")

    async def _handle_answer_mode_callback(self, callback: CallbackQuery):
        if not callback.data:
            return

        mode = callback.data.replace(self.ANSWER_MODE_CALLBACK_PREFIX, "")

        if mode not in self.SUPPORTED_MODES:
            await callback.answer("Unknown mode.")
            return

        user_id = callback.from_user.id
        self.user_answer_mod[user_id] = mode

        text = "Full" if mode == 'full' else 'Short'
        await callback.message.edit_text(
            text='Successfully changed mode to: ' + text,
            parse_mode=None,
            reply_markup=None
        )
        await callback.answer(f"✅ Answer mode set to «{mode}»")

    async def _query_api(self, user_content: List[str]) -> tuple:
        question = {'question': user_content}
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(self.rag_url + 'request_processing/', json=question)
            response.raise_for_status()
            data = response.json()
            
            for k in ("full_answer", "short_answer", "docs"):
                if k not in data:
                    raise KeyError(self.API_RESPONSE_ERROR_MESSAGE)

            return data["full_answer"], data["short_answer"], data["docs"]

    async def _transcribe_voice(self, message: Message) -> str:
        if not message.voice:
            return ""

        user_id = message.from_user.id if message.from_user else None

        try:
            voice_file = await self.bot.get_file(message.voice.file_id)

            voice_ogg_buffer = io.BytesIO()
            await self.bot.download_file(voice_file.file_path, destination=voice_ogg_buffer)
            voice_ogg_buffer.seek(0)

            files = {
                'file': ('voice.ogg', voice_ogg_buffer, 'audio/ogg')
            }

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(self.transcription_url, files=files)
                response.raise_for_status()
                data = response.json()
                return data.get("transcription", "")

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error during transcription for user {user_id}: {e.response.status_code} - {e.response.text}")
            return "An error occurred while processing your message."
        except httpx.RequestError as e:
            logger.error(f"Network error during transcription for user {user_id}: {e}")
            return "Transcription service is temporarily unavailable."
        except Exception as e:
            logger.error(f"Unknown error during transcription: {e}")
            return "Something went wrong."

    async def _handle_zip_for_indexing(self, message: Message):
        file_name = message.document.file_name
        if not file_name or not file_name.lower().endswith('.zip'):
            await message.reply("Please send a file with .zip extension")
            return
            
        placeholder = await message.reply("⏳ Sending file to server for indexing...")
        
        try:
            file_info = await self.bot.get_file(message.document.file_id)
            file_data = await self.bot.download_file(file_info.file_path)
            
            files = {'file': (file_name, file_data, 'application/zip')}
            
            async with httpx.AsyncClient(timeout=3600.0) as client:
                response = await client.post(self.rag_url + 'index/', files=files)
                response.raise_for_status()
                result = response.json()
                await placeholder.edit_text(f"✅ {result.get('message', 'Indexing completed successfully.')}")
                
        except httpx.HTTPStatusError as e:
            error_msg = f"Server error: {e.response.status_code}"
            try:
                error_detail = e.response.json().get('detail', '')
                if error_detail:
                    error_msg += f" - {error_detail}"
            except:
                pass
            logger.error(f"HTTP error during indexing: {error_msg}")
            await placeholder.edit_text(f"❌ {error_msg}")
        except httpx.RequestError as e:
            logger.error(f"Network error during indexing: {e}")
            await placeholder.edit_text(f"❌ Network error: {e}")
        except Exception as e:
            logger.error(f"Error during indexing: {e}")
            traceback.print_exc()
            await placeholder.edit_text(f"❌ Error: {e}")

    async def _parse_document(self, message: Message) -> Optional[List[str]]:
        if not message.document:
            return None

        user_id = message.from_user.id if message.from_user else None
        file_name = message.document.file_name or ""
        file_extension = file_name.split('.')[-1].lower()
        
        logger.debug(f"Parsing document for user {user_id}: {file_name} (extension: {file_extension})")

        with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as tmp_file:
            tmp_path = tmp_file.name
            
        try:
            file_info = await self.bot.get_file(message.document.file_id)
            await self.bot.download_file(file_info.file_path, tmp_path)
            
            if file_extension == "csv":
                df = pd.read_csv(tmp_path)
                if self.CSV_REQUIRED_COLUMN not in [el.lower() for el in df.columns]:
                    await message.reply(f"Error: CSV file is missing required column '{self.CSV_REQUIRED_COLUMN}'.")
                    return None
                return df[self.CSV_REQUIRED_COLUMN].dropna().astype(str).to_list()
            
            elif file_extension in ["xls", "xlsx"]:
                df = pd.read_excel(tmp_path)
                return df.iloc[:, 0].dropna().astype(str).to_list()

        except Exception as e:
            logger.error(f"Error parsing document for user {user_id}: {e}")
            await message.reply(f"Could not process file: {e}")
            return None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        
        return None


def create_bot(
    bot_token: str = None,
    db_path: str = None,
    rag_url: str = None,
    transcription_url: str = None
) -> TNexusBot:
    from src.t_nexus.backend.config import settings
    
    return TNexusBot(
        bot_token=bot_token or settings.BOT_TOKEN,
        db_path=db_path or settings.TG_DB_PATH,
        rag_url=rag_url or settings.RAG_URL,
        transcription_url=transcription_url or settings.TRANSCRIPTION_URL
    )