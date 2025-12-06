import secrets
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import List, Dict, Any, Generator, Optional

from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, select
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session

class TGBase(DeclarativeBase):
    pass

class TGMessage(TGBase):
    __tablename__ = 'tg_messages'

    id = Column(Integer, primary_key=True)
    role = Column(String(20), nullable=False)
    user_id = Column(Integer, nullable=True)
    user_name = Column(String(255), nullable=True)
    content = Column(Text, nullable=False)
    conv_id = Column(String(32), nullable=False, index=True)
    timestamp = Column(Integer, nullable=False)
    message_id = Column(Integer, nullable=True)

class TGConversation(TGBase):
    __tablename__ = 'tg_conversations'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    conv_id = Column(String(32), nullable=False, unique=True)
    timestamp = Column(Integer, nullable=False)

class TGLike(TGBase):
    __tablename__ = 'tg_likes'

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False, index=True)
    message_id = Column(Integer, nullable=False, index=True)
    feedback = Column(String(255), nullable=False)
    is_correct = Column(Boolean, nullable=False, default=True)


class TelegramDatabase:

    def __init__(self, db_path: str):
        self.engine = create_engine(f'sqlite:///{db_path}')
        TGBase.metadata.create_all(self.engine)
        self._session_factory = sessionmaker(bind=self.engine)

    @contextmanager
    def _session_scope(self) -> Generator[Session, None, None]:
        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @staticmethod
    def _get_current_ts() -> int:
        return int(datetime.now(timezone.utc).timestamp())

    def create_conv_id(self, user_id: int) -> str:
        conv_id = secrets.token_hex(nbytes=16)
        with self._session_scope() as session:
            new_conv = TGConversation(
                user_id=user_id, 
                conv_id=conv_id, 
                timestamp=self._get_current_ts()
            )
            session.add(new_conv)
        return conv_id

    def get_current_conv_id(self, user_id: int) -> str:
        with self._session_scope() as session:
            latest_conv = session.query(TGConversation.conv_id)\
                                 .filter(TGConversation.user_id == user_id)\
                                 .order_by(TGConversation.timestamp.desc())\
                                 .first()
            if latest_conv is None:
                return self.create_conv_id(user_id)
            return latest_conv[0]

    def _format_message_for_output(self, msg: TGMessage, include_meta: bool) -> Dict[str, Any]:
        message_data = {
            "role": msg.role,
            "text": self._parse_content(msg.content)
        }
        if include_meta:
            message_data["timestamp"] = msg.timestamp
        return message_data

    def fetch_conversation(self, conv_id: str, include_meta: bool = False) -> List[Dict[str, Any]]:
        with self._session_scope() as session:
            messages = session.query(TGMessage)\
                              .filter(TGMessage.conv_id == conv_id)\
                              .order_by(TGMessage.timestamp)\
                              .all()
            
            return [self._format_message_for_output(m, include_meta) for m in messages]

    def save_user_message(self, content: Any, conv_id: str, user_id: int, user_name: Optional[str] = None) -> None:
        with self._session_scope() as session:
            new_message = TGMessage(
                role="user",
                content=self._serialize_content(content),
                conv_id=conv_id,
                user_id=user_id,
                user_name=user_name,
                timestamp=self._get_current_ts()
            )
            session.add(new_message)

    def save_assistant_message(self, content: str, conv_id: str, message_id: int) -> None:
        with self._session_scope() as session:
            new_message = TGMessage(
                role="assistant",
                content=content,
                conv_id=conv_id,
                timestamp=self._get_current_ts(),
                message_id=message_id,
            )
            session.add(new_message)

    def save_feedback(self, feedback: str, user_id: int, message_id: int, is_correct: bool = True) -> None:
        with self._session_scope() as session:
            new_feedback = TGLike(
                user_id=user_id,
                message_id=message_id,
                feedback=feedback,
                is_correct=is_correct
            )
            session.add(new_feedback)

    def get_all_conv_ids(self) -> List[str]:
        with self._session_scope() as session:
            stmt = select(TGConversation.conv_id)
            return list(session.scalars(stmt).all())

    def get_feedback_stats(self) -> Dict[str, int]:
        with self._session_scope() as session:
            likes = session.query(TGLike).filter(TGLike.feedback == "like").count()
            dislikes = session.query(TGLike).filter(TGLike.feedback == "dislike").count()
            return {"likes": likes, "dislikes": dislikes}

    def get_total_conversations(self) -> int:
        with self._session_scope() as session:
            return session.query(TGConversation).count()

    def get_total_messages(self) -> int:
        with self._session_scope() as session:
            return session.query(TGMessage).count()

    def _serialize_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content)

    def _parse_content(self, content: str) -> Any:
        try:
            parsed_content = json.loads(content)
            if isinstance(parsed_content, list) and all(isinstance(m, dict) for m in parsed_content):
                return parsed_content
            return content
        except json.JSONDecodeError:
            return content