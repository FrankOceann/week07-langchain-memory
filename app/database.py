import os

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


def get_mysql_url(mysql_url: str | None = None) -> str:
    load_dotenv()

    resolved_url = (
        mysql_url
        if mysql_url is not None
        else os.getenv("MYSQL_URL", "")
    )

    if not resolved_url:
        raise ValueError("缺少 MYSQL_URL，无法连接长期记忆数据库。")

    return resolved_url


def build_engine(mysql_url: str | None = None):
    return create_engine(
        get_mysql_url(mysql_url),
        pool_pre_ping=True,
    )


def build_session_factory(mysql_url: str | None = None):
    return sessionmaker(
        bind=build_engine(mysql_url),
        expire_on_commit=False,
    )