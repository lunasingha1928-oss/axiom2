from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import settings

class Base(DeclarativeBase):
    pass

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=False,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    # Import all models to register them on Base.metadata
    import app.models.project
    import app.models.risk
    import app.models.photo
    import app.models.sensor
    import app.models.weather
    import app.models.satellite_image_cache

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
