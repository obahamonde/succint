import os
import re
from functools import cached_property
from io import BytesIO
from tempfile import TemporaryDirectory
from typing import AsyncGenerator, Optional
from uuid import uuid4

import torch
import whisper  # type: ignore
from httpx import AsyncClient
from pydantic import BaseModel, Field, computed_field
from pytube import YouTube  # type: ignore

video_id_pattern = re.compile(r"v=([a-zA-Z0-9_-]{11})")


class Segment(BaseModel):
    start: float
    end: float
    text: str


class Transcript(BaseModel):
    text: str
    segments: list[Segment]
    language: str = Field(...)


class AudioTranscription:
    @cached_property
    @torch.no_grad()  # type: ignore
    def audio(self):
        return whisper.load_model(name="small", download_root="./audio").to("cuda")

    @cached_property
    @torch.no_grad()  # type: ignore
    def device(self):
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    async def transcribe(self, audio: AsyncGenerator[bytes, None]):
        tensor = await self._load_audio(audio)
        response = self.audio.transcribe(tensor)  # type: ignore
        if isinstance(response, dict):  # type: ignore
            segments = [
                Segment(**r) for r in response.get("segments", []) if isinstance(r, dict)  # type: ignore
            ]
            return Transcript(
                text=response.get("text", ""), segments=segments, language=response.get("language", "")  # type: ignore
            )  # type: ignore
        segments = [Segment(**r) for r in response if isinstance(r, dict)]  # type: ignore
        return Transcript(text="", segments=segments, language="")

    async def _load_audio(self, audio: AsyncGenerator[bytes, None]):
        with TemporaryDirectory() as tmp:
            path = f"{tmp}/{str(uuid4())}.mp3"
            with open(path, "wb") as file:
                async for chunk in audio:
                    file.write(chunk)
            response = whisper.load_audio(path)
            os.remove(path)
            return response
    async def run(self, *, audio: AsyncGenerator[bytes, None]):
        return await self.transcribe(audio)


class Video(BaseModel):
    """Transcribes a video. With the detailed token output"""

    video_id: str = Field(...)
    title: Optional[str] = Field(default=None)
    thumbnail: Optional[str] = Field(default=None)
    length: Optional[int] = Field(default=None)
    views: Optional[int] = Field(default=None)

    @computed_field
    @cached_property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    async def stream(self) -> AsyncGenerator[bytes, None]:
        with BytesIO() as buffer:
            stream = self._pull_stream()
            stream.stream_to_buffer(buffer)
            buffer.seek(0)
            for chunk in buffer:
                yield chunk

    def _pull_stream(self):
        stream = YouTube(self.url).streams.filter(only_audio=True).first()  # type: ignore
        assert stream is not None
        return stream

    async def search(self) -> AsyncGenerator[str, None]:
        async with AsyncClient() as client:
            response = await client.get(
                self.url,
            )
            for match in video_id_pattern.finditer(response.text):
                url = match.group(1)
                yield url.split("=")[-1]
