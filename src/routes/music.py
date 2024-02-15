from io import BytesIO
from pathlib import Path

import scipy
import scipy.io.wavfile
from fastapi import APIRouter, UploadFile
from transformers import pipeline

from ..services import ObjectStorage

synthesiser = pipeline("text-to-audio", "facebook/musicgen-small")
storage = ObjectStorage()

app = APIRouter(tags=["music"])


@app.post("/music")
async def generate_music(prompt: str, key: str):
    music = synthesiser(prompt, forward_params={"do_sample": True})
    path = Path(key)
    scipy.io.wavfile.write(path, rate=music.sampling_rate, data=music.audio)  # type: ignore
    return await storage.put(
        key=key,
        file=UploadFile(
            BytesIO(
                path.read_bytes(),
            ),
            filename=f"{path}.wav",
        ),
    )
