# Python + Chromium for Pypuppeteer

FROM python:3.10

ENV PYTHONUNBUFFERED=1

ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN apt-get update && apt-get install -y \
	chromium \
	chromium-driver \
	&& rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip


WORKDIR /app


COPY requirements.txt .


RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
