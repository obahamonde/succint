FROM ubuntu:latest

RUN apt-get update && apt-get install -y python3 python3-pip

RUN apt-get install -y curl
RUN apt-get install -y git
RUN curl -sL https://deb.nodesource.com/setup_16.x | bash -
RUN apt-get install -y nodejs

RUN apt-get install -y ffmpeg

RUN npm install -g lighthouse
RUN npm install -g prisma



WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get install -y wget gnupg
RUN wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
RUN apt-get update && apt-get install -y google-chrome-stable

COPY . .
RUN pip3 install -r requirements.txt
RUN pip3 install -U prisma

RUN python3 -m prisma generate
RUN python3 -m prisma py fetch


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080","--reload"]