FROM postgres:latest

ENV POSTGRES_USER postgres
ENV POSTGRES_PASSWORD postgres
ENV POSTGRES_DB postgres


RUN apt-get update && apt-get install -y \
	build-essential \
	postgresql-server-dev-all \
	git \
	&& rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
git clone --branch v0.5.1 https://github.com/pgvector/pgvector.git && \
cd /tmp/pgvector && \
make && \
make install 

COPY ./pg_hba.conf /var/lib/postgresql/data/pg_hba.conf