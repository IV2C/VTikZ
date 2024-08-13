FROM alpine:3.20.2

WORKDIR /varbench_evaluation
COPY varbench/ varbench/


RUN apk add python3 py3-pip
RUN apk add git


COPY docker/requirements.txt requirements.txt


#setting up python environment

RUN python3 -m venv .env
RUN . .env/bin/activate;pip3 install -r requirements.txt
