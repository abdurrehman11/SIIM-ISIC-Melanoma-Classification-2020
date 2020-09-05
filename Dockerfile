FROM python:3.8.1

LABEL dockerize skin cancer web-app

RUN adduser --disabled-password abdurrehman

WORKDIR /home/abdurrehman/app

RUN chown -R abdurrehman:abdurrehman /home/abdurrehman

COPY ./requirements.txt /home/abdurrehman/app
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt
COPY . /home/abdurrehman/app