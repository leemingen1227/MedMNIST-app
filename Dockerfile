# this is an official Python runtime, used as the parent image
FROM python:3.8

# set the working directory in the container to /app
WORKDIR /app

# add the current directory to the container as /app
ADD . /app

RUN pip install -r requirements.txt

CMD python app.py

