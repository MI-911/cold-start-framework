FROM python:3.7
ADD requirements.txt /app/
WORKDIR /app
RUN pip install -r requirements.txt
ADD . /app/
ENTRYPOINT ["python", "entrypoints/interview.py"]