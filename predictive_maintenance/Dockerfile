FROM python:3.10-slim
ADD . /app
WORKDIR /app
COPY models /app/models
RUN pip install -r requirements.txt
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]