FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY /lora /app/lora-model
COPY . .

CMD ["python", "app.py"]
