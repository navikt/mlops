FROM python:3.11

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt


COPY /lora/fine_tuned_lora ./lora
COPY . .

CMD ["python", "app.py"]
