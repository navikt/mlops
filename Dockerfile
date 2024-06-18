FROM python:3.9

WORKDIR /app

EXPOSE 8080

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
