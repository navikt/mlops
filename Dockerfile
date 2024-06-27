FROM huggingface/optimum-nvidia

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt


COPY . .

CMD ["python", "app.py"]
