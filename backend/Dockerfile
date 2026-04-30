FROM python:3.12

WORKDIR /app

COPY ./requirements-cpu.txt /app/requirements.txt

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . /app

CMD ["fastapi", "run", "server.py", "--port", "8000"]