FROM python:3.14

WORKDIR /app

COPY ./req_win.txt /app/req_win.txt

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir --upgrade -r /app/req_win.txt

COPY . /app

CMD ["fastapi", "run", "server.py", "--port", "8000"]