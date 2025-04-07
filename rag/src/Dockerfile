FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Use Gunicorn to serve the app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app"]
# Run in development mode
#CMD ["python", "run.py"]
