FROM python:3.11-slim

WORKDIR /app

COPY ./templates ./templates
COPY app.py app.py
COPY requirement.txt requirement.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Expose port 5000 for Flask
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]