FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

# Copy the source code
COPY src /app/src
COPY Dataset /app/Dataset
COPY __init__.py /app/

EXPOSE 5000

# Create the start script in a location that won't be overwritten
RUN mkdir -p /scripts && \
    echo '#!/bin/bash\n\
python /app/src/retrain_models.py && \
python /app/src/api.py' > /scripts/start.sh && \
chmod +x /scripts/start.sh

# Start both the training and API
CMD ["/scripts/start.sh"]
