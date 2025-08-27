FROM python:3.11-slim-buster

# Install system dependencies for sounddevice, torchaudio, and PortAudio
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    libportaudiocpp0 \
    portaudio19-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install git
RUN apt-get update && apt-get install -y git
RUN pip install --upgrade pip

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Command to run the app
CMD ["python", "WhisperTurbofinal.py"]
