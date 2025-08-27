FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

ENV PYTHONUNBUFFERED 1

WORKDIR /usr/src/app

# Install necessary tools
RUN apt-get update && apt-get install -y git wget unzip python3-pip && rm -rf /var/lib/apt/lists/*

# Install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Install gdown for downloading from Google Drive
RUN pip install gdown

# Download and move weights to the correct directories
RUN gdown https://drive.google.com/uc?id=1Zw7zEhr8ukTPrT89Nmjo4An9NlZyp7P0 -O /usr/src/app/mlx_models/distil-large-v3/weights.npz
RUN gdown https://drive.google.com/uc?id=1zKNFUtoV7p8ApXI6P-XBKhoc48PhyBYk -O /usr/src/app/mlx_models/large-v3/weights.npz

# Copy the rest of the app
COPY . .

# Run the pipeline
CMD ["python3", "all_pipeline.py", "--local_mac_optimal_settings", "--device", "mps", "--stt_model_name", "large-v3", "--language", "auto"]
