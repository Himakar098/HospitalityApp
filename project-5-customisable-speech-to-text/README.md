[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=15486635&assignment_repo_type=AssignmentRepo)
# How to Localize the All Pipeline of this Repository or Use Docker to Deploy the All Pipeline

## For Localize users
### Step 1:
First, clone this repository. The repository link is:
```bash
git clone https://github.com/uwa-computer-science/project-5-customisable-speech-to-text-team2/tree/7829b5d82afe62f834ab23727ec68e81d9c6b4b0/All%20pipeline
```
### Step 2:
Get in to the repository folder
```bash
cd project-5-customisable-speech-to-text-team2/All\ pipeline
```

### Step 3:
Install the required dependencies.
```bash
pip install -r requirements.txt
```
For mac users:
```bash
pip install -r requirements_mac.txt
```

### Step 4:
Download the `distil-large-v3` and `large-v3` weights.

To download the weights, use the following commands and store the downloaded weights in the corresponding directories.

For `distil-large-v3`:
```bash
# Download the distil-large-v3 weights
gdown https://drive.google.com/drive/folders/19ym3kFked0jXtu6XiJcvJHtl0QGcawhC?usp=sharing
# Move the weights to the correct directory
mv weights.npz /project-5-customisable-speech-to-text-team2/All\ pipeline/mlx_models/distil-large-v3/
```
For `large-v3`:
```bash
# Download the large-v3 weights
gdown https://drive.google.com/file/d/1zKNFUtoV7p8ApXI6P-XBKhoc48PhyBYk/view?usp=sharing
# Move the weights to the correct directody
mv weights.npz /project-5-customisable-speech-to-text-team2/All\ pipeline/mlx_models/large-v3/
```

### Step 5:
Run the command in the cmd_line.txt file in the repository, or directly use the following command to run the pipeline:
```bash
python all_pipeline.py \
    --local_mac_optimal_settings \
    --device mps \
    --stt_model_name large-v3 \
    --language auto
```

## For Docker Users
### Step 1:
First, clone this repository. The repository link is:
```bash
git clone https://github.com/uwa-computer-science/project-5-customisable-speech-to-text-team2/tree/7829b5d82afe62f834ab23727ec68e81d9c6b4b0/All%20pipeline
```
### Step 2:
Get in to the repository folder
```bash
cd project-5-customisable-speech-to-text-team2/All\ pipeline
```

### Step 3:
Run the docker-compose on your terminal
```bash
docker-compose up
```
For Mac Users (especially M1, M2 chips users), run this code:
```bash
DOCKERFILE=Dockerfile.arm64 docker-compose up
```



