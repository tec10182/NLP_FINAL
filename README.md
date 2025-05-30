# 1. Translate dialects

```
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118 # For GPU

conda create --name value python=3.10.13
conda activate value

cd multi-value
pip install -r requirements.txt
python translate.py