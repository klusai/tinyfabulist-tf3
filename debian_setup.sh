apt install python3.11-venv python3.11
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

nohup training/main.py > ./train_$(date +%Y%m%d_%H%M%S).txt 2>&1 &