apt install python3.10-venv python3.10
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

nohup python -m tf3.training.main > ./train_$(date +%Y%m%d_%H%M%S).txt 2>&1 &