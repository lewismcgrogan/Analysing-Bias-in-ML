# Census Income Fairness Project (Adult Dataset)

## Data
Place these files in ./data:
- adult.data
- adult.test
- adult.names

## Run
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m src.run_experiment --data_dir data --out_dir results/runs
