# Using SKLearn to Predict Ticket Price Viability

## Setup

### Without Virtualenv

```bash
git clone https://github.com/samuelstevens/buckeyetix-ml.git
cd buckeyetix-ml

pip install -r requirements.txt
```

### With Virtualenv and Python 3 as default python

```bash
git clone https://github.com/samuelstevens/buckeyetix-ml.git
cd buckeyetix-ml

python3 -m venv venv

. venv/bin/activate

pip install -r requirements.txt
```

## Running

### Local

Ensure that folder `~/tmp` exists (and that you can write/read from it).

```bash
cp data/tickets.json ~/tmp/
python src/script.py
python src/datasci/nested_cross_val.py
```

## To Do

- [x] Add row number as parameter
- [x] Optimize process past simply trying every option
