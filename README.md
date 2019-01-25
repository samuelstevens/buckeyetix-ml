# Using Keras to Predict Ticket Price Viability

## Setup

### Without Virtualenv

```bash
git clone https://github.com/samuelstevens/buckeyetix-ml.git
cd buckeyetix-ml


npm install
pip install -r requirements.txt
```

### With Virtualenv and Python 2.7 as default python

```bash
git clone https://github.com/samuelstevens/buckeyetix-ml.git
cd buckeyetix-ml

npm install

virtualenv venv --python $(which python) # if 2.7
source venv/bin/activate

pip install -r requirements.txt
```

## Running

Ensure that folder `~/tmp` exists (and that you can write/read from it).
