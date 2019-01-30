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

Ensure that folder `~/tmp` exists (and that you can write/read from it).

```bash
cp data/tickets.json ~/tmp/
python src/util/script.py
python src/datasci/linearSVC.py

# one line version (so that it's in bash history)
python src/util/script.py; python src/datasci/linearSVC.py
```

## Notes

linearSVC.py seems to be the best balance between not overfitting and still being accurate. gradient.py can reach ~84% with particular parameters (most likely overfitted) and cross_val.py sits around 70% (not as accurate). linearSVC.py uses a LinearSVC with sklearn's StandardScaler() to scale and normalize the data. linearSVC.py achieves about 78% accuracy on both the training data and test data, even when the script.py is run each time (randomizing training vs testing datasets).

## To Do

- [x] Add row number as parameter
- [x] Optimize process past simply trying every option
