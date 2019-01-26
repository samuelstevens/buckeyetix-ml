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

### Local

Ensure that folder `~/tmp` exists (and that you can write/read from it).

```bash
cp data/tickets.json ~/tmp/
python src/script.py
python src/models/all_models.py
```

### Remote

```bash
python -u src/models/all_models.py > ~/tmp/$(date +"%d-%h-%y-%T").out &
```

## To Do

- [x] Add row number as parameter
- [ ] Add weather (numerical rating maybe) as parameter
- [ ] Optimize process past simply trying every option
- [ ] Have output written to a file to see progress/errors
- [ ] Record time take to find result

current process (on stdlinux): 23584

```

```
