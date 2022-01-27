# Predict Birth Weights

This is used to predict the weight of a newborn child. Inspired by a round of bets my family made regarding the weight and time of birth our future cousin.

## Data

The data was gathered from the [CDC Website](https://www.cdc.gov/nchs/data_access/vitalstatsonline.htmhttps://www.cdc.gov/nchs/data_access/vitalstatsonline.html). On the website, you'll find a flat file (around 5gb) and a pdf file that outlines how the data is interpreted.

The most difficult part of this project is dealing with a 5 gb file with no formatting other than a pdf explaining which position in a string corresponds to each variable.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install requirements:

```bash
pip install -r req.txt
```

## Run

First, update the dictionary with the values for your prediction. For my test, Decision Tree performed the best, so we ran the prediction with that model.

Run code either in an IDE or in the terminal with:
```bash
python3 predict_weight.py
```

## Improvements

- This was written in just under two hours initially, so no feature selection was done on the variables. Ideally, we'd look at more than just 5 predictor variables and analyze the amount of variance explained by each variable to decide what to keep and what to remove.
- Automatically run predictions on the best performing model using the 'metrics' package
- Allow variable inputs via terminal

## Bet results

I never actually ran this to predict the weight of my new-born cousin as this required me to ask my cousin in-law how much weight she gained during pregnancy. I went with a safe bet of 8 pounds 10 oz and lost.
