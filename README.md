# Zoomcamp -  Capstone Project - Pricy

## Problem description

Pricing a short-term rental property is a complex task influenced by many factors such as location, property type, capacity, host characteristics, and available amenities. Hosts often struggle to determine an optimal price that is competitive in the market while still maximizing revenue. Manual pricing can be inconsistent, subjective, and may not adapt well to market patterns.

The objective of this project is to predict the nightly price of a rental listing based on its attributes. This is a regression problem, where the target variable is a continuous numeric value (price), and the input consists of structured data (numerical and categorical features) as well as unstructured text data (amenities description).

## How a Model Can Be Used to Solve the Problem

A machine learning regression model can learn patterns from historical rental listings and their prices to make accurate predictions for new or existing listings.

1. Feature Representation

Each listing is transformed into a numerical feature vector using:

- Numerical features such as number of bedrooms, bathrooms, location coordinates, and minimum nights
- Categorical features such as neighbourhood, property type, and room type, encoded using one-hot encoding
- Text features derived from the amenities list, converted into numerical form using TF-IDF vectorization
- This preprocessing allows the model to handle heterogeneous data sources in a unified format.

2. Model Training

A regression model (e.g., Decision Tree Regressor, Random Forest Regressor, or Gradient Boosting model) is trained on historical data where the true price is known.

During training, the model learns:

- How different features influence price
- Interactions between variables (e.g., location + room type)
- Non-linear relationships between listing attributes and price

3. Prediction

Once trained, the model can:

- Predict the expected price for a new listing
- Suggest price adjustments for existing listings
- Support hosts in making data-driven pricing decisions
- Given a listing’s attributes, the model outputs a predicted nightly price.

4. Evaluation

Model performance is evaluated using Root Mean Squared Error (RMSE), which measures the average difference between predicted and actual prices. A lower RMSE indicates better predictive accuracy.

Practical Applications

- Helping hosts price listings competitively
- Identifying under-priced or over-priced properties
- Supporting automated pricing tools for rental platforms
- Improving market transparency and consistency

## Local Setup

Simply install [uv](https://docs.astral.sh/uv/getting-started/installation/) using the following command and sync dependencies.

```bash
pip install uv
uv sync
```

To run Jupyter Notebook run:

```bash
uv run jupyter lab
```

To run FastAPI server and access interactive API doc run:
```bash
uv run fastapi run predict.py
```

## Docker Setup

Using [Dockerfile](./Dockerfile) we can create an image with our trained model, run FastAPI server and expose a port to the api:

```
docker build -t pricy .
docker run -p 8000:8000 -it --rm pricy
```

Interactive API can then be found [here](http://127.0.0.1:8000/docs#/default/predict_price_predict_price_post).

## EDA

Exploratory Data Analysis is present in [notebook.ipynb](./notebook.ipynb).

## Model training

Model training is done post EDA in [notebook.ipynb](./notebook.ipynb).

## Exporting notebook to script

Model training lives in its own module [train.py](./train.py) and can be reproduced using the command
```bash
uv run python train.py
```

Prediction and API in its own module [predict.py](./predict.py) and can be run using
```bash
uv run fastapi run predict.py
```
Interactive API should be visible [here](http://127.0.0.1:8000/docs#/default/predict_price_predict_price_post).

## Reproducibility

Notebook, model training and prediction api can be easily reproduced as explain in the steps above.

## Model deployment

Model is deployed using FastAPI defined in [predict.py](./predict.py).

## Dependency and environment management

Dependency and Virtual Environment is maintained by uv, via [pyproject.toml](./pyproject.toml) and [uv.lock](./uv.lock) as defined in [Local Setup](#local-setup).

## Containerization

[Dockerfile](./Dockerfile) handles creating image that can be used to run API container as defined in [Docker Setup](#docker-setup).

## Cloud deployment

Public deployment can be accessed here - [https://pricy-production.up.railway.app/docs#/default/predict_price_predict_price_post](https://pricy-production.up.railway.app/docs#/default/predict_price_predict_price_post).
