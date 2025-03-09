# Robo_advising

# **Robo Advisor Project**
This project implements a **Robo Advisor** using **machine learning and financial optimization techniques**. The system allows users to determine their **risk tolerance** and receive an optimized **investment portfolio allocation** based on historical financial data.

## **Project Structure**
The project consists of four main components:

1. **Lab2a.ipynb** - Data Preprocessing
2. **Lab2b.ipynb** - Machine Learning for Risk Tolerance Prediction
3. **Lab2c.py** - Robo Advisor Dashboard
4. **Lab2d.py** - Enhanced Dashboard with Additional Features

---

## **1️⃣ Lab2a.ipynb - Data Preprocessing**
This Jupyter Notebook is responsible for:
- **Loading and cleaning investor data** for model training.
- **Handling missing values and standardizing datasets**.
- **Preparing training data** for risk tolerance prediction.
- **Exporting processed data for use in machine learning**.

---

## **2️⃣ Lab2b.ipynb - Machine Learning for Risk Tolerance Prediction**
This notebook focuses on **machine learning model training** to predict an investor’s risk tolerance. It includes:
- **Feature selection from investor data** (e.g., age, income, education, marital status, etc.).
- **Training a machine learning model** (e.g., regression, decision trees, or another ML algorithm).
- **Evaluating model performance** using metrics like accuracy, R², or mean squared error.
- **Saving the trained model (`build_lab2b.sav`)** for deployment in the Robo Advisor dashboard.

The output from this step is later used in `Lab2c.py` and `Lab2d.py` to make real-time risk tolerance predictions.

---

## **3️⃣ Lab2c.py - Robo Advisor Dashboard**
This script builds an **interactive web dashboard using Dash** that allows users to:
- **Input investor characteristics** (Age, Income, Risk Tolerance, etc.).
- **Predict risk tolerance** using the trained machine learning model (`build_lab2b.sav`).
- **Select investment assets** from a list of available stocks.
- **Generate an optimal asset allocation** using a **Mean-Variance Portfolio Optimization Model**.
- **Visualize the portfolio allocation** via bar charts.
- **Track portfolio performance over time** through interactive line charts.
- **Interact with the dashboard** using sliders, dropdowns, and buttons.

---

## **4️⃣ Lab2d.py - Enhanced Robo Advisor Dashboard**
This script extends `Lab2c.py` by adding:
- **Date selection for portfolio visualization** to see past allocation changes.
- **Custom investment amount selection** to tailor portfolio calculations.
- **A regression analysis module** to compare the portfolio’s performance against the S&P 500 index.
- **Dynamic pie charts** for portfolio allocation on different dates.
- **A scatter plot for Alpha and Beta analysis** of the portfolio vs. the benchmark.

---

## **🔧 Setup & Installation**
To run the dashboards, install the required dependencies:
```bash
pip install dash plotly pandas numpy cvxopt scikit-learn
```
To start the dashboard, run:
```bash
python Lab2c.py  # or Lab2d.py for the enhanced version
```
Then open the **localhost URL** in your browser.

---

## **📈 Features**
✅ **Machine learning-based risk tolerance prediction**  
✅ **Optimized stock allocation using Mean-Variance Portfolio Theory**  
✅ **Interactive UI for investors**  
✅ **Visualizes portfolio performance & stock allocations**  
✅ **Regression analysis on portfolio vs S&P 500**  

---