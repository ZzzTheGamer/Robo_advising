# Robo_advising

# **Robo Advisor Project**
This project implements a Robo Advisor system for portfolio management using Python and Dash. The system helps investors determine their risk tolerance and allocate assets accordingly.

## **Project Structure**
The project consists of four key components:

1. **Lab2a.ipynb** - Data Preprocessing & Model Training
2. **Lab2b.ipynb** - Portfolio Optimization Model
3. **Lab2c.py** - Dashboard for Robo Advisor
4. **Lab2d.py** - Enhanced Dashboard with Additional Features

---

## **1️⃣ Lab2a.ipynb - Data Preprocessing & Model Training**
This Jupyter Notebook handles:
- Data preprocessing for investor profiles and asset price history.
- Cleaning missing data and preparing it for machine learning.
- Training a **risk tolerance prediction model** using regression.
- Saving the trained model as `build_lab2b.sav` for later use in the dashboard.

---

## **2️⃣ Lab2b.ipynb - Portfolio Optimization Model**
This notebook contains:
- Implementation of a **Mean-Variance Portfolio Optimization Model**.
- Using `cvxopt` to calculate optimal asset allocation based on **investor risk tolerance**.
- Generating allocation weights for stocks based on historical price data.
- Saving processed financial data for use in dashboards (`SP500Data.csv`).

---

## **3️⃣ Lab2c.py - Robo Advisor Dashboard**
This script:
- Implements a **Dash-based web application** for Robo Advisory.
- Allows users to input investor characteristics (Age, Income, Risk Tolerance, etc.).
- Predicts **risk tolerance** using the trained model (`build_lab2b.sav`).
- Displays **optimal asset allocation** using bar charts.
- Shows **portfolio performance over time** using a line chart.
- Uses **interactive controls** like sliders, dropdowns, and buttons for user input.

---

## **4️⃣ Lab2d.py - Enhanced Robo Advisor Dashboard**
This script extends `Lab2c.py` by:
- Adding **date selection for portfolio performance visualization**.
- Allowing users to specify **starting capital for investments**.
- Implementing **dynamic pie charts** to visualize **portfolio allocation over time**.
- Including **a regression analysis module** to measure the relationship between portfolio performance and the S&P 500 index.
- Enabling **real-time updates** via a date slider.

---

## **🔧 Setup & Installation**
To run the dashboards, install the required dependencies:
```bash
pip install dash plotly pandas numpy cvxopt
```
To start the dashboard, run:
```bash
python Lab2c.py  # or Lab2d.py for the enhanced version
```
Then open the **localhost URL** in your browser.

---

## **📈 Features**
✅ **Predicts investor risk tolerance**  
✅ **Optimizes stock allocation using Mean-Variance Portfolio Theory**  
✅ **Interactive UI for investors**  
✅ **Visualizes portfolio performance & stock allocations**  
✅ **Regression analysis on portfolio vs S&P 500**  

---