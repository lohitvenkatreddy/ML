'''A3. Please refer to the data present in “IRCTC Stock Price” data sheet of the above excel file. Do the
following after loading the data to your programming platform.
• Calculate the mean and variance of the Price data present in column D.
(Suggestion: if you use Python, you may use statistics.mean() &
statistics.variance() methods).
• Select the price data for all Wednesdays and calculate the sample mean. Compare the mean
with the population mean and note your observations.
• Select the price data for the month of Apr and calculate the sample mean. Compare the
mean with the population mean and note your observations.
• From the Chg% (available in column I) find the probability of making a loss over the stock.
(Suggestion: use lambda function to find negative values)
• Calculate the probability of making a profit on Wednesday.
• Calculate the conditional probability of making profit, given that today is Wednesday.
• Make a scatter plot of Chg% data against the day of the week'''

import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt

def load_stock_data(file_path, sheet_name, date_column):
    data = pd.read_excel(file_path, sheet_name=sheet_name)
    data[date_column] = pd.to_datetime(data[date_column], errors="coerce")  # Convert date column
    return data

def calculate_mean_variance(data, price_column):
    mean_price = statistics.mean(data[price_column])
    variance_price = statistics.variance(data[price_column])
    return mean_price, variance_price

def filter_wednesdays(data, day_column, price_column):
    data[day_column] = data[day_column].astype(str).str.strip().str.lower()
    wed_data = data[data[day_column].isin(["wednesday", "wed"])]
    if len(wed_data) == 0:
        return None, None
    sample_mean = statistics.mean(wed_data[price_column])
    return sample_mean, wed_data

def filter_month(data, date_column, price_column, month_name):
    month_data = data[data[date_column].dt.month_name() == month_name]
    if len(month_data) == 0:
        return None, None
    sample_mean = statistics.mean(month_data[price_column])
    return sample_mean, month_data

def probability_of_loss(data, change_column):
    total_days = len(data)
    loss_days = len(data[data[change_column] < 0])
    return loss_days / total_days if total_days != 0 else 0

def probability_of_profit_on_wednesday(data, day_column, change_column):
    data[day_column] = data[day_column].astype(str).str.strip().str.lower()
    wednesday_data = data[data[day_column].isin(["wednesday", "wed"])]
    total_wednesdays = len(wednesday_data)
    if total_wednesdays == 0:
        return 0
    profit_days = len(wednesday_data[wednesday_data[change_column] > 0])
    return profit_days / total_wednesdays

def conditional_probability_profit_given_wednesday(data, day_column, change_column):
    return probability_of_profit_on_wednesday(data, day_column, change_column)

def plot_chg_percent_vs_day(data, day_column, change_column):
    plt.figure(figsize=(8, 5))
    plt.scatter(data[day_column], data[change_column], color='blue')
    plt.title("Change % vs Day of the Week")
    plt.xlabel("Day of the Week")
    plt.ylabel("Chg%")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file_path = "LabData.xlsx"
    sheet_name = "IRCTC Stock Price"
    price_column = 'Price'
    change_column = 'Chg%'
    day_column = 'Day'
    date_column = 'Date'

    data = load_stock_data(file_path, sheet_name, date_column)

    mean_price, variance_price = calculate_mean_variance(data, price_column)
    wed_sample_mean, wed_data = filter_wednesdays(data, day_column, price_column)
    apr_sample_mean, apr_data = filter_month(data, date_column, price_column, "April")
    loss_probability = probability_of_loss(data, change_column)
    profit_wed_probability = probability_of_profit_on_wednesday(data, day_column, change_column)
    conditional_profit_probability = conditional_probability_profit_given_wednesday(data, day_column, change_column)

    print(f"Mean of Price: {mean_price:.2f}")
    print(f"Variance of Price: {variance_price:.2f}")

    if wed_sample_mean is not None:
        print(f"Sample mean (Wednesdays): {wed_sample_mean:.2f}")
    else:
        print("No Wednesday data found.")

    if apr_sample_mean is not None:
        print(f"Sample mean (April): {apr_sample_mean:.2f}")
    else:
        print("No April data found.")

    print(f"Probability of loss: {loss_probability * 100:.2f}%")
    print(f"Probability of profit on Wednesday: {profit_wed_probability * 100:.2f}%")
    print(f"Conditional probability of profit given Wednesday: {conditional_profit_probability * 100:.2f}%")

    plot_chg_percent_vs_day(data, day_column, change_column)
