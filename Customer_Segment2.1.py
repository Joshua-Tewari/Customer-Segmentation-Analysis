import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
file_path = 'Online Retail.xlsx'
df = pd.read_excel(file_path)

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Drop rows with missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Add TotalPrice column
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Basic statistics
total_revenue = df['TotalPrice'].sum()
total_quantity = df['Quantity'].sum()
unique_customers = df['CustomerID'].nunique()
num_invoices = df['InvoiceNo'].nunique()

print("Total Revenue: £{:.2f}".format(total_revenue))
print("Total Quantity Sold:", total_quantity)
print("Unique Customers:", unique_customers)
print("Number of Invoices:", num_invoices)

# Sales over time
df.set_index('InvoiceDate', inplace=True)
monthly_sales = df['TotalPrice'].resample('M').sum()

plt.figure(figsize=(12, 6))
monthly_sales.plot()
plt.title('Monthly Sales Revenue')
plt.ylabel('Revenue (£)')
plt.xlabel('Month')
plt.grid()
plt.show()

# Best-selling products
product_sales = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
product_sales.plot(kind='bar')
plt.title('Top 10 Best-Selling Products')
plt.ylabel('Total Quantity Sold')
plt.xlabel('Product')
plt.xticks(rotation=45)
plt.show()

# Top customers
top_customers = df.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
top_customers.plot(kind='bar')
plt.title('Top 10 Customers by Revenue')
plt.ylabel('Revenue (£)')
plt.xlabel('Customer ID')
plt.xticks(rotation=45)
plt.show()

# Country analysis
country_sales = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12, 6))
country_sales.plot(kind='bar')
plt.title('Top 10 Countries by Revenue')
plt.ylabel('Revenue (£)')
plt.xlabel('Country')
plt.xticks(rotation=45)
plt.show()

# Generate a word cloud for product descriptions


# Customer Segmentation using K-Means Clustering
customer_df = df.groupby('CustomerID').agg({
    'TotalPrice': 'sum',
    'InvoiceNo': 'count',
    'Quantity': 'sum'
}).reset_index()

scaler = StandardScaler()
customer_df_scaled = scaler.fit_transform(customer_df[['TotalPrice', 'InvoiceNo', 'Quantity']])

kmeans = KMeans(n_clusters=5, random_state=42)
customer_df['Cluster'] = kmeans.fit_predict(customer_df_scaled)

plt.figure(figsize=(12, 6))
for i in range(5):
    plt.scatter(customer_df[customer_df['Cluster'] == i]['TotalPrice'], 
                customer_df[customer_df['Cluster'] == i]['Quantity'], label=f'Cluster {i}')
plt.xlabel('Total Price')
plt.ylabel('Total Quantity')
plt.title('Customer Segmentation')
plt.legend()
plt.grid()
plt.show()

# Time Series Forecasting using ARIMA
train_size = int(len(monthly_sales) * 0.8)
train, test = monthly_sales[0:train_size], monthly_sales[train_size:]

model = ARIMA(train, order=(5,1,0))  # You can adjust the order parameters as needed
model_fit = model.fit()
predictions = model_fit.forecast(steps=len(test))

plt.figure(figsize=(12, 6))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(test.index, predictions, label='Forecast')
plt.title('Monthly Sales Forecast')
plt.ylabel('Revenue (£)')
plt.xlabel('Month')
plt.legend()
plt.grid()
plt.show()

mse = mean_squared_error(test, predictions)
print('Mean Squared Error:', mse)
