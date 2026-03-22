pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn


//ex.py

import pandas as pd

df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

print(df.columns)

//1_Measures_of_Central_Tendency.py

import pandas as pd

# change dataset name if needed
df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# change column if faculty asks another column
sales = df['Sales']

mean_value = sales.mean()
median_value = sales.median()
mode_value = sales.mode()

print("Mean:", mean_value)
print("Median:", median_value)
print("Mode:", mode_value)

# show rows and columns after operation
print("\nRows and Columns:", df.shape)
print(df.head())

//1_Measures_of_Dispersion.py
import pandas as pd

df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# change column if needed
sales = df['Sales']

range_value = sales.max() - sales.min()
print("Range:", range_value)

variance = sales.var()
print("Variance:", variance)

std_dev = sales.std()
print("Standard Deviation:", std_dev)

Q1 = sales.quantile(0.25)
Q3 = sales.quantile(0.75)

IQR = Q3 - Q1
print("Interquartile Range:", IQR)

# show rows and columns after operation
print("\nRows and Columns:", df.shape)
print(df.head())

//1_Shape_of_the_Distribution.py

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# change column if needed
sales = df['Sales']

plt.hist(sales, bins=10, rwidth=0.8)

plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.title("Sales Distribution")

plt.show()

# show rows and columns
print("\nRows and Columns:", df.shape)
print(df.head())

//2_Handling_Missing_Values.py

import pandas as pd

# CHANGE DATASET NAME IF NEEDED
df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# CHANGE COLUMN IF FACULTY ASKS ANOTHER COLUMN
df['Sales'].fillna(df['Sales'].mean(), inplace=True)

print("Missing values after handling:")
print(df.isnull().sum())

print("\nRows and Columns:", df.shape)
print(df.head())

//2_scaling.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# CHANGE COLUMN IF NEEDED
scaler = StandardScaler()

df['Sales_scaled'] = scaler.fit_transform(df[['Sales']])

print(df[['Sales','Sales_scaled']].head())

print("\nRows and Columns:", df.shape)
print(df.head())

//2_Normalization.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# CHANGE COLUMN IF NEEDED
scaler = MinMaxScaler()

df['Sales_normalized'] = scaler.fit_transform(df[['Sales']])

print(df[['Sales','Sales_normalized']].head())

print("\nRows and Columns:", df.shape)
print(df.head())

//2_Oversampling.py

import pandas as pd
from sklearn.utils import resample

df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# CHANGE TARGET COLUMN IF NEEDED
target = 'Category'

majority = df[df[target] == df[target].value_counts().idxmax()]
minority = df[df[target] == df[target].value_counts().idxmin()]

minority_upsampled = resample(minority,
replace=True,
n_samples=len(majority),
random_state=42)

df_oversampled = pd.concat([majority, minority_upsampled])

print("After Oversampling:")
print(df_oversampled[target].value_counts())
print("\nRows and Columns:", df.shape)
print(df.head())

//2_Oversampling.py(alternate)

import pandas as pd
from sklearn.utils import resample

# CHANGE DATASET NAME IF FACULTY GIVES ANOTHER DATASET
df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# CHANGE TARGET COLUMN IF DATASET HAS DIFFERENT CLASS LABEL
target = 'Category'

majority = df[df[target] == df[target].value_counts().idxmax()]
minority = df[df[target] == df[target].value_counts().idxmin()]

minority_upsampled = resample(
    minority,
    replace=True,
    n_samples=len(majority),
    random_state=42
)

df_oversampled = pd.concat([majority, minority_upsampled])

print("After Oversampling:")
print(df_oversampled[target].value_counts())

print("\nRows and Columns:", df.shape)
print(df.head())


//2_Undersampling.py

import pandas as pd
from sklearn.utils import resample

df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

target = 'Category'

majority = df[df[target] == df[target].value_counts().idxmax()]
minority = df[df[target] == df[target].value_counts().idxmin()]

majority_downsampled = resample(majority,
replace=False,
n_samples=len(minority),
random_state=42)

df_undersampled = pd.concat([majority_downsampled, minority])

print("After Undersampling:")
print(df_undersampled[target].value_counts())
print("\nRows and Columns:", df.shape)
print(df.head())

//2_Undersampling.py(alternate)

import pandas as pd
from sklearn.utils import resample

# CHANGE DATASET NAME IF FACULTY GIVES ANOTHER DATASET
df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# CHANGE TARGET COLUMN IF DATASET HAS DIFFERENT CLASS LABEL
target = 'Category'

majority = df[df[target] == df[target].value_counts().idxmax()]
minority = df[df[target] == df[target].value_counts().idxmin()]

majority_downsampled = resample(
    majority,
    replace=False,
    n_samples=len(minority),
    random_state=42
)

df_undersampled = pd.concat([majority_downsampled, minority])

print("After Undersampling:")
print(df_undersampled[target].value_counts())

print("\nRows and Columns:", df.shape)
print(df.head())

//2_Histogram.py

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

sales = df['Sales']   # change column if needed

plt.figure(figsize=(8,5))

plt.hist(sales, bins=15, color='skyblue', edgecolor='black')

plt.title("Histogram of Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")

plt.grid(True)

plt.show()

//2_BoxPlot.py

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# CHANGE COLUMN IF NEEDED
plt.boxplot(df['Sales'])

plt.title("Box Plot of Sales")

plt.show()

//2_Scatter_Plot.py

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# CHANGE COLUMN NAME IF FACULTY DATASET HAS DIFFERENT COLUMN
x = df['Sales']      # change if needed
y = df['Profit']     # change if needed

plt.figure(figsize=(8,5))

plt.scatter(x, y, color='red', alpha=0.6)

plt.title("Scatter Plot: Sales vs Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")

plt.grid(True)

plt.show()

//2_Line_Plot.py

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# convert Order Date to datetime
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')

# remove rows with invalid dates
df = df.dropna(subset=['Order Date'])

# group by month and calculate total sales
monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()

# sort months
monthly_sales = monthly_sales.sort_index()

months = monthly_sales.index.astype(str)
sales = monthly_sales.values

plt.figure(figsize=(10,5))

plt.plot(months, sales, marker='o')

plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.title("Monthly Sales Trend")

plt.xticks(months[::3], rotation=45)

plt.tight_layout()
plt.show()

//2_Line_Plot.py(alternate1)

import pandas as pd
import matplotlib.pyplot as plt

# CHANGE DATASET NAME IF FACULTY GIVES ANOTHER DATASET
df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# CHANGE COLUMN IF DATASET USES DIFFERENT NUMERIC COLUMN
y = df['Sales']

x = range(len(y))

plt.figure(figsize=(8,5))

plt.plot(x, y, marker='o')

plt.xlabel("Index")
plt.ylabel("Sales")
plt.title("Line Plot")

plt.grid(True)

plt.show()

print("\nRows and Columns:", df.shape)
print(df.head())

//2_Line_Plot.py(alternate)

import pandas as pd
import matplotlib.pyplot as plt

# CHANGE DATASET NAME IF FACULTY GIVES ANOTHER DATASET
df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# CHANGE COLUMN IF FACULTY USES ANOTHER COLUMN
y = df['Sales']

x = range(len(y))

plt.figure(figsize=(8,5))

plt.plot(x, y, marker='o')

plt.xlabel("Index")
plt.ylabel("Sales")
plt.title("Line Plot of Sales")

plt.grid(True)

plt.show()

print("\nRows and Columns:", df.shape)
print(df.head())

//2_Bar_Chart.py

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# CHANGE COLUMN IF NEEDED
df['Category'].value_counts().plot(kind='bar')

plt.title("Category Distribution")

plt.show()

//3_Covariance.py

import pandas as pd

# change dataset name if needed
df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# change columns if faculty asks other columns
sales = df['Sales']
profit = df['Profit']

cov_value = sales.cov(profit)

print("Covariance between Sales and Profit:", cov_value)

print("\nRows and Columns:", df.shape)
print(df.head())

//3_Correlation.py

import pandas as pd

df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# change columns if needed
sales = df['Sales']
profit = df['Profit']

corr_value = sales.corr(profit)

print("Correlation between Sales and Profit:", corr_value)

print("\nRows and Columns:", df.shape)
print(df.head())

//3_Heatmap.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# choose numeric columns
corr_matrix = df[['Sales','Profit','Discount']].corr()

sns.heatmap(corr_matrix, annot=True)

plt.title("Correlation Heatmap")

plt.show()

print("\nRows and Columns:", df.shape)

//3_Heatmap.py(alternate1)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# CHANGE DATASET NAME IF FACULTY GIVES ANOTHER DATASET
df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# Automatically select numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Calculate correlation matrix
corr_matrix = numeric_df.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.show()

print("\nRows and Columns:", df.shape)
print(df.head())

//3_Heatmap.py(alternate)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# CHANGE DATASET NAME IF FACULTY GIVES ANOTHER DATASET
df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# Select only numeric columns automatically
numeric_df = df.select_dtypes(include=['number'])

# Calculate correlation matrix
corr_matrix = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

plt.title("Correlation Heatmap")

plt.show()

print("\nRows and Columns:", df.shape)
print(df.head())

//3_ScatterPlot.py

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# change columns if needed
x = df['Sales']
y = df['Profit']

plt.scatter(x,y)

plt.xlabel("Sales")
plt.ylabel("Profit")

plt.title("Sales vs Profit")

plt.show()

print("\nRows and Columns:", df.shape)
print(df.head())

//Common_Preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# Remove missing values
df = df.dropna()

# Create classification column
median_profit = df['Profit'].median()
df['Profit_Class'] = np.where(df['Profit'] > median_profit, 1, 0)

# Encode text columns
le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop(['Profit','Profit_Class'], axis=1)
y = df['Profit_Class']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

print("Dataset Shape:", df.shape)
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)
print("First 5 rows:\n", df.head())

//4_Logistic_Regression.py

from Common_Preprocessing import X_train, X_test, y_train, y_test
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print("Classification Report:\n", classification_report(y_test, pred))

//4_Decision_Tree.py

from Common_Preprocessing import X_train, X_test, y_train, y_test
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Decision Tree")
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print("Classification Report:\n", classification_report(y_test, pred))

//4_rf.py

from Common_Preprocessing import X_train, X_test, y_train, y_test
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Random Forest")
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print("Classification Report:\n", classification_report(y_test, pred))

//4_svm.py

from Common_Preprocessing import X_train, X_test, y_train, y_test
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = SVC()

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("SVM")
print("Accuracy:", accuracy_score(y_test, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred))
print("Classification Report:\n", classification_report(y_test, pred))

//Exp5_Preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("Supermart Grocery Sales - Retail Analytics Dataset.csv")

# Remove missing values
df = df.dropna()

# Encode categorical columns
le = LabelEncoder()
for col in df.select_dtypes(include='object'):
    df[col] = le.fit_transform(df[col])

# ----------------------
# For Linear Regression
# ----------------------

X_linear = df.drop(['Profit'], axis=1)
y_linear = df['Profit']

X_train_lin, X_test_lin, y_train_lin, y_test_lin = train_test_split(
    X_linear, y_linear, test_size=0.3, random_state=42)

# ----------------------
# For Logistic Regression
# ----------------------

median_profit = df['Profit'].median()
df['Profit_Class'] = np.where(df['Profit'] > median_profit, 1, 0)

X_log = df.drop(['Profit','Profit_Class'], axis=1)
y_log = df['Profit_Class']

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_log, y_log, test_size=0.3, random_state=42)

print("Dataset Shape:", df.shape)
print("First 5 rows:\n", df.head())

//5_LinearRegression.py

from Exp5_Preprocessing import X_train_lin, X_test_lin, y_train_lin, y_test_lin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

model = LinearRegression()

model.fit(X_train_lin, y_train_lin)

pred = model.predict(X_test_lin)

print("Linear Regression Performance")

print("MAE:", mean_absolute_error(y_test_lin, pred))
print("MSE:", mean_squared_error(y_test_lin, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test_lin, pred)))
print("R2 Score:", r2_score(y_test_lin, pred))


//5_LogisticRegression.py

from Exp5_Preprocessing import X_train_log, X_test_log, y_train_log, y_test_log
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = LogisticRegression(max_iter=5000)

model.fit(X_train_log, y_train_log)

pred = model.predict(X_test_log)

print("Logistic Regression Performance")

print("Accuracy:", accuracy_score(y_test_log, pred))
print("Confusion Matrix:\n", confusion_matrix(y_test_log, pred))
print("Classification Report:\n", classification_report(y_test_log, pred))