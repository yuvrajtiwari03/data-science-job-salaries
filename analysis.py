import pandas as pd

# dataset load karo
df = pd.read_csv("ds_salaries.csv")

# Top 5 rows dekho
print(df.head())

# Shape aur info
print("Shape:", df.shape)
print(df.info())
df = df.drop(columns=['Unnamed: 0'])
#Experience level ko full name se replace kiya

df['experience_level'] = df['experience_level'].replace({
    'EN': 'Entry',
    'MI': 'Mid',
    'SE': 'Senior',
    'EX': 'Executive'
})
# same as it in
df['employment_type'] = df['employment_type'].replace({
    'FT': 'Full-time',
    'PT': 'Part-time',
    'CT': 'Contract',
    'FL': 'Freelance'
})
#remote ratio ko readable bnanya

df['job_type'] = df['remote_ratio'].replace({
    0: 'Onsite',
    50: 'Hybrid',
    100: 'Remote'
})
#aab dataset pura clean hai 

df['salary_in_usd'] = df['salary_in_usd'].astype(int)
print(df.head())
print(df.info())

# 1.country wise avg salary 

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import pycountry

country_salary = df.groupby("employee_residence")["salary_in_usd"].mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(12,6))
bars = sns.barplot(x=country_salary.index, y=country_salary.values, 
                   palette=["#EFFD29","#6B5B95","#88B04B","#FFA500",
                            "#009B77","#D65076","#45B8AC","#EFC050",
                            "#5B5EA6","#EA455E"])

# Add values on top of bars
for i, v in enumerate(country_salary.values):
    plt.text(i, v+2000, f"${int(v):,}", ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title("Top 10 Countries by Avg Salary (USD)", fontsize=14, fontweight='bold', color="darkblue")
plt.xlabel("Country", fontsize=12)
plt.ylabel("Avg Salary (USD)", fontsize=12)
plt.xticks(rotation=60, fontsize=12,)

# Format y-axis as USD
plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))

# Light grid
plt.grid(axis='y', linestyle='--', alpha=0.1)
plt.tight_layout()
plt.savefig("country_salary.png", dpi=300, bbox_inches="tight")  
plt.show()


# 2.Job Title-wise Average Salary

job_salary = (
    df.groupby('job_title')['salary_in_usd']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

print(job_salary)

plt.figure(figsize=(12,6))
top_jobs = df.groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False).head(10)

custom_colors = ["#f3ff0b", "#1aed56", "#ef913f", "#eb2626", 
                 "#4438e8", "#cb13e7", "#e377c2", "#7f7f7f", 
                 "#bcbd22", "#10d2e8"]  

bars = plt.barh(job_salary.index, job_salary.values)

# ab labels 
# Add values INSIDE the bars (white text)
for bar in bars:
    plt.text(
        bar.get_width() - 40000,   # thoda andar likhna
        bar.get_y() + bar.get_height()/2,
        f"${int(bar.get_width()):,}",  
        va='center', ha='right', fontsize=10, color="white", fontweight="bold"
    )
plt.gca().xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${int(x/1000)}K'))
sns.barplot(
    
    x=top_jobs.values,
    y=top_jobs.index,
     palette=custom_colors
  
)

plt.title("Top 10 Job Titles by Avg Salary (USD)", fontsize=16, fontweight="bold", color="darkgreen")
plt.xlabel("Average Salary (USD)", fontsize=13)
plt.ylabel("Job Title", fontsize=13)

plt.yticks(fontsize=12)   #  Job titles ke font size 

plt.tight_layout()        #  Auto adjust karega spacing


plt.grid(axis='both', linestyle='--', alpha=0.1)
plt.tight_layout()
plt.savefig("job_salary.png", dpi=300, bbox_inches="tight")
plt.show()


#3.Experience Level-wise Salary Distribution

import seaborn as sns   

plt.figure(figsize=(10,6))

sns.set_style("white")  

sns.boxplot(
    x='experience_level', 
    y='salary_in_usd', 
    data=df, 
    palette="coolwarm", 
    width=0.5,
    linewidth=2,
    showfliers=False     
    
)

plt.title(" Salary Distribution by Experience Level", fontsize=16, fontweight='bold', color="navy")
plt.ylabel("Salary (USD)", fontsize=12, fontweight='bold')
plt.xlabel("Experience Level", fontsize=12, fontweight='bold')

# Y-axis me $ formatting
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
plt.gca().yaxis.set_major_formatter(tick)

plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.savefig("experience_salary.png", dpi=300, bbox_inches="tight")
plt.show()

#4.Correlation + Heatmap

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df_encoded = df.copy()

# Select categorical columns automatically
categorical_cols = df_encoded.select_dtypes(include=['object']).columns

# Encode categorical columns
le = LabelEncoder()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

corr = df_encoded.corr()

# Unique colormap
plt.figure(figsize=(12,7))
sns.heatmap(corr, annot=True, cmap="magma", linewidths=0.5, cbar_kws={'shrink': 0.8})

plt.title("Unique Correlation Heatmap", fontsize=14, fontweight='bold', color="darkred")

# Rotate x-axis labels
plt.xticks(rotation=45, ha="right")  
plt.yticks(rotation=0) 

plt.tight_layout() 
plt.savefig("correlation_heatmap.png", dpi=300, bbox_inches="tight")
plt.show()



# 5.Multi-variable Regression (Salary Prediction)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# Encode categorical columns
df_encoded = df.copy()
categorical_cols = df_encoded.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Features & Target
X = df_encoded.drop("salary_in_usd", axis=1)
y = df_encoded["salary_in_usd"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Scatter plot with gradient color
import matplotlib.ticker as mtick

plt.figure(figsize=(8,6))

# Scatter with clearer colors
scatter = plt.scatter(
    y_test, y_pred, 
    c=y_test, 
    cmap="plasma",   
    alpha=0.8, 
    edgecolor="black", 
    s=70
)

# Trend line
z = np.polyfit(y_test, y_pred, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), color="darkorange", linewidth=2.5, label="Trend Line")

# Titles
plt.title("Actual vs Predicted Salary (Regression Model)", fontsize=15, fontweight='bold', color="navy")
plt.xlabel("Actual Salary (USD)", fontsize=12, fontweight="bold")
plt.ylabel("Predicted Salary (USD)", fontsize=12, fontweight="bold")

# Dollar formatting for axes
fmt = '${x:,.0f}'
tick = mtick.StrMethodFormatter(fmt)
plt.gca().xaxis.set_major_formatter(tick)
plt.gca().yaxis.set_major_formatter(tick)

# Colorbar with $ formatting
cbar = plt.colorbar(scatter)
cbar.set_label("Actual Salary (USD)")
cbar.ax.yaxis.set_major_formatter(tick)   #  Dollar format on colorbar

# Grid + Legend
plt.grid(alpha=0.3, linestyle="--")
plt.legend()

plt.tight_layout()
plt.savefig("regression.png", dpi=300, bbox_inches="tight")
plt.show()


#  Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
print(f"R² Score: {r2:.2f}")
