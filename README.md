💼 Data Science Job Salaries – Analysis Project

This project explores the global salaries in Data Science and related roles. Using a dataset of salaries across different countries, job titles, experience levels, and employment types, I performed data cleaning, exploratory data analysis (EDA), and applied a simple regression model to understand salary predictors.

The goal was to identify key trends in salaries and present the findings through clean and attractive visualizations.

⚡ Project Workflow

Data Cleaning

Converted abbreviations into readable values (e.g., EN → Entry, FT → Full-time).

Standardized salary column into USD.

Made remote ratio more interpretable (0 → Onsite, 50 → Hybrid, 100 → Remote).

Exploratory Data Analysis (EDA)

Country-wise salary comparison

Job title-wise average salary

Salary distribution by experience level

Correlation heatmap of key factors

Regression Model

Built a simple Linear Regression model to predict salaries.

Evaluated using Mean Absolute Error (MAE) and R² Score.

Visualized predicted vs actual salaries.

📊 Key Insights from the Data

🌍 Country-wise Salaries

Malaysia 🇲🇾 has the highest average salary.

Puerto Rico 🇵🇷 and the US 🇺🇸 also show strong pay scales.

Switzerland 🇨🇭 and New Zealand 🇳🇿 remain competitive but below the top.

👩‍💻 Job Roles

Senior leadership roles (Data Analytics Lead, Principal Data Engineer) dominate the salary charts.

Specialized roles like Financial Data Analyst and Principal Data Scientist also pay exceptionally well.

Analysts and entry-level roles remain on the lower end of the spectrum.

📈 Experience Levels

Executives earn nearly 200K USD on average.

Senior roles earn about 138K USD, nearly 2x more than Mid-level employees.

Entry-level roles average around 61K USD, showing strong salary growth potential.

🔗 Correlation Factors

Salary is most strongly dependent on experience level and job role.

Country and work location type (onsite/remote) also influence pay.

🤖 Regression Model

Regression confirms experience level is the strongest salary predictor.

Country and job role add important context to explain salary variations.

🛠️ Tools & Technologies

Python (Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn)

Jupyter Notebook / VS Code for analysis

Git & GitHub for version control

📌 How to Run the Project

Clone the repository:

git clone https://github.com/your-username/data-science-job-salaries.git
cd data-science-job-salaries


Install dependencies:

pip install -r requirements.txt


Run the analysis:

python analysis.py


Visualizations will be saved inside the figures/ folder.

✨ This project highlights how experience, job role, and geography shape salaries in the Data Science domain.
