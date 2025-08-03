import pandas as pd
import numpy as np

# Reproducibility
np.random.seed(42)

# Number of samples
default_n = 6705

departments = ['Sales', 'Engineering', 'HR', 'Marketing']
education = ['Bachelors', 'Masters', 'PhD']
gender = ['Male', 'Female']

# Generate features
years_experience = np.random.normal(5, 2, default_n).clip(0)
deps = np.random.choice(departments, default_n)
edus = np.random.choice(education, default_n, p=[0.6,0.3,0.1])
gends = np.random.choice(gender, default_n, p=[0.7,0.3])

# Simulate salary
base = 30000 + years_experience * 2000
edu_bonus = np.array([5000 if e=='Masters' else 10000 if e=='PhD' else 0 for e in edus])
dep_bonus = np.array([3000 if d=='Engineering' else 0 for d in deps])
salary = base + edu_bonus + dep_bonus + np.random.normal(0, 2000, default_n)

# Build DF
df = pd.DataFrame({
    'YearsExperience': years_experience.round(1),
    'Department': deps,
    'Education': edus,
    'Gender': gends,
    'Salary': salary.round(2)
})

# Save CSV
import os
os.makedirs('data/raw', exist_ok=True)
df.to_csv('data/raw/salary_data.csv', index=False)
print(f"Generated data/raw/salary_data.csv with {len(df)} records")
