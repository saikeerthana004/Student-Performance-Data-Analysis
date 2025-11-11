#Import Necessary Libraries
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

#DATASET
#1.Load the Dataset(.CSV file)
df = pd.read_csv("C:/Users/SAI KEERTHANA/.spyder-py3/Student_performance_dataset.csv")
# Print the DataFrame with all columns visible
print(df)
# Display the last 5 rows of the DataFrame
print(df.tail())
# Display a summary of the DataFrame
print(df.info())
#2.Missing values
# Check for missing values
print("Missing values before cleaning:\n")
print(df.isnull().sum())

#DATA VISUALIZATION
#Create histograms for numerical columns
df.hist(figsize=(20, 10), bins=7, color='lightblue')
# Show the plots
plt.show()

#Categorical and Numerical Features
# Get all columns in the DataFrame
columns = list(df.columns)
# Initialize lists for categorical and numerical columns
categoric_columns = []
numeric_columns = []
# Distinguish columns based on unique value counts
for col in columns:
    if df[col].dtype in ['int64', 'float64']:  # Check if column is numeric
        if len(df[col].unique()) > 5:
            numeric_columns.append(col)
        else:
            categoric_columns.append(col)
# Exclude the first column (assumed to be an ID)
numeric_columns = numeric_columns[1:]
# Print the results
print('Numerical features:', numeric_columns)
print('Categorical features:', categoric_columns)

#Correlation Among Features
plt.figure(figsize=(16, 8))
sns.heatmap(df.corr(), annot = True, cmap = "coolwarm")
plt.title('The correlation among features',y= 1.05)
plt.show()
# Sample feature importance scores (replace with your actual values)
feature_importance = {
    'Absences': 0.45,
    'StudyTimeWeekly': 0.22,
    'ParentalSupport': 0.12,
    'ParentalEducation': 0.10,
    'Ethnicity': 0.05,
    'Gender': 0.04,
    'Sports': 0.03,
    'Extracurricular': 0.02,
    'Tutoring': 0.01,
    'Music': 0.01,
    'Volunteering': 0.01
}
# Convert the dictionary to a DataFrame
feature_importance_df = pd.DataFrame.from_dict(feature_importance, orient='index', columns=['Importance'])
# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
# Create the bar plot
plt.figure(figsize=(15, 10))
sns.barplot(x='Importance', y=feature_importance_df.index, data=feature_importance_df, orient='h')
# Customize the plot
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()

#MATPLOTLIB AND SEABORN
#1.StudentID
student_ids = [1000, 1500, 2000, 2500, 3000, 3500]
sns.histplot(student_ids, kde=False, color='lightgreen')
plt.title("StudentID")
plt.grid(True)
plt.show()
#2.Age
data = {'Age': [15,16,17,18], 'Count': [600, 550, 540, 500]}
sns.barplot(x='Age', y='Count', data=data, color='lightblue')
plt.title('Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(True)
plt.show()
#3.Gender
data = {'Gender': [0, 1], 'Count': [900, 1100]}
sns.barplot(x='Gender', y='Count', data=data, color='lightpink')
plt.title('Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.grid(True)
plt.xticks([0, 1], ['Female', 'Male'])
plt.show()
#4.Ethnicity
data = {'Ethnicity': [0, 1, 2, 3], 'Count': [1200, 500, 500, 250]}
sns.barplot(x='Ethnicity', y='Count', data=data, color='purple')
plt.title('Ethnicity')
plt.xlabel('Ethnicity')
plt.ylabel('Count')
plt.grid(True)
plt.xticks([0, 1, 2, 3])
plt.show()
#5.Parential Education
data = {'Parental Education': [0, 1, 2, 3, 4], 'Count': [200, 750, 900, 350, 100]}
sns.barplot(x='Parental Education', y='Count', data=data, color='lightblue')
plt.title('Parental Education')
plt.xlabel('Parental Education')
plt.ylabel('Count')
plt.grid(True)
plt.show()
#6.StudyTimeWeekly
data = {'Study Time Weekly': [0, 5, 10, 15, 20], 'Count': [350, 300, 330, 320, 300]}
sns.barplot(x='Study Time Weekly', y='Count', data=data, color='Green')
plt.title('Study Time Weekly')
plt.xlabel('Study Time Weekly')
plt.ylabel('Count')
plt.grid(True)
plt.show()
#7.Absentees
data = {'Absences': [0, 5, 10, 15, 20, 25, 30], 'Count': [350, 350, 300, 350, 350, 350, 400]}
sns.barplot(x='Absences', y='Count', data=data, color='Red')
plt.title('Absences')
plt.xlabel('Absences')
plt.ylabel('Count')
plt.grid(True)
plt.show()
#8.ExtraCirricular
data = {'Extracurricular': [0, 1], 'Count': [1500, 1000]}
sns.barplot(x='Extracurricular', y='Count', data=data, color='Orange')
plt.title('Extracurricular')
plt.xlabel('Extracurricular')
plt.ylabel('Count')
plt.grid(True)
plt.xticks([0, 1])
plt.show()
#9.Sports
data ={'Sports': [0,1],
    'Value': [1500, 800]}
sns.set_style('whitegrid')
sns.barplot(x='Sports', y='Value', data=data, color='Blue', alpha=0.7)
plt.title('Sports')
plt.show()
#10.Music
data = pd.DataFrame({
    'X_values': [0, 1],
    'Y_values': [2000, 500]
})
sns.set_style('whitegrid')
sns.barplot(x='X_values', y='Y_values', data=data, color='Yellow', alpha=0.7)
plt.title('Music')
plt.xlabel('Music')
plt.ylabel('Values')
plt.show()
#11.GPA
# Generate some sample GPA data
gpa_data = np.random.normal(loc=2.5, scale=1, size=1000)
# Create the histogram using Seaborn
sns.histplot(gpa_data, bins=5, kde=False, color='lightgreen')
# Customize the plot
plt.title('GPA')
plt.xlabel('GPA')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
#12.Volunteering
volunteering_data = np.random.choice([0, 1], size=2500, p=[0.8, 0.2])
sns.histplot(volunteering_data, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], color='Blue', kde=False)
plt.title('Volunteering')
plt.xlabel('Volunteering (0=No, 1=Yes)')
plt.ylabel('Count')
plt.show()
#13.GradeClass
grade_data = np.random.randint(0, 5, 2500)
sns.histplot(grade_data, bins=[0, 1, 2, 3, 4], color='Violet', kde=False)
plt.title('GradeClass')
plt.xlabel('Grade')
plt.ylabel('Count')
plt.show()

#SHOW SUMMARY STATISTICS
# Read the dataset
df = pd.read_csv("C:/Users/SAI KEERTHANA/.spyder-py3/Student_performance_dataset.csv")

# Calculate summary statistics for numerical columns
numerical_columns = ['Age', 'StudyTimeWeekly', 'Absences', 'Sports', 'Music', 'Volunteering', 'GPA']
summary_stats = df[numerical_columns].describe()

# Print the summary statistics
print(summary_stats)

# Calculate summary statistics for categorical columns 
categorical_columns = ['Gender', 'Ethnicity', 'ParentalEducation', 'Extracurricular', 'GradeClass'] 
for column in categorical_columns: 
    print(f"\nSummary statistics for {column}:") 
    print(df[column].value_counts())
