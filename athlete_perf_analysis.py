import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load CSV
file_path = './athlete_stress_recovery_data.csv'  # Update with your file path
df = pd.read_csv(file_path)

# Display basic info
print("\n--- Dataset Info ---")
print(df.info())

print("\n--- First 5 Rows ---")
print(df.head())

# Check for missing values
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Descriptive stats
print("\n--- Descriptive Statistics ---")
print(df.describe())

# -------------------------
# Visualizations
# -------------------------

# 1. Distribution of Performance Score
plt.figure(figsize=(8, 5))
sns.histplot(df['Performance_Score'], kde=True, bins=30)
plt.title('Distribution of Performance Score')
plt.xlabel('Performance Score')
plt.ylabel('Frequency')
plt.show()

# 2. Training Hours vs Performance
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Training_Hours_per_Week', y='Performance_Score', hue='Gender')
plt.title('Training Hours vs Performance Score')
plt.show()

# 3. HRV vs Stress Score
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='HRV', y='Perceived_Stress_Score', hue='Gender')
plt.title('HRV vs Perceived Stress Score')
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(12, 8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# -------------------------
# Actionable Insights
# -------------------------

print("\n--- Actionable Insights ---")

# 1. Performance vs Sleep
if corr.loc['Sleep_Hours', 'Performance_Score'] > 0.3:
    print("✅ Athletes with more sleep hours tend to have better performance. Encourage 7–9 hours of sleep.")

# 2. Training vs Performance
if corr.loc['Training_Hours_per_Week', 'Performance_Score'] > 0.3:
    print("✅ More training hours are generally associated with better performance. Monitor for overtraining.")

# 3. High Stress Impact
if corr.loc['Perceived_Stress_Score', 'Performance_Score'] < -0.3:
    print("⚠️ High perceived stress correlates with lower performance. Recommend stress reduction strategies.")

# 4. HRV and Injury Risk
if corr.loc['HRV', 'Injury_Risk_Score'] < -0.3:
    print("⚠️ Lower HRV is associated with higher injury risk. Use HRV as an early warning metric.")

# 5. Resting Heart Rate
if corr.loc['Resting_Heart_Rate', 'Injury_Risk_Score'] > 0.3:
    print("⚠️ Higher resting heart rate may indicate higher injury risk. Monitor recovery readiness.")

# Gender-based Training Insights
avg_training_by_gender = df.groupby("Gender")["Training_Hours_per_Week"].mean()
print("\nAverage Training Hours per Gender:\n", avg_training_by_gender)

# Sport-based Performance
top_sports = df.groupby("Sport")["Performance_Score"].mean().sort_values(ascending=False).head()
print("\nTop Performing Sports by Average Score:\n", top_sports)

# Key metrics for pairwise analysis
key_metrics = [
    'Training_Hours_per_Week',
    'Resting_Heart_Rate',
    'HRV',
    'Perceived_Stress_Score',
    'Sleep_Hours',
    'Performance_Score',
    'Injury_Risk_Score'
]

# Ensure there are no NaNs in selected columns
pairplot_df = df[key_metrics + ['Gender']].dropna()

# Set a modern style
sns.set(style="whitegrid", font_scale=1.1)

# Create pairplot
g = sns.pairplot(
    pairplot_df,
    hue='Gender',
    diag_kind='kde',
    corner=True,
    plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'k'},
    height=2.5
)

# Add a clear title
g.fig.suptitle("Pairwise Relationships Among Key Athlete Metrics", y=1.02)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Gender', y='Perceived_Stress_Score')
plt.title('Perceived Stress Score by Gender')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x='Sport', y='Performance_Score')
plt.xticks(rotation=45)
plt.title('Performance Score by Sport')
plt.show()

bins = [0, 33, 66, 100]
labels = ['Low Risk', 'Moderate Risk', 'High Risk']
df['Injury_Risk_Level'] = pd.cut(df['Injury_Risk_Score'], bins=bins, labels=labels)

plt.figure(figsize=(7, 5))
sns.countplot(data=df, x='Injury_Risk_Level', palette='Set2')
plt.title('Injury Risk Distribution')
plt.xlabel('Risk Level')
plt.ylabel('Number of Athletes')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x='Sleep_Hours',
    y='Perceived_Stress_Score',
    size='Performance_Score',
    hue='Gender',
    palette='coolwarm',
    sizes=(20, 300),
    alpha=0.6,
    edgecolor='k'
)
plt.title('Stress vs Sleep Colored by Gender and Scaled by Performance')
plt.xlabel('Sleep Hours')
plt.ylabel('Perceived Stress Score')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2)
plt.show()

high_perf = df[df['Performance_Score'] > df['Performance_Score'].quantile(0.75)]
print("Average stats of top performers:")
print(high_perf.describe())

df['Sleep_Bucket'] = pd.cut(df['Sleep_Hours'], bins=[0, 5, 7, 9, 12], labels=['<5', '5-7', '7-9', '9+'])
sns.boxplot(data=df, x='Sleep_Bucket', y='Performance_Score', palette='viridis')
plt.title("Performance by Sleep Duration")
plt.show()