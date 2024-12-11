import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

data_path = "C:/Users/Hp/OneDrive/Masaüstü/ai2/Car_Insurance_Claim.csv"
data_frame = pd.read_csv(data_path)

print(data_frame.head(500))
print(data_frame['RACE'].unique())

# Missing data analysis
print(data_frame.isnull())
print(data_frame.isnull().sum())

# Calculate missing value ratios
missing_percentage = data_frame.isnull().sum() / len(data_frame) * 100
print("Eksik Veri Oranları (%):\n", missing_percentage)

# Fill in missing values
data_frame['CREDIT_SCORE'] = data_frame['CREDIT_SCORE'].fillna(data_frame['CREDIT_SCORE'].mean())
data_frame['ANNUAL_MILEAGE'] = data_frame['ANNUAL_MILEAGE'].fillna(data_frame['ANNUAL_MILEAGE'].mean())
print("Eksik değerler dolduruldu:\n", data_frame.isnull().sum())

# Check data types
print(data_frame.dtypes)

# Convert categorical columns to dummy variables
data_frame = pd.get_dummies(data_frame, columns=['GENDER', 'VEHICLE_TYPE'], drop_first=True)
data_frame.rename(columns={'VEHICLE_TYPE_sports car': 'vehicle_sports_car'}, inplace=True)

print(data_frame.head())  # Dummy transformed data
print(data_frame.columns)
print(data_frame.dtypes)

# Relationships between independent variables
sns.pairplot(data_frame[['OUTCOME', 'CREDIT_SCORE', 'ANNUAL_MILEAGE']])
plt.show()

#Splitting into training and test data
train, test = train_test_split(data_frame, test_size=0.2, random_state=42)

#Poisson regression model
formula = 'OUTCOME ~ SPEEDING_VIOLATIONS + DUIS + PAST_ACCIDENTS + CREDIT_SCORE + VEHICLE_OWNERSHIP + ANNUAL_MILEAGE + GENDER_male + vehicle_sports_car'

# The model is set up with training data
model_train = smf.poisson(formula=formula, data=train)
results_train = model_train.fit()

print(results_train.summary())

#Pseudo R² calculation
null_model = smf.poisson('OUTCOME ~ 1', data=train).fit()  # Sadece sabit bir model
pseudo_r2 = 1 - (results_train.llf / null_model.llf)
print("Pseudo R²:", pseudo_r2)

#Predictions
test['predicted_OUTCOME'] = results_train.predict(test)

# Distribution of residues
residuals = test['OUTCOME'] - test['predicted_OUTCOME']
sns.histplot(residuals, kde=True)
plt.title('Distribution of residues')
plt.xlabel('Residual Values')
plt.ylabel('Frequence')
plt.show()

#Comparing actual values ​​with estimates
plt.scatter(test['OUTCOME'], test['predicted_OUTCOME'])
plt.xlabel('Real OUTCOME')
plt.ylabel('Estimated OUTCOME')
plt.title('Actual Values vs. Estimates')
plt.show()

#Visualizing coefficients
coefficients = results_train.params
sns.barplot(x=coefficients.index, y=coefficients.values)
plt.xticks(rotation=45)
plt.xlabel('Variables')
plt.ylabel('Coefficient Values')
plt.title('Poisson Regression Coefficients')
plt.show()

#Performance metrics
mse = mean_squared_error(test['OUTCOME'], test['predicted_OUTCOME'])
mae = mean_absolute_error(test['OUTCOME'], test['predicted_OUTCOME'])
print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)

#Deviance calculation
deviance = 2 * (null_model.llf - results_train.llf)
degrees_of_freedom = results_train.df_resid
print("Deviance:", deviance)
print("Degrees of Freedom:", degrees_of_freedom)

#Overdispersion control
if deviance > degrees_of_freedom:
    print("Overdispersion was detected in the model.")
else:
    print("No excessive scattering was detected in the model.")