import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('Titanic-Dataset.csv')
print(df)

# Check for missing values
print(df.isnull().sum())

# Drop columns with high missing value percentages
df.drop(['Cabin'], axis=1, inplace=True)

# Fill missing values in 'Age' column with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing values in 'Embarked' column with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop rows with missing values in 'Fare' column
df.dropna(subset=['Fare'], inplace=True)

# Encode categorical variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Display the cleaned dataset
df.head()

# Information about the dataset
df.info()

# Convert the 'Sex_male' to int and add it to the table as 'Sex
df['Sex'] = df['Sex_male'].astype(int)

# Print the types of the Columns(eg.Int,float.Obj)
print(df.dtypes)

# Remove 'Sex_male' as we have now 'Sex' in int
df.drop('Sex_male', axis=1)

# Check number of unique elements in each column
print(df.nunique())

# Drop Ticket as it has too many values
df = df.drop(['Ticket'], axis=1)

plt.figure(figsize=(5, 5))
sns.histplot(data=df, x='Age', bins=20, kde=True)
plt.title('Distribution of Passenger Ages')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Pclass', hue='Survived')
plt.title('Survival Count by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()

# Define X and y to train the data
X = df.drop(['Survived', 'Name'], axis=1)
y = df[['Survived']]

print(X)

print(y)

# Split the data into training, validation, and test sets
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Choose a machine learning algorithm
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_pred = model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)


# Evaluate the model using classification report
print("Classification Report:")
print(classification_report(y_val, y_pred))

y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)