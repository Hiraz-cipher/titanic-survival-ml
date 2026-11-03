import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Step 1: Load data
df = pd.read_csv('Titanic-Dataset.csv')
print("Data loaded!")

# Step 2: Identify features and labels
features = ['Pclass', 'Sex', 'Age', 'Embarked']
y = df['Survived']

# Step 3: Mapping
# Handle missing values and mapping
df_clean = df.copy()
df_clean['Age'] = df_clean['Age'].fillna(df_clean['Age'].median())
# Map Sex: male=0, female=1
df_clean['Sex'] = df_clean['Sex'].map({'male': 0, 'female': 1})
# Handle missing Embarked values and map: S=0, C=1, Q=2
df_clean['Embarked'] = df_clean['Embarked'].fillna('S')  # Most common port
df_clean['Embarked'] = df_clean['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X = df_clean[features]

print("Features prepared!")

# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Build classifier
dtree = DecisionTreeClassifier()

# Step 6: Model fitting
dtree.fit(X_train, y_train)
print("Model trained!")

# Step 7: Predictions
predictions = dtree.predict(X_test)

# Step 8: Check accuracy and other metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)

print(f"Accuracy:  {accuracy:.4f} ({accuracy:.2%})")
print(f"Precision: {precision:.4f} ({precision:.2%})")
print(f"Recall:    {recall:.4f} ({recall:.2%})")


# Interactive Prediction Function
def predict_survival():
    print("\n" + "=" * 50)
    print("TITANIC SURVIVAL PREDICTOR")
    print("=" * 50)

    while True:
        try:
            # Get user input
            print("\nEnter passenger details:")

            # Passenger Class
            print("Passenger Class:")
            print("1 = First Class (Upper)")
            print("2 = Second Class (Middle)")
            print("3 = Third Class (Lower)")
            pclass = int(input("Enter class (1, 2, or 3): "))

            if pclass not in [1, 2, 3]:
                print("Invalid class! Please enter 1, 2, or 3")
                continue

            # Gender
            print("\nGender:")
            gender_input = input("Enter gender (male/female): ").lower().strip()

            if gender_input not in ['male', 'female']:
                print("Invalid gender! Please enter 'male' or 'female'")
                continue

            # Map gender to number
            gender = 0 if gender_input == 'male' else 1

            # Age
            age = float(input("\nEnter age: "))

            if age < 0 or age > 120:
                print("Invalid age! Please enter a realistic age")
                continue

            # Embarked Port
            print("\nPort of Embarkation:")
            print("S = Southampton (England)")
            print("C = Cherbourg (France)")
            print("Q = Queenstown (Ireland)")
            embarked_input = input("Enter port (S, C, or Q): ").upper().strip()

            if embarked_input not in ['S', 'C', 'Q']:
                print("Invalid port! Please enter S, C, or Q")
                continue

            # Map embarked to number
            embarked = {'S': 0, 'C': 1, 'Q': 2}[embarked_input]

            # Create prediction data
            passenger_data = pd.DataFrame([[pclass, gender, age, embarked]],
                                          columns=['Pclass', 'Sex', 'Age', 'Embarked'])

            # Make prediction
            prediction = dtree.predict(passenger_data)[0]
            probability = dtree.predict_proba(passenger_data)[0]

            # Display results
            print("\n" + "-" * 40)
            print("PREDICTION RESULTS")
            print("-" * 40)
            print(f"Passenger Details:")
            print(f"  Class: {pclass} ({'First' if pclass == 1 else 'Second' if pclass == 2 else 'Third'})")
            print(f"  Gender: {gender_input.title()}")
            print(f"  Age: {age}")
            print(f"  Embarked: {embarked_input} ({'Southampton' if embarked_input == 'S' else 'Cherbourg' if embarked_input == 'C' else 'Queenstown'})")
            print()

            if prediction == 1:
                print("🎉 PREDICTION: This passenger would have SURVIVED!")
                print(f"   Survival probability: {probability[1]:.1%}")
            else:
                print("💔 PREDICTION: This passenger would NOT have survived")
                print(f"   Death probability: {probability[0]:.1%}")

            print("-" * 40)

            # Ask if user wants to try again
            again = input("\nWould you like to predict for another passenger? (yes/no): ").lower().strip()
            if again not in ['yes', 'y']:
                break

        except ValueError:
            print("Invalid input! Please enter numbers for class and age.")
        except Exception as e:
            print(f"An error occurred: {e}")

    print("\nThank you for using the Titanic Survival Predictor!")


# Run the interactive predictor
predict_survival()