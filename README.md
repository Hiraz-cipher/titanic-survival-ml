# titanic-survival-ml
Comparing a decision tree and KNN model to predict Titanic passenger survival (Python, scikit-learn)

# Titanic Survival Prediction: Decision Tree vs KNN

A machine learning comparison of two classification models that predict whether a
Titanic passenger survived, using passenger class, sex, age and port of embarkation.
Each script also includes an interactive predictor: enter a passenger's details and
get a prediction.

## Dataset
- Titanic passenger data: 891 passengers, 12 columns (`Titanic-Dataset.csv`)
- Source: [where you downloaded the dataset]
- Target: `Survived` (1 = survived, 0 = did not survive)
- Missing values: Age (177), Embarked (2), Cabin (687, not used)

## Preprocessing
- Age: missing values filled with the median age
- Sex: male = 0, female = 1
- Embarked: missing values filled with S (the most common port), then S = 0, C = 1, Q = 2
- Features used: Pclass, Sex, Age, Embarked
- Split: 80% training, 20% testing, with `random_state = 42`, so both models use the same split

## Models
- **Decision tree:** scikit-learn `DecisionTreeClassifier` with default settings
- **K-nearest neighbours:** `KNeighborsClassifier` with k = 5, features not scaled

## Results (test set)
| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Decision tree | 77.65% | 75.00% | 68.92% |
| KNN (k = 5) | 79.33% | 79.37% | 67.57% |

- **Accuracy:** share of all predictions that were correct
- **Precision:** of the passengers predicted to survive, the share that actually did
- **Recall:** of the passengers who actually survived, the share the model found

KNN is slightly more accurate and precise. The decision tree finds slightly more of the survivors.

## Files
- `decision_tree_titanic.py`: decision tree model and interactive predictor
- `knn_titanic.py`: KNN model and interactive predictor
- `Titanic-Dataset.csv`: passenger data (keep it in the same folder as the scripts)

## How to run
1. Install Python 3, then install the libraries: `pip install pandas scikit-learn`
2. Keep the CSV in the same folder as the scripts.
3. Run `python decision_tree_titanic.py` or `python knn_titanic.py`.
4. The model trains, prints the results, then asks for passenger details.
   Type `yes` to predict another passenger, or `no` to finish.

## Demo videos
- [Decision tree demo](https://onedrive.live.com/photos?photosData=%2Fshare%2F09848B11F3B69ABB%21sf6d35eb26dc04f5fb462c5f1663e5289%3Fithint%3Dvideo%26e%3DHh4EgM%26migratedtospo%3Dtrue&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3YvYy8wOTg0OGIxMWYzYjY5YWJiL0ViSmUwX2JBYlY5UHRHTEY4V1ktVW9rQlVHVGp5OHV5OWxEM0NJN2Vfb2g1Z0E%5FZT1IaDRFZ00)
- [KNN demo](https://onedrive.live.com/photos?photosData=%2Fshare%2F09848B11F3B69ABB%21s98c5c3d87d8745d3ab3654a96abc95ce%3Fithint%3Dvideo%26e%3DJKhtnz%26migratedtospo%3Dtrue&redeem=aHR0cHM6Ly8xZHJ2Lm1zL3YvYy8wOTg0OGIxMWYzYjY5YWJiL0VkakR4WmlIZmRORnF6WlVxV3E4bGM0Ql9aNk5iRmE0Uk1uVEdrRFUzN0dkeHc%5FZT1KS2h0bno)

## What I'd improve next
- Use more features (Fare, SibSp, Parch), which may improve accuracy
- Scale the features for KNN, so Age does not dominate the distance calculation
- Try different values of k, and use cross-validation instead of a single split
- Limit the decision tree depth to reduce overfitting
