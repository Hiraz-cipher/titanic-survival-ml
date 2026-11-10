# Titanic Survival Prediction: Decision Tree vs KNN

A machine learning comparison of two classification models that predict whether a
Titanic passenger survived, using passenger class, sex, age and port of embarkation.
Each script also includes an interactive predictor: enter a passenger's details and
get a prediction.

## Dataset
- Titanic passenger data: 891 passengers, 12 columns (`Titanic-Dataset.csv`)
- Provenance: the original download URL was not recorded. The committed CSV uses the familiar Titanic passenger training-data structure; its exact download source has not been verified.
- [Dataset details and provenance limits](DATASET.md)
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

## Previously reported results (test set)
| Model | Accuracy | Precision | Recall |
|---|---|---|---|
| Decision tree | 77.65% | 75.00% | 68.92% |
| KNN (k = 5) | 79.33% | 79.37% | 67.57% |

- **Accuracy:** share of all predictions that were correct
- **Precision:** of the passengers predicted to survive, the share that actually did
- **Recall:** of the passengers who actually survived, the share the model found

These are the previously recorded project results, not a new benchmark. The decision tree has no fixed model seed, so its results can vary between runs. In this recorded run, KNN was slightly more accurate and precise, while the decision tree recovered slightly more survivors.

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

## Limitations and next steps

Age imputation is performed before the train/test split, which leaks information from the test set into preprocessing. A future comparison should fit preprocessing on training data only, using a pipeline. The current results should be presented as an exploratory baseline.
- Use more features (Fare, SibSp, Parch), which may improve accuracy
- Scale the features for KNN, so Age does not dominate the distance calculation
- Try different values of k, and use cross-validation instead of a single split
- Limit the decision tree depth to reduce overfitting
