# Training, K Fold Validation and Prediction on classifying Meal and NoMeal Data


## Input
- Meal Data (mealData**X**.csv) and No Meal Data (Nomeal**X**.csv) of 5 subjects.
- Ground truth labels of Meal and No Meal data for 5 subjects which I assumed to be **1 for Meal** and **0 for No Meal** respectively.


## Project Plan
1.	Data Preprocessing and filling out the missing data.
2. 	Extract features from Meal and No Meal data
3.	Train the Support Vector Nachine model on the data.
4. 	K-Fold Cross Validatate the training data for evaluation of the model.
5. 	Testing using Test data for Prediction (Calculation of Accuracy, F1 score, Precision and Recall).


## Code Execution
Open CMD and navigate to the location of the file

#### Command Line
```
python test.py -f <<FILENAME>>
```
#### Example
```
python test.py -f E:\PythonProjects\DM2\test.csv
python test.py -f test.csv
```

