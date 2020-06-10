# Training, K Fold Validation and Prediction on classifying Meal and NoMeal Data

## Project Plan
1.	Data Preprocessing and filling out the missing data.
2. 	Extract features from Meal and No Meal data
3.	Train the Support Vector Nachine model on the data.
4. 	K-Fold Cross Validatate the training data for evaluation of the model.
5. 	Testing using Test data for Prediction (Calculation of Accuracy, F1 score, Precision and Recall).


## Input
- Meal Data (mealData**X**.csv) and No Meal Data (Nomeal**X**.csv) of 5 subjects.
- Ground truth labels of Meal and No Meal data for 5 subjects which I assumed to be **1 for Meal** and **0 for No Meal** respectively.
- The input data can be found in **"[data](data)"**


## Code Execution
Open CMD/Terminal and navigate to the location of the file


#### Command Line/ Terminal
```
python test.py -f <<FILENAME>>
```


#### Example
```
python test.py -f E:\PythonProjects\DM2\test.csv
python test.py -f test.csv
```


## Output


#### train.py
```
C:\Users\anish\Desktop\MealNoMeal> python train.py
----K-FOLD CROSS VALIDATION----
----SUPPORT VECTOR MACHINE----
Accuracy :  64.19753086419753
Report :                precision    recall  f1-score   support

           0       0.63      0.77      0.69        43
           1       0.66      0.50      0.57        38

    accuracy                           0.64        81
   macro avg       0.64      0.63      0.63        81
weighted avg       0.64      0.64      0.63        81

----SUPPORT VECTOR MACHINE----
Accuracy :  65.4320987654321
Report :                precision    recall  f1-score   support

           0       0.65      0.77      0.71        44
           1       0.66      0.51      0.58        37

    accuracy                           0.65        81
   macro avg       0.65      0.64      0.64        81
weighted avg       0.65      0.65      0.65        81

----SUPPORT VECTOR MACHINE----
Accuracy :  69.1358024691358
Report :                precision    recall  f1-score   support

           0       0.64      0.69      0.67        36
           1       0.74      0.69      0.71        45

    accuracy                           0.69        81
   macro avg       0.69      0.69      0.69        81
weighted avg       0.69      0.69      0.69        81

----SUPPORT VECTOR MACHINE----
Accuracy :  62.5
Report :                precision    recall  f1-score   support

           0       0.64      0.61      0.62        41
           1       0.61      0.64      0.62        39

    accuracy                           0.62        80
   macro avg       0.63      0.63      0.62        80
weighted avg       0.63      0.62      0.62        80

----SUPPORT VECTOR MACHINE----
Accuracy :  60.0
Report :                precision    recall  f1-score   support

           0       0.54      0.69      0.61        36
           1       0.68      0.52      0.59        44

    accuracy                           0.60        80
   macro avg       0.61      0.61      0.60        80
weighted avg       0.62      0.60      0.60        80

----SUPPORT VECTOR MACHINE----
Accuracy :  63.74999999999999
Report :                precision    recall  f1-score   support

           0       0.62      0.73      0.67        41
           1       0.66      0.54      0.59        39

    accuracy                           0.64        80
   macro avg       0.64      0.64      0.63        80
weighted avg       0.64      0.64      0.63        80
```

#### test.py
```

PS C:\Users\anish\Desktop\MealNoMeal> python test.py -f test.csv
-------------------------------PREDICTED OUTPUT-------------------------------
[1 1 1 1 0 1 1 1 0 1 1 1 0 1 1 1 0 1 1 1 1 1 0 0 1 1 1 1 0 1 1 0 1 0 1 1 1
 0 1 1 1 0 1 1 0 1 1]

-------------------------------------------------------------------------------------------
|| DISCLAIMER: Accuracy is shown assuming that all of the data belongs to the MEAL Class ||

-------------------------------------------------------------------------------------------------------------------
|||| Accuracy: 74.46808510638297 || Precision: 74.46808510638297 || Recall:100.0 || F1-Score:85.36585365853657 ||||
-------------------------------------------------------------------------------------------------------------------
```

## Tested Running Environment
- **OS:** Windows 10
- **Python:** 3.7
