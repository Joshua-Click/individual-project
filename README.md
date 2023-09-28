### Project goals

- Use 3 similar cleaned datasets to find the features that best predict diabetes in patients. 


### Project description

- I've taken 3 datasets from kaggle that were cleaned for modeling to determine diabetic patients.  

### The Plan 

### Initial hypotheses

- The dataset with 50/50 Non Diabetic/Diabetic patients will perform the best on modelsets
- What features will have the greatest weight on determining diabetic patients?
- Using the encoded target variable is_diabetic, how will each set perform on models and will this effect how future questionaires should be formatted?

### Acquire: 
- Acquire the data kaggle as 3 separate csv's
- 

### Explore: 
- 
- 
- 
- 

### Modeling: 
- Use drivers in explore to build predictive models of different types
- Evaluate models on train and validate data
- Select the best model based on accuracy
- Evaluate the test data


### Data dictionary:

| Feature | Definition |
|--------|-----------|
|Diabetes_012| Ever Told you have diabetes 0=No/OnlyDuringPrgnancy, 1=Prediabetic, 2=Diabetic|
|HighBP| Have you ever been told you have High Blood Pressure, 0=No, 1=Yes|
|HighChol| Have you ever been told you have had High Cholesterol, 0=No, 1=Yes|
|CholCheck| Cholesterol Check in last 5 years?, 0=No, 1=Yes|
|BMI| Body Mass Index Number|
|Smoker| Smoke at least 100 cigarettes,etc in your entire life?, 0=No, 1=Yes|
|Stroke| Ever had a stroke, 0=No, 1=Yes|
|HeartDiseaseorAttack| Respondents that have ever reported having coronary heart disease or myocardial infarction? 0=No, 1=Yes|
|PhysActivity| Adults who|
|Fruits| Last time the house was sold in 2017|
|Veggies| Last time the house was sold in 2017|
|HvyAlcoholConsump| Last time the house was sold in 2017|
|AnyHealthcare| Last time the house was sold in 2017|
|NoDocbcCost| Last time the house was sold in 2017|
|GenHlth| Last time the house was sold in 2017|
|MentHlth| Last time the house was sold in 2017|
|PhysHlth| Last time the house was sold in 2017|
|DiffWalk| Last time the house was sold in 2017|
|Sex| Last time the house was sold in 2017|
|Age| Last time the house was sold in 2017|
|Education| Last time the house was sold in 2017|
|Income| Last time the house was sold in 2017|

### How to Reproduce
- Clone this repo
- Acquire data from MySql (Should make a zillow.csv after)
- Run Notebook

### Key findings 
- Bedrooms, Bathrooms, and Finished_area were the best features to use. Garages did not make the cut due to too much missing data and too many assumptions.
- Model 4, Polynomial Regression with a degree of 3 worked best.

### Takeaways and Conclusions
- After running scaled data through the model
- Test data ran on Model 4
    - 39.7% accuracy overall
- Still quite low and only within 338k of the correct home value price which is alot of error.
### Recommendations
- Recommend splitting data into counties and adding more features to the data collected in order to potentially predict Home Values in the future.
- Simply having just bedrooms, bathrooms, and finished area are not enough to predict home values.