### Project goals

- Use 3 similar cleaned datasets to find the features that best predict diabetes in patients. 


### Project description

- I've taken 2 datasets from kaggle that were cleaned for modeling to determine diabetic patients.  

### The Plan 

### Initial hypotheses

- What features will have the greatest weight on determining diabetic patients?
- Using the encoded target variable diabetit_012, how will each set perform on models and will this effect how future questionaires should be formatted?

### Acquire: 
I acquired 2 datasets from Kaggle as csv downloads and saved them in my local folder from this site:
* https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset/data?select=diabetes_012_health_indicators_BRFSS2015.csv


- diabetes _ 012 _ health _ indicators _ BRFSS2015.csv


- diabetes _ binary _ health _ indicators _ BRFSS2015.csv


### Prepare:
- Lowercased the column names
- Created a third dataframe with equal weight of the target variables on diabetes_012
    0=No Diabetes / 1=Prediabetic / 2=Diabetic
    - 13983 Rows / 22 Columns

### Explore: 
- Explored on the first data set
- How does the percentage of diabetics/prediabetics and the features appear in stacked percentages?
- What do the features look like based on Age?
- What features is diabetes_012 dependent on?

### Modeling: 
- Modeling was done for all 3 sets.
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
|PhysActivity| Adults who reported doing physical activity or exercise during the past 30 days other than their regular job, 0=No, 1=Yes|
|Fruits| Consume Fruit 1 or more times per day, 0=No, 1=Yes|
|Veggies| Consume Vegetables 1 or more times per day, 0=No, 1=Yes|
|HvyAlcoholConsump| Heavy drinkers (adult men having more than 14 drinks per week and adult women having more than 7 drinks per week), 0=No, 1=Yes|
|AnyHealthcare| Do you have any kind of health care coverage, including health insurance, prepaid plans such as HMOs, or government plans such as Medicare, or Indian Health Service? 0=No, 1=Yes|
|NoDocbcCost| Was there a time in the past 12 months when you needed to see a doctor but could not because of cost? 0=No, 1=Yes|
|GenHlth| Would you say that in general your health is: 1 to 5 scale, 1 is excellent, 5 is poor|
|MentHlth| Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good? Scale 1 - 30|
|PhysHlth| Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good? Scale 1 - 30|
|DiffWalk| Do you have serious difficulty walking or climbing stairs? 0=No, 1=Yes|
|Sex| Indicate sex of respondent. 0= Female, 1= Male |
|Age| Fourteen-level age category,1 is 18-24, all the way to 80 and Older, 5 yr increments |
|Education| What is the highest grade or year of school you completed? 1 - 6, 1 is never attended school or kindergarten, 6 is college 4 yrs or more|
|Income| Is your annual household income from all sources: (If respondent refuses at any income level, code "Refused.") 1-8, 1 is < 10k, 8 is > 75k|

### How to Reproduce
- Clone this repo
- Acquire data from Kaggle in link above in Acquire and save csv's to local folder.
- Run Notebook

### Key findings 
- It seems highbp, highchol, smoker, stroke, and heardiseaseorattack greatly increases the percentage of diabetics and prediabetic patients as they get older.
- Having a lower income/education level puts younger patients at higher risk of diabetes
- BMI visually shows as bmi increases, so does the amount of patients with diabetes.

### Takeaways and Conclusions
- Logistic Regression Models worked best on all 3 datasets
- Highbp, highchol, bmi, heartdiseaseorattack are big indicators for diabetes.
- It is very difficult to predict diabetes off of a questionairre, however, looking at the way the prediabetic patients answered their questions is a very good way to start predicting diabetes.

### Recommendations
- Have more questions tailored to finding more information of pre-diabetic patients.




