# cmpsci4200-project1-part2

#part 1
1.the main target is the target column, which is whether or not a person is risky.
important statistics are, gender, academic level, family type, owning property, income, and housing situation.
These categories together should hopefully give the machine learning algorithm  enough data to predict if someone is risky or not.

2. There is no missing data in this dataset.

3. All but two of the non numeric columns should be kept for model training. These two dropped columns are income type and occupation. 

4. non-numeric columns in the dataset are family statues, housing situation, and education.
family and housing situation should be one hot encoded
education should be label encoded.

5. The three aggregate columns made are 
df['Income by Gender'] = df.groupby('Gender')['Total_income'].transform('mean')
df['Income by Education'] = df.groupby('Education_type')['Total_income'].transform('mean')
df['Income by Income Type'] = df.groupby('Income_type')['Total_income'].transform('mean')

6. All of the non-numeric columns being kept provide some useful information and have a correlation with the target, such as a person being single seems to directly correlate with a person being risky, or how academics make more money then those who only completed lower levels of education but where far likelier to be less risky then say those who only completed secondary education.  than others. The two non-numeric columns being dropped are income type and occupation, the reasons being that occupation is very specific and doesn’t have a strong direct connection with the target, and income type is being dropped since we already have the person's income and a bunch of other categorical information about the person, that and over half were simply classified as working which isn’t very descriptive or helpful in figuring out useful information about the person..

As for the encoding, education should be labeled encoded since finishing highschool is clearly a better education then not finishing highschool. As for the other two, it can be difficult to say whether being single or being married is strictly better or worse along with living in housing vs an apartment, since that's more subjective than anything else, so I chose to one hot encode these columns. 

7.i split my dataframe in all of the remaining columns in x, anf the target as y.
8. Using the standard scalier did not change the values of score when testing the two machine learning algorithms, so I simply continued without scaling. 
9.  Shown in the python file.

#part 2
see python code

#part 3
1-3.
for radom state = 1
For the linear regression model
so at 10% for test size, we are getting a score of 0.011167409447384036
at .30  for test size,we are getting a score of 0.004053267605449329, 
At.60  for test size, we are getting a score of 0.005941517268295282
at .90  for test size, we are getting a score of -0.0273504576528156.


for the decision tree.
at 10%  for test size, we are getting a score oft 0.7044284243048403
at 30%  for test size, we are getting a score of 0.7531754205286646
at 60%  for test size, we are getting a score of 0.7619292825266049
at 90%  for test size, we are getting a score of at 0.7622153564481062



for radom state = 15

For the linear regression model
so at 10%  for test size, we are getting a score of 0.016097126935587514
at 30% for test size, we are getting a score of 0.008945842539672011, 
at 60% for test size, we are getting a score of 0.00846877731466722
at 90% for test size, we are getting a score off -0.06644186334663171 


For the decision tree
at 10% for test size, we are getting a score of 0.7713697219361483
at 30% for test size, we are getting a score of 0.7662203913491246
at 60% for test size, we are getting a score of 0.7734294541709578
at 90% for test size, we are getting a score of 0.746538505549834





for radom state = 30

For the linear regression model
so at 10% for test size, we are getting a score of-0.012793953061021712
at 30% for test size, we are getting a score of 0.0013214506726588748
at 60% for test size, we are getting a score of 0.004985837668201931
at  90% for test size, we are getting a score of -0.021151903018850504 


For the decision tree
at 10% for test size, we are getting a score of 0.7631307929969104
at 30% for test size, we are getting a score of 0.7638173704085136
at 60% for test size, we are getting a score of 0.7591829728801922
at 90% for test size, we are getting a score of at 0.7635885112713126
#plot for linier regression at random state 1
![](/linierregvstestinc.PNG)

#plot for linier regression at random state 15
![](/linierregvstestinc15.PNG)

#plot for linier regression at random state 30
![](/linierregvstestinc30.PNG)

#plot for decision tree at random state 1
![](/treevstestinc.PNG)

#plot for decision tree at random state 1
![](/treevstestinc15.PNG)

#plot for decision tree at random state 1
![](/treevstestinc30.PNG)


5.
![](/2.PNG)
looking at the decision tree visualization, we can see a small sample of how it works. It uses a rule for each node to split the samples depending on if the value is true or false, true it goes left and false it goes right. The next thing to look at is the fill, the darker the fill, the closer the model is to an answer with that box. boxes that are very dark with no new branches leaf nodes are at a point where the model has found an answer. white nodes means the samples are split between the targets of the model. This particular tree only goes to a depth of 5, since the whole plot for this model is hundreds of nodes large, with a lot of depth. So it may need a lot of decisions and rules, it does manage to achieve a decent amount of accuracy.

6. I would make the argument that the decision tree was the better of the two models for this project. It manages to achieve a reasonable degree of accuracy of around 70-75%, however it would seem that the model suffers a bit from overfitting, as often increasing the test data leads to an increase of accuracy, not much and not every time, but it is definitely there.  The Other reason why is that the regression model is so inaccurate that even if the decision tree was half as  accurate as it was in its current state, it would still be more than the regression model's best prediction of 1.6%.

