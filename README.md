# ft_linear_regression
This project will be your first steps into AI and Machine Learning.
You're going to start with a simple, basic machine learning algorithm.
You will have to create a program that predicts the price of a car by using a linear function train with a gradient descent algorithm.

`You will implement a simple linear regression with a single feature - in this case, the mileage of the car`  <br />
`To do so, you need to create two programs :`  <br />
* `The first program will be used to predict the price of a car for a given mileage.` <br />
`When you launch the program, it should prompt you for a mileage, and then give`  <br />
`you back the estimated price for that mileage. The program will use the following`  <br />
`hypothesis to predict the price : ` <br />
`estimateP rice(mileage) = θ0 + (θ1 ∗ mileage) ` <br /> 
`Before the run of the training program, theta0 and theta1 will be set to 0. ` <br />
* `The second program will be used to train your model. It will read your dataset`  <br />
`file and perform a linear regression on the data. ` <br />
`Once the linear regression has completed, you will save the variables theta0 and`  <br />
`theta1 for use in the first program. ` <br />
`You will be using the following formulas : ` <br />
tmpθ0 = learningRate ∗
1
m
mX−1
i=0 
(estimate Price(mileage[i]) − price[i]) <br/>

tmpθ1 = learningRate ∗
1
m
mX−1
i=0 <br/>

(estimateP rice(mileage[i]) − price[i]) ∗ mileage[i]
I let you guess what m is :) <br />
`Note that the estimatePrice is the same as in our first program, but here it uses` <br />
`your temporary, lastly computed theta0 and theta1.` <br />
`Also, don’t forget to simultaneously update theta0 and theta1` <br />

