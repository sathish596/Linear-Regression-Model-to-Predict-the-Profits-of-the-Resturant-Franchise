# Linear-Regression-Model-to-Predict-the-Profits-of-the-Resturant-Franchise
I had collected data of  restaurant franchise and are considering different cities for opening a new outlet.  I would like to expand your business to cities that may give your restaurant higher profits. The chain already has restaurants in various cities and I have data for profits and populations from the cities. 
Outline
1 - Packages
2 - Linear regression with one variable
2.1 Problem Statement
2.2 Dataset
2.3 Refresher on linear regression
2.4 Compute Cost
Exercise 1
2.5 Gradient descent
Exercise 2
2.6 Learning parameters using batch gradient descent

1 - Packages
First, let's run the cell below to import all the packages that you will need during this assignment.

numpy is the fundamental package for working with matrices in Python.
matplotlib is a famous library to plot graphs in Python.
utils.py contains helper functions for this assignment. You do not need to modify code in this file.

[2]
0s
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
%matplotlib inline
2 - Problem Statement
Suppose you are the CEO of a restaurant franchise and are considering different cities for opening a new outlet.

You would like to expand your business to cities that may give your restaurant higher profits.
The chain already has restaurants in various cities and you have data for profits and populations from the cities.
You also have data on cities that are candidates for a new restaurant.
For these cities, you have the city population.
Can you use the data to help you identify which cities may potentially give your business higher profits?

3 - Dataset
You will start by loading the dataset for this task.

The load_data() function shown below loads the data into variables x_train and y_train
x_train is the population of a city
y_train is the profit of a restaurant in that city. A negative value for profit indicates a loss.
Both X_train and y_train are numpy arrays.

[7]
0s
# load the dataset
x_train, y_train = /content/load_data

Next steps:
View the variables
Before starting on any task, it is useful to get more familiar with your dataset.

A good place to start is to just print out each variable and see what it contains.
The code below prints the variable x_train and the type of the variable.


[ ]
# print x_train
print("Type of x_train:",type(x_train))
print("First five elements of x_train are:\n", x_train[:5]) 
Type of x_train: <class 'numpy.ndarray'>
First five elements of x_train are:
 [6.1101 5.5277 8.5186 7.0032 5.8598]
x_train is a numpy array that contains decimal values that are all greater than zero.

These values represent the city population times 10,000
For example, 6.1101 means that the population for that city is 61,101
Now, let's print y_train


[ ]
# print y_train
print("Type of y_train:",type(y_train))
print("First five elements of y_train are:\n", y_train[:5])  
Type of y_train: <class 'numpy.ndarray'>
First five elements of y_train are:
 [17.592   9.1302 13.662  11.854   6.8233]
Similarly, y_train is a numpy array that has decimal values, some negative, some positive.

These represent your restaurant's average monthly profits in each city, in units of $10,000.
For example, 17.592 represents $175,920 in average monthly profits for that city.
-2.6807 represents -$26,807 in average monthly loss for that city.
Check the dimensions of your variables
Another useful way to get familiar with your data is to view its dimensions.

Please print the shape of x_train and y_train and see how many training examples you have in your dataset.


[ ]
print ('The shape of x_train is:', x_train.shape)
print ('The shape of y_train is: ', y_train.shape)
print ('Number of training examples (m):', len(x_train))
The shape of x_train is: (97,)
The shape of y_train is:  (97,)
Number of training examples (m): 97
The city population array has 97 data points, and the monthly average profits also has 97 data points. These are NumPy 1D arrays.

Visualize your data
It is often useful to understand the data by visualizing it.

For this dataset, you can use a scatter plot to visualize the data, since it has only two properties to plot (profit and population).
Many other problems that you will encounter in real life have more than two properties (for example, population, average household income, monthly profits, monthly sales).When you have more than two properties, you can still use a scatter plot to see the relationship between each pair of properties.

[ ]
# Create a scatter plot of the data. To change the markers to red "x",
# we used the 'marker' and 'c' parameters
plt.scatter(x_train, y_train, marker='x', c='r') 

# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
plt.show()

Your goal is to build a linear regression model to fit this data.

With this model, you can then input a new city's population, and have the model estimate your restaurant's potential monthly profits for that city.

4 - Refresher on linear regression
In this practice lab, you will fit the linear regression parameters (w,b) to your dataset.

The model function for linear regression, which is a function that maps from x (city population) to y (your restaurant's monthly profit for that city) is represented as
fw,b(x)=wx+b
To train a linear regression model, you want to find the best (w,b) parameters that fit your dataset.

To compare how one choice of (w,b) is better or worse than another choice, you can evaluate it with a cost function J(w,b)

J is a function of (w,b). That is, the value of the cost J(w,b) depends on the value of (w,b).
The choice of (w,b) that fits your data the best is the one that has the smallest cost J(w,b).

To find the values (w,b) that gets the smallest possible cost J(w,b), you can use a method called gradient descent.
With each step of gradient descent, your parameters (w,b) come closer to the optimal values that will achieve the lowest cost J(w,b).
The trained linear regression model can then take the input feature x (city population) and output a prediction fw,b(x) (predicted monthly profit for a restaurant in that city).

5 - Compute Cost
Gradient descent involves repeated steps to adjust the value of your parameter (w,b) to gradually get a smaller and smaller cost J(w,b).

At each step of gradient descent, it will be helpful for you to monitor your progress by computing the cost J(w,b) as (w,b) gets updated.
In this section, you will implement a function to calculate J(w,b) so that you can check the progress of your gradient descent implementation.
Cost function
As you may recall from the lecture, for one variable, the cost function for linear regression J(w,b) is defined as

J(w,b)=12m∑i=0m−1(fw,b(x(i))−y(i))2

You can think of fw,b(x(i)) as the model's prediction of your restaurant's profit, as opposed to y(i), which is the actual profit that is recorded in the data.
m is the number of training examples in the dataset
Model prediction
For linear regression with one variable, the prediction of the model fw,b for an example x(i) is representented as:
fw,b(x(i))=wx(i)+b

This is the equation for a line, with an intercept b and a slope w

Implementation
Please complete the compute_cost() function below to compute the cost J(w,b).
