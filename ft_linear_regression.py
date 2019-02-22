import sys
import os
import matplotlib.animation as animation
import numpy as np
from feature_scaling import mean_normalization
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
line, = ax.plot(1)

class LinearRegression():
    theta = [0,0] 

    def __init__(self, num_iterations = 1000, learning_rate = 0.01, verbose = False, display = False, feature_scaling=mean_normalization):
        self.feature_scaling = feature_scaling
        self.points = []
        self.display = display
        self.verbose = verbose
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
        self.theta = [0, 0]
        
    def cost_function(self, theta, mileage, price):
        m = len(mileage)
        su = 0
        for i in range(0, m):
            su += (self.estimate_price(theta, mileage[i]) - price[i]) ** 2
        return (1 / (2 * m)) * su 

    def predict(self, mileage, file_name):
        theta = [0, 0]
        norm = [1, 1]
        if (os.path.exists(file_name)):
            theta, norm = np.genfromtxt(file_name, delimiter=",")
        price = self.estimate_price(theta, self.feature_scaling(mileage, norm[0], norm[1])) 
        print(price)
        return (price)
    
    def estimate_price(self, theta, mileage):
        return theta[0] + (theta[1] * mileage)

    def compute_theta0(self, theta, mileage, price):
        return self.estimate_price(theta, mileage) - price

    def compute_theta1(self, theta, mileage, price):
        return (self.estimate_price(theta, mileage) - price) * mileage

    def step_gradient(self, theta, points):
        tmp_theta = [0.0, 0.0]
        mileage, price = points
        m = len(mileage)
        for i in range(0, len(mileage)):
            tmp_theta[0] += self.compute_theta0(theta, mileage[i], price[i]) 
            tmp_theta[1] += self.compute_theta1(theta, mileage[i], price[i]) 
        tmp_theta[0] *= (self.learning_rate / m)
        tmp_theta[1] *= (self.learning_rate / m)
        # self.debug_gradient(theta, tmp_theta)
        theta[0] -= tmp_theta[0]
        theta[1] -= tmp_theta[1]
        return theta 

    def gradient_descent_runner(self, points, theta):
        cost = []
        price = points[:, 1:]
        # price = self.feature_scaling(ar[:, 1:])
        mileage = points[:, :1];
        tmp =[mileage.mean(), mileage.std()] 
        mileage = self.feature_scaling(mileage)
        # print(mileage)
        for i in range(self.num_iterations):
            cost.append(self.cost_function(theta, mileage, price))
            theta = self.step_gradient(theta, [mileage, price])

        if (self.display is True):
            self.display_cost(cost)
            self.display_gradient(points, theta)

        np.savetxt("predict.csv", np.asarray([theta, tmp]), delimiter=",")
        return theta

    def display_cost(self, cost):
        fig = plt.figure()
        fig.suptitle("Cost function minimization")
        ax = plt.axes()
        plt.xlabel("m iterations")
        plt.ylabel("Mean Squared Error")
        ax.plot(cost)
        plt.show()

    def display_gradient(self, points, theta):
        mileage = points[:, :1]
        price = points[:, 1:]
        fig = plt.figure()
        fig.suptitle('t')
        ax = plt.axes()
        plt.xlabel("Mileage")
        plt.ylabel("Price")
        plt.scatter(mileage, price)
        reg_line = theta[1] * self.feature_scaling(mileage) + theta[0]
        ax.plot(mileage, reg_line, 'r-', mileage, price, 'o')
        plt.show()

    def animate(self, i):
        points = self.points
        theta = self.gradient_descent_runner(self.points, self.theta)
        self.theta = theta
        mileage = points[:, :1]
        price = points[:, 1:]
        reg_line = theta[1] * self.feature_scaling(mileage) + theta[0]
        if self.testl is not False:
           self.testl.remove()
        self.testl, = ax.plot(mileage, reg_line,color='green')
        return line,

    def animation(self, points, theta):
        self.points = points
        self.theta =[0, 0]
        self.num_iterations = 1
        mileage = points[:, :1]
        price = points[:, 1:]
        plt.scatter(mileage, price)
        self.testl = False
        plt.xlabel("Mileage")
        plt.ylabel("Price")
        ani = animation.FuncAnimation(fig, self.animate,
        frames=np.arange(1, 1000), interval=10, blit=True)
        plt.show()
