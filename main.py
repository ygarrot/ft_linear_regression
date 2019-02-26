import numpy as np
import argparse
import feature_scaling
import matplotlib.pyplot as plt
from feature_scaling import rescaling
from ft_linear_regression import LinearRegression
import os.path

def run():
    parser = argparse.ArgumentParser(description='process linear regression')
    parser.add_argument("-d", "--display", dest="display", default=False, action='store_true',
                             help="display plot and function")
    parser.add_argument("-a", "--animation", dest="animation", default=False, action='store_true',
                              help="much wow such waw") 
    parser.add_argument("file_name", type=str, default="Data.csv", help="file_name")
    parser.add_argument("-n", "--normalize", dest="normalization", default="mean_normalization", choices=["rescaling", "mean_normalization"],
            help="choose normalize function") 
    opt = parser.parse_args()
    lr = LinearRegression(display=opt.display, feature_scaling=getattr(feature_scaling, opt.normalization))
    theta = [0, 0]
    predict_path = "predict.csv"
    points = np.genfromtxt(opt.file_name, delimiter=",")
    if (os.path.exists(predict_path)):
        ar = np.genfromtxt(predict_path, delimiter=",")
        theta = ar[0]
    points = np.delete(points, (0), 0)
    mileage = points[:,0]
    price = points[:,1]
    print ("Starting gradient descent at theta[0] = ", theta[0], ", theta[1] = ", theta[1])
    print ("Running...")
    if (opt.animation is True):
       lr.animation(points, theta)
    else:
        theta = lr.gradient_descent_runner(points, theta)
    print ("Starting gradient descent at theta[0] = ", theta[0], ", theta[1] = ", theta[1])
 
if __name__ == '__main__':
    run()
