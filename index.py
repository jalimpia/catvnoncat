import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from scipy import ndimage
from matplotlib import image
from PIL import Image
from lr_utils import load_dataset
from matplotlib.image import imread

from flask import Flask,render_template, request
from werkzeug import secure_filename
import os.path


app = Flask(__name__)

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s 
	
def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim,1))
    b = 0
    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))
    return w,b
	
def propagate(w,b,X,Y):
    m = X.shape[1]
    
    #Forward Propagation
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1/m) * np.sum(Y * np.log(A) + (1-Y) * (np.log(1-A)))
    
    #Backward Propagation
    dw = (1/m) * np.dot(X, (A-Y).T)
    db = (1/m) * np.sum(A-Y)
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {'dw':dw, 'db':db}
    return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    costs = []
    for i in range(num_iterations):
        #Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)
        
        #Get derivatives
        dw = grads['dw']
        db = grads['db']
        
        #Update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        #Record costs
        if i % 100 == 0:
            costs.append(cost)
            
        #Print cost every 100 training examples
        if print_cost and i % 100 == 0:
            print('Cost after iteration %i: %f' % (i,cost))
        
    params = {'w': w, 'b':b}
    grads = {'dw':dw, 'db':db}
    
    return params, grads, costs

def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    #Compute probabilty vector
    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        Y_prediction[0,i] = 1 if A[0,i] > 0.5 else 0
        
    assert(Y_prediction.shape == (1,m))
    
    return Y_prediction
	
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    #Initialize parameters with 0s
    w, b = initialize_with_zeros(X_train.shape[0])
    
    #Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    #Retrieve parameters w, b from dictionary
    w = parameters['w']
    b = parameters['b']
    
    #Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)
    
    #Print tes/train errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    d = {'costs' : costs,
        'Y_prediction_test' : Y_prediction_test,
        'Y_prediction_train' : Y_prediction_train,
        'w' : w,
        'b' : b,
        'learning_rate' : learning_rate,
        'num_iterations' : num_iterations}
    return d	



train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten/255. #dot (.) for float
test_set_x = test_set_x_flatten/255.	

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 100000, learning_rate = 0.005, print_cost = True)
	
def train(path):

	img = Image.open(path)
	img = img.resize((64,64))
	image = np.array(img).reshape((1, 64*64*3)).T
	my_predicted_image = predict(d["w"], d["b"], image)
	
	return str(np.squeeze(my_predicted_image))
	
	

@app.route("/")
def index():
	return render_template("index.html")
	
@app.route("/submit", methods=['POST'])
def submit():
	UPLOAD_FOLDER = 'static\img'
	file = request.files['photo']
	filename = secure_filename(file.filename)
	path = os.path.join(UPLOAD_FOLDER, filename)
	file.save(path)
	predict = train(path)
	
	return render_template("index.html",predict=predict,path=path)
	
if __name__=="__main__":
	app.run(debug=True, host="0.0.0.0")
			