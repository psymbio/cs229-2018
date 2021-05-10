import numpy as np
import util

from linear_model import LinearModel

def h(theta, x):
    return 1/(1+np.exp(-np.dot(x,theta)))

def gradient(x,y, theta):
    m, = y.shape
    return -1/m * np.dot(x.T, y - h(theta, x))

def hessian(x,y, theta):
    m, = y.shape
    htx = np.reshape(h(theta,x), (-1,1))
    return 1/m * np.dot(x.T, x*htx*(1-htx))

def main(train_path, eval_path, pred_path):
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    # logistic regression model loaded
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)
    # Plot data and decision boundary
    util.plot(x_train, y_train, model.theta, 'output/p01b_{}.png'.format(pred_path[-5]))
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred, fmt='%d')

class LogisticRegression(LinearModel):
    def fit(self, x, y):
	  def nexttheta(theta):
	  	grad = gradient(x,y,theta)
		H = hessian(x,y,theta)
		H_inv = np.linalg.inv(H)
		return theta - np.dot(H_inv, grad)

        m,n = x.shape
        theta_prev = np.zeros(n)
        theta_next = nexttheta(theta_prev)
        
        while np.linalg.norm(theta_prev - theta_next, 1) > self.eps:
            theta_prev = theta_next
            theta_next = nexttheta(theta_prev)
            
        self.theta = theta_next
    def predict(self, x):
        return 1 / (1 + np.exp(-x.dot(self.theta)))

