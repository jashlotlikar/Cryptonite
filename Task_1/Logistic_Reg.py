import numpy as np
import matplotlib.pyplot as plt

class Logistic_regression:
    def __init__(self, dataset = None):
        
        if dataset is None:
            self.dataset = np.array([np.linspace(0, 10, 100), np.linspace(0,20,100) + np.random.randn(100)])
        else:
            self.dataset = np.asarray(dataset)
        self.coefficients = np.zeros(np.size(self.dataset, axis=0))
        self.confusionMatrix = np.array([[0,0],[0,0]])
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.specificity = None
        self.F1_Score = None

    def calculate_Loss(self):
        prediction = np.clip(self.predict(), 1e-15, 1 - 1e-15)
        y = self.dataset[-1]
        if np.max(y) > 1:
            y = y / 100
        y_pred = prediction
        Loss = -np.mean(y * np.log(y_pred) + (1-y)*np.log(1 - y_pred))
        return Loss

    def calculate_ConfusionMatrix(self):
        x = self.dataset[0]
        y = self.dataset[-1]
        y_pred = self.predict(x)
        y_pred_10 = [1 if x >= 0.5 else 0 for x in y_pred]
        self.confusionMatrix[1][0] = np.count_nonzero((y - y_pred_10) == 1)
        self.confusionMatrix[0][1] = np.count_nonzero((y - y_pred_10) == -1)
        self.confusionMatrix[0][0] = np.count_nonzero(y_pred_10 == 1)-np.count_nonzero((y - y_pred_10) == 1)
        self.confusionMatrix[1][1] = np.count_nonzero(y_pred_10 == 0)-np.count_nonzero((y - y_pred_10) == -1)

    def calculate_Metrics(self):
        cmatrix = self.confusionMatrix
        self.accuracy = (cmatrix[0][0] + cmatrix[1][1])/(cmatrix[0][0] + cmatrix[0][1] + cmatrix[1][0] + cmatrix[1][1])
        self.precision = (cmatrix[0][0])/(cmatrix[0][0] + cmatrix[1][0])
        self.recall = (cmatrix[0][0])/(cmatrix[0][0] + cmatrix[0][1])
        self.specificity = (cmatrix[1][1])/(cmatrix[1][1] + cmatrix[1][0])
        self.F1_score = 2 * (self.precision * self.recall)/(self.precision + self.recall)

    def predict(self, input = None):
        coefficients = self.coefficients
        if input is None: 
            input = self.dataset[0] 
        output = self.sigmoid(coefficients[1]*input + coefficients[0])
        return output

    def gradient_descent(self, learning_rate = 0.1):
        dataset = self.dataset
        x = dataset[0]
        y = dataset[-1]
        if np.max(y) > 1:
            y = y / 100
        y_pred = self.predict()
        dJ_dm = (y_pred - y) * x
        dJ_dc = (y_pred - y)
        self.coefficients[0] -= learning_rate * np.mean(dJ_dc)
        self.coefficients[1] -= learning_rate * np.mean(dJ_dm)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def plot(self):
        x = self.dataset[0]
        x_curve = np.linspace(x.min(), x.max(), 200)
        y_curve = self.predict(x_curve) 

        plt.scatter(x, self.dataset[1] , label='Observed Value')
        plt.plot(x_curve, y_curve, label='Predicted Value', color='red')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    reg = Logistic_regression()