# Do not use packages that are not in standard distribution of python
import numpy as np
np.random.seed(1024)
from ._base_network import _baseNetwork

class TwoLayerNet(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10, hidden_size=128):
        super().__init__(input_size, num_classes)

        self.hidden_size = hidden_size
        self._weight_init()

        self.loss =0

    def _weight_init(self):
        '''
        initialize weights of the network
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the first layer of shape (num_features, hidden_size)
        - b1: The bias term of the first layer of shape (hidden_size,)
        - W2: The weight matrix of the second layer of shape (hidden_size, num_classes)
        - b2: The bias term of the second layer of shape (num_classes,)
        '''

        # initialize weights
        self.weights['b1'] = np.zeros(self.hidden_size)
        self.weights['b2'] = np.zeros(self.num_classes)
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.hidden_size)
        np.random.seed(1024)
        self.weights['W2'] = 0.001 * np.random.randn(self.hidden_size, self.num_classes)




    def forward(self, X, y, mode='train'):
        '''
        The forward pass of the two-layer net. The activation function used in between the two layers is sigmoid, which
        is to be implemented in self.,sigmoid.
        The method forward should compute the loss of input batch X and gradients of each weights.
        Further, it should also compute the accuracy of given batch. The loss and
        accuracy are returned by the method and gradients are stored in self.gradients

        :param X: a batch of images (N, input_size)
        :param y: labels of images in the batch (N,)
        :param mode: if mode is training, compute and update gradients;else, just return the loss and accuracy
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
            self.gradients: gradients are not explicitly returned but rather updated in the class member self.gradients
        '''
        accuracy = None
        X = np.asarray(X)
        y = np.asarray(y)
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process:                                      #
        #        1) Call sigmoid function between the two layers for non-linearity  #
        #        2) The output of the second layer should be passed to softmax      #
        #        function before computing the cross entropy loss                   #
        #    2) Compute Cross-Entropy Loss and batch accuracy based on network      #
        #       outputs                                                             #
        #############################################################################

        layer1 = self.get_XxW_b(X)

        layer1_derivative = self.sigmoid(layer1) #[1,128]

        scores = layer1_derivative.dot(self.weights["W2"]) + self.weights["b2"]
        #  layer1_dev = [1,128]
        # w2 = [128,10]
        # scores [1,10]
        prediction = self.softmax(scores)

        self.loss = self.cross_entropy_loss(prediction,y)
        # self.addRegularization()

        accuracy = self.compute_accuracy(prediction,y)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if mode != 'train':
            return self.loss, accuracy
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight and bias by chain rule         #
        #        2) Store the gradients in self.gradients                           #
        #    HINT: You will need to compute gradients backwards, i.e, compute       #
        #          gradients of W2 and b2 first, then compute it for W1 and b1      #
        #          You may also want to implement the analytical derivative of      #
        #          the sigmoid function in self.sigmoid_dev first                   #
        #############################################################################

        N, D = X.shape

        prediction[np.arange(N) ,y] -= 1
        prediction /= N

        self.set_W2_gradient(layer1_derivative, prediction)
        self.set_b2_gradient(prediction)

        dw1 = self.get_W1(prediction)

        layer1_derivative = self.get_layer1_derivative(dw1, layer1)

        self.gradients["W1"] = X.T.dot(layer1_derivative)

        self.gradients["b1"] = layer1_derivative.sum(axis=0,keepdims=True)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return self.loss, accuracy

    # def addRegularization(self):
        # self.loss += 0.001 * (
        #             np.sum(self.weights["W2"] * self.weights["W2"]) + np.sum(self.weights["W1"] * self.weights["W1"]))

    def get_layer1_derivative(self, dw1, layer1):
        layer1_dev = dw1 * self.sigmoid_dev(layer1).astype(np.float64)
        return layer1_dev

    def get_W1(self, prediction):
        dw1 = prediction.dot(self.weights["W2"].T)
        return dw1

    def set_b2_gradient(self, prediction):
        self.gradients["b2"] = np.sum(prediction, axis=0, keepdims=True)

    def set_W2_gradient(self, layer1_derivative, prediction):
        self.gradients["W2"] = layer1_derivative.T.dot(prediction)

    def get_XxW_b(self, X):
        # X = [1,784] W = [ 784 , 128]
        # Z = [1,128]
        Z = X.dot(self.weights["W1"]) + self.weights["b1"]
        return Z


