"""
softmax with loss
"""
import time

import numpy as np
import torch
from torch.autograd import Function, Variable


def softmax(z):
    z -= np.max(z)
    sm = np.exp(z) / np.expand_dims(np.sum(np.exp(z), axis=1), axis=1)
    return sm


def softmax_grad(s):
    # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
    # input s is softmax value of the original input x.
    # s.shape = (1, n)
    # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])
    # initialize the 2-D jacobian matrix.
    jacobian_m = np.diag(s)
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1 - s[i])
            else:
                jacobian_m[i][j] = -s[i] * s[j]
    return jacobian_m


def softmax_grad_vectorized(softmax):
    # Reshape the 1-d softmax to 2-d so that np.dot will do the matrix multiplication
    s = softmax.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def SoftmaxLossFunc(x, target):
    exp_x = torch.exp(x)
    y = exp_x / exp_x.sum(1).unsqueeze(1).expand_as(exp_x)
    t = torch.zeros(y.size()).type(y.data.type())
    for n in range(t.size(0)):
        t[n][target.data[n]] = 1

    t = Variable(t)
    output = (-t * torch.log(y)).sum() / y.size(0)
    return output


class SoftmaxLoss(Function):
    r"""
    softmax with cross entropy
    log_softmax:
    y = log(\frac{e^x}{\sum e^{x_k}})
    negative likelyhood:
    z = - \sum t_i y_i, where t is one hot target
    """

    @staticmethod
    def forward(ctx, x, target):
        """
        forward propagation
        """
        assert x.dim() == 2, "dimension of input should be 2"
        exp_x = torch.exp(x)
        y = exp_x / exp_x.sum(1).unsqueeze(1).expand_as(exp_x)

        # parameter "target" is a LongTensor and denotes the labels of classes, here we need to convert it into one hot vectors
        t = torch.zeros(y.size()).type(y.type())
        for n in range(t.size(0)):
            t[n][target[n]] = 1

        output = (-t * torch.log(y)).sum() / y.size(0)

        # output should be a tensor, but the output of sum() is float
        output = torch.Tensor([output]).type(y.type())
        ctx.save_for_backward(y, t)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        backward propagation
        # softmax with ce loss backprop see https://www.youtube.com/watch?v=5-rVLSc2XdE
        """
        y, t = ctx.saved_tensors

        # grads = []
        # for i in range(y.size(0)):
        #     grads.append(softmax_grad(y[i]))

        grads = softmax_grad_vectorized(y)
        grad_input = grad_output * (y - t) / y.size(0)
        return grad_input, None


def test_softmax_loss_backward():
    """
    analyse the difference between autograd and manual grad
    """
    x_size = 100
    num_classes = 4
    # generate random testing data
    x = torch.randn(x_size, num_classes).double()
    x_var = Variable(x, requires_grad=True)

    # testing labels
    target = torch.LongTensor(np.repeat(np.array(range(num_classes)), x_size // num_classes))
    target_var = Variable(target)

    # compute outputs of softmax loss
    softmax_loss = SoftmaxLoss.apply
    y = softmax_loss(x_var, target_var)

    # clone testing data
    x_copy = x.clone()
    x_var_copy = Variable(x_copy, requires_grad=True)

    # compute output of softmax loss
    y_hat = SoftmaxLossFunc(x_var_copy, target_var)

    # compute gradient of input data with two different method
    t0 = time.time()
    y.backward()  # manual gradient
    t1 = time.time()
    y_hat.backward()  # auto gradient
    t2 = time.time()

    dist = (y_hat - y).data.abs().sum()
    print("=====================================================")
    print("|===> testing softmax loss forward")
    print("distance between y_hat and y: ", dist)

    dist = (x_var.grad - x_var_copy.grad).data.abs().sum()
    print("|===> testing softmax loss backward")
    print("y: ", y)
    print("y_hat: ", y_hat)
    print("x_grad: ", x_var.grad)
    print("x_copy.grad: ", x_var_copy.grad)
    print("distance between x.grad and x_copy.grad: ", dist)

    print("|===> comparing time-costing")
    print("time of manual gradient: ", t1 - t0)
    print("time of auto gradient: ", t2 - t1)
    # different dist=1.38e-7 with float precision
    # different dist=3.34e-16 with double precision


test_softmax_loss_backward()
