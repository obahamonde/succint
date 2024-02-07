# Deep Learning

---
1. BASIC CONCEPTS
---

**Activation Function**: Entrophy to initialize an state in a non linear way for a set of layers of a DNN (Deep Neural Network).


_Activation Function_

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

```python
import torch

def sigmoid(x:torch.Tensor):
	return 1 / (1 + torch.exp(-x))
```

**Forward Pass**: The process of passing the input data through the network to get the output.

_Forward Pass_

$$\hat{y} = \sigma(w_2 \cdot \sigma(w_1 \cdot x + b_1) + b_2)$$

Where:

- $\hat{y}$: Predicted output

- $\sigma$: Activation function

- $w_1, w_2$: Weights

- $b_1, b_2$: Biases

- $x$: Input data

```python
import torch

def forward_pass(x:torch.Tensor, w1:torch.Tensor, w2:torch.Tensor, b1:torch.Tensor, b2:torch.Tensor):
	return sigmoid(w2 @ sigmoid(w1 @ x + b1) + b2)
```


**Loss Function**: The function that measures the error between the predicted output and the actual output.

_Loss Function_

$$L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

Where:

- $N$: Number of samples
- $y_i$: Actual output
- $\hat{y}_i$: Predicted output

```python
import torch

def loss_function(y:torch.Tensor, y_hat:torch.Tensor):
	return torch.mean((y - y_hat)**2)
```

**Backpropagation**: The process of updating the weights of the network using the gradient descent algorithm after the forward pass in one iteration (epoch), we compute a new loss function and update the weights.
Calculating the first order derivative of the loss function with respect to the weights. Pytorch does this automatically.

$$\frac{\partial L}{\partial w}$$

```python
import torch

def backpropagation(loss:torch.Tensor, weight:torch.Tensor):
	return torch.autograd.grad(loss, weight)
```

**Gradient Descent**: The process of updating the weights of the network to minimize the loss function. 

_Gradient Descent_

$$w_{i+1} = w_i - \alpha \frac{\partial L}{\partial w}$$

Where:

- $w_{i+1}$: New weight

- $w_i$: Old weight

- $\alpha$: Learning rate

- $\frac{\partial L}{\partial w}$: Gradient of the loss function with respect to the weight

```python
import torch

def gradient_descent(weight:torch.Tensor, learning_rate:float, gradient:torch.Tensor):
	return weight - learning_rate * gradient
```

**Vanishing Gradient Problem**: The problem of the gradients becoming very small as they are backpropagated through the network, which makes the weights of the network not updating properly.

_Vanishing Gradient Problem_

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w}$$

Where:

- $\frac{\partial L}{\partial \hat{y}}$: Gradient of the loss function with respect to the predicted output

- $\frac{\partial \hat{y}}{\partial w}$: Gradient of the predicted output with respect to the weight

```python
import torch

def vanishing_gradient_problem(gradient_loss:torch.Tensor, gradient_output:torch.Tensor):
	return gradient_loss * gradient_output
```

**Exploding Gradient Problem**: The problem of the gradients becoming very large as they are backpropagated through the network, which makes the weights of the network updating too much.

_Exploding Gradient Problem_

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial w}$$

Where:

- $\frac{\partial L}{\partial \hat{y}}$: Gradient of the loss function with respect to the predicted output

- $\frac{\partial \hat{y}}{\partial w}$: Gradient of the predicted output with respect to the weight

```python

import torch

def exploding_gradient_problem(gradient_loss:torch.Tensor, gradient_output:torch.Tensor):
	return gradient_loss * gradient_output
```





