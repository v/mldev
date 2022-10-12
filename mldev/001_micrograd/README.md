# micrograd

Andrej Karpathy has a [nice lecture on YouTube](https://www.youtube.com/watch?v=VMj-3S1tku0) where he walks through backpropagation and neural networks from scratch.

I watched this lecture, took notes and did the linked exercise.

## Lecture notes

[The micrograd library](https://github.com/karpathy/micrograd) which is a toy implementation of automatic differentiation and backpropagation is tiny (~200 LOC). Yet, it's able to train and predict a multi-layer neural network.

Karpathy builds up all of neural network training using plain Python, and no external libraries. He doesn't even use matricies in his implementation, which makes neural networks appear very approachable for a Python programmer with basic knowledge. 


I feel that this is pretty different from my first experience with [fastai](https://www.fast.ai/), where I feel like I am trying to learn the framework after Lecture 1, and every line of code does a lot more. Doing the fast.ai exercises feels more frustrating right now, because I feel as though I am trying to remember an incantation, whereas doing the micrograd exercises feels like I am thinking about the math / structure of a neural network.

### Don't be scared of the calculus

I feel that this lecture helps me understand neural nets from the ground up without making my head hurt with calculus that I don't remember.

For example, Karpathy's explains chain rule in a nice way. He points out that the first formulation on Wikipedia is hard to understand:

if $h = f \circ g$ is the function such that $h(x) = f(g(x))$ for every x, then the chain rule is

$$ h'(x) = f'(g(x)) g'(x). $$

whereas this formulation is much more intuitive: 

If a variable z depends on the variable y, which itself depends on the variable x (that is, y and z are dependent variables), then z depends on x as well, via the intermediate variable y. In this case, the chain rule is expressed as:

$$ \frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx} $$

Once you understand this formulation, it makes sense that the `_backward()` function always has an implementation that looks like:

```python
def _backward():
     # __something__ is the derivative of the current op and it's being multiplied
     # by the derivative of the next op
    self.grad += out.grad * __something__
```

Similarly, I find that the numerical way to compute gradients is very practical, and a very nice trick.

Given that the definition of a derivative is as follows:

$$ f'(x) = \lim_{h\to0} \frac{f(x + h) - f(x)}{h}  $$

you can compute a derivate numerically in Python:

```python

def f(x):
    # implementation of f goes here
    # ...

    # example:
    return x ** 3 + 3 * (x ** 2) / 10

h = 0.000001
x = 10

# f'(10)
print((f(x + h) - f(x)) / h)
```

### Visualizing the network

It is very neat that Karpathy can draw Graphviz graphs to visualize the networks he builds in micrograd using the `draw_dot` function. 

```python
def draw_dot(root, format='svg', rankdir='LR'):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ['LR', 'TB']
    nodes, edges = trace(root)
    dot = Digraph(format=format, graph_attr={'rankdir': rankdir}) #, node_attr={'rankdir': 'TB'})
    
    for n in nodes:
        dot.node(name=str(id(n)), label = "{ data %.4f | grad %.4f }" % (n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=str(id(n)) + n._op, label=n._op)
            dot.edge(str(id(n)) + n._op, str(id(n)))
    
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    
    return dot
```

I wonder if there is something like this for visualizing Pytorch networks. I am sure that something like it can be built with custom code. From a quick google search, [this library seems similar](https://github.com/szagoruyko/pytorchviz). 

**From pytorchviz**

Example usage of `make_dot`
```python
model = nn.Sequential()
model.add_module('W0', nn.Linear(8, 16))
model.add_module('tanh', nn.Tanh())
model.add_module('W1', nn.Linear(16, 1))

x = torch.randn(1, 8)
y = model(x)

make_dot(y.mean(), params=dict(model.named_parameters()))
```

![](https://user-images.githubusercontent.com/13428986/110844921-ff3f7500-8277-11eb-912e-3ba03623fdf5.png)

I am not sure I fully understand where the node names come from - AccumulateGrad, TBackward, AddmmBackward.


## Exercise

It was very nice that [an exercise](https://colab.research.google.com/drive/1FPTx1RXtBfc4MaTkf7viZZD4U2F9gtKN?usp=sharing) was included with the lecture. 

It took me about 2 hours to finish the exercise.

Section 1 took me about 10 minutes to finish. I wrote some code, and it pretty much worked on the first try.
Section 2 took me about an hour to finish. 

My first attempt had a bug where the gradient didn't match the expected values.

I figured that the bug was in one of the forward, or backward passes, but I didn't know where it was because I'd implemented a fair number of functions:
- `__radd__`
- `__mul__`
- `__truediv__`
- `__neg__`
- `__pow__`
- `__pow__`
- `exp`

To find the bug, I ran the forward passes and confirmed that they matched what I'd expect by doing the math by hand. There was no bug in the forward pass.

To test the backward passes, I wrote small chunks of backward pass code and compared the result with Pytorch.

Example snippet:

```python
# confirm that softmax backward pass gives the
# same gradient as torch

logits = [Value(10.0), Value(3.0)]
probs = softmax(logits)
probs[1].backward()
print(f"dim 0 grad {logits[0].grad}")
print(f"dim 1 grad {logits[1].grad}")


#### pytorch version
import torch
logitst = torch.Tensor([10, 3]); logitst.requires_grad = True
probs = torch.softmax(logitst, dim=0)
print(probs[1])
probs[1].backward()
print(logitst.grad.data)
```

It turned out that the bug was in my implementation of `_backward` and I found it using the following snippet:

```python
# verify that the log backward pass gives the
# same gradient as torch

a = Value(2.0)
b = -a.log()

b.backward()
print(f"a grad {a.grad}")

#### pytorch version
import torch
at = torch.Tensor([2.0]); at.requires_grad=True
bt = -at.log()

bt.backward()
print(f"a grad", at.grad.data)
```

The buggy implementation of `log` looked like this:

```python
  def log(self):
    l = log(self.data)
    out = Value(l, (self,), 'log')
    def _backward():
      # derivative of log(x) = 1 / x
      # this should be
      # self.grad += out.grad / self.data

      self.grad += 1.0 * out.grad   # bug is here
    
    out._backward = _backward
    return out
```

It was a copy-paste bug :facepalm: