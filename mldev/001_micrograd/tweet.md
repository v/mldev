### Numerical derivatives

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

### Chain rule formulation
Wikipedia has two ways of explaining chain rule.

**The confusing way**

if $h = f \circ g$ is the function such that $h(x) = f(g(x))$ for every x, then the chain rule is

$$ h'(x) = f'(g(x)) g'(x). $$

**The intuitive way**

If a variable *z* depends on the variable *y*, which itself depends on the variable *x* (that is, *y* and *z* are dependent variables), then *z* depends on *x* as well, via the intermediate variable *y*. 

In this case, the chain rule is expressed as:

$$ \frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx} $$

Just from the formulation, the second statement is more obvious.


### Chain rule and backpropagation

Given *x*, *y* and *z* such that

 `z = f(y)` and `y = g(x)`

 then chain rule tells us that

```
y.grad = f'(y)

x.grad = g'(x) * y.grad
```

This is why the backward function of a neural net operation looks like this:


```python
def _backward():
     # __something__ is the derivative of the current op and
     # it's being multiplied # by the derivative of the next op
    self.grad += out.grad * __something__
```