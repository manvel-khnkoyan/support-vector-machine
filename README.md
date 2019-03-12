

**SVM: From Theory To Practice**

In this article we are going to create simple support vector machine without deep analyze.


***Liner SVM***


As we know, the purpose of the SVM is to search for the hyperline, which will be the best separation between two different types:

Shorter we need to find linear equation

> a x + b y + c = 0

or in multi deministion space:

> a1 x1 + a2 x2 + .... + b  = 0

or shorter 

> **w** **x** + b = 0
  
lets define 1 for one type  
> **w** **x** + b = 1

lets define -1 for another type  
> **w****x** + b = -1

or in general:
> yi ( **w** **x** + b ) >= 0



According support vector machine therory we come on lagrangian equations: (for more details there many documentation you can get from internet)

> Sum(i) ( ai ) - 0.5 * Sum(i,j) ( ai aj yi yj <**xi** **xj**> )

> Sum(i) (ai yi) = 0

where to whether 1 or -1 depend on type


Maximising this equation as a optimisation problem. So we have "y" and "**x**" and our purpose is find "a".


There are many methods to find these variables, but my solution is solve it analytically according coordinate descent algorithm.
The purpose of this algorithm is change each ai element till lagrangian showed above will reach maximum. But don't forget that sum of ai * yi should be 0.


```python

def coordinate_descent(dataset, alfa, precision, delta, Q, type ):
    go = True
    while go:
        go = False
        for index, value in enumerate(dataset):
            # detecting direction which direction cause rising of lagrangian 
            direction = detect_maximum_direction(dataset, alfa, index, delta, type, Q)
            next = get_lagrangian(dataset, alfa, type, Q)
            # changing alfa[index] till lagrangian will get maximum
            while True:
                prev = next
                change_alfa(dataset, alfa, index, direction * delta)
                next = get_lagrangian(dataset, alfa, type, Q)
                if (next - prev) < precision:
                    break
                go = True
    return alfa

``` 

In this function we find "a" -s (in code it is "alfa"), looping over each "a" till we get in maximum of lagrangian. (for more details look up code)


Having "a"-s we easyly can find hypreline :

weight: **w** = Sum(i) ai yi **xi**  
basis: b - avg( <**x** **w**>  )


now you can do prediction 

> Sign( **w** **x** + b) 

or
> Sign( Sum(i)( ai yi <**x**  **x_test**> ) + b )

So having "a" -s you can build a hyperline and make a prediction


***Finding Support Vectors***

The example we discussed supposed that we have only few vectors to analyze. But when the number of input vectors are high, "coordinate_descent" function can't find "a" because of hard work.
(remember that "a" is find analytically)

According lagrangian equations, function gets his maximum when we to take into account only those opposite vectors which distance is minimum.
So we can just filter given vectors and name theme support vectors, and process only these vectors instead of all.


Here are code how I did it

```python
def find_support_vectors(dataset, C=0.5, slack=0.5):
    distances = {}
    for i,v1 in enumerate(dataset):
        for j,v2 in enumerate(dataset):
            if i != j and v1[2] != v2[2]:
                distance = math.sqrt((v1[0] - v2[0])**2 + (v1[1] - v2[1])**2)
                if i not in distances or distance < distances[i]:
                    distances[i] = distance

    support_vectors = []
    for index,vector in enumerate(dataset):
        distance = distances[index]
        if distance > C - slack and distance < C + slack:
            support_vectors.append(vector)
    return support_vectors
```

Having support vectors we can easily analyze any complex data.



***Kernel Trick***

When we have non line data separatable, in this case we are doing some trick - instead of calculating support vectors dot product, we change it with some function.
In my example I use radial basis function (RBF) :

> exp(-Q * || **x1** - **x2** ||) 

(where Q is some constant) or :  
```python
math.exp(-Q*(x1**2 + x2**2))
``` 


***Results***

As as test I generated about 100 random points with liner and paraboloid (non liner) seperation. 
Then generated random 20 points and tested them. To show result I used matplotlib.

To test yourself - clone the project and run in your terminal `python3 predict_liner.py` to see result liner svm or `python3 predict_nonliner.py`  to see non liner result.

Example 1: Liner SVM

<img src="{{site.url}}/img/liner.png"> 

Example 2: Non Liner SVM 

<img src="{{site.url}}/img/nonliner.png"> 

where

<img src="{{site.url}}/img/description.png">