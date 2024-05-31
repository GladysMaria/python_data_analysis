# Plotting 


```python
import matplotlib.pyplot as plt
squares = [1,4,9,16,25]
fig, ax = plt.subplots()
ax.plot(squares)
ax.set(xlabel='Value', ylabel='Square of Value',
       title='Square Numbers')
ax.grid()

fig.savefig("test_linear.png")
plt.show()
```


    
![png](output_1_0.png)
    



```python
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
t = np.arange(0.0, 2.0, 0.01)
s = 1 + np.sin(2 * np.pi * t)

fig, ax = plt.subplots()
ax.plot(t, s)

ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()

fig.savefig("test_linear_2.png")
plt.show()
```


    
![png](output_2_0.png)
    



```python

```


```python
# Bar color demo

import matplotlib.pyplot as plt

fig, ax = plt.subplots()


fruits = ['apple', 'blueberry', 'cherry', 'orange']
counts = [40, 100, 30, 55]
bar_labels = ['red', 'blue', '_red', 'orange']
bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

ax.bar(fruits, counts, label=bar_labels, color=bar_colors)

ax.set_ylabel('fruit supply')
ax.set_title('Fruit supply by kind and color')
ax.legend(title='Fruit color')
fig.savefig("test_bar.png")
plt.show()
```


    
![png](output_4_0.png)
    



```python
import matplotlib.pyplot as plt
import numpy as np

# Fixing random state for reproducibility
np.random.seed(19680801)

fig, ax = plt.subplots()

# Example data
people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

ax.barh(y_pos, performance, xerr=error, align='center')
ax.set_yticks(y_pos, labels=people)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Performance')
ax.set_title('How fast do you want to go today?')
fig.savefig("bar_h.png")
plt.show()
```


    
![png](output_5_0.png)
    



```python
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d

plt.style.use('_mpl-gallery')

# Make data
X, Y, Z = axes3d.get_test_data(0.05)

# Plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)

ax.set(xticklabels=[],
       yticklabels=[],
       zticklabels=[])
fig.savefig("volumen.png")

plt.show()
```


    
![png](output_6_0.png)
    


## Del Libro "Python Crash course"

### Capítulo "Generating Data"

*Plotting a Simple Line Graph"*


```python
import matplotlib.pyplot as plt
squares =[1,4,9,16,25]
```


```python
fig, ax = plt.subplots()
ax.plot(squares)
plt.show()
```


    
![png](output_11_0.png)
    



```python
import matplotlib.pyplot as plt
squares =[1,4,9,16,25]
```


```python
fig, ax = plt.subplots()

ax.plot(squares, linewidth=3)

ax.set_title("Square Numbers", fontsize=24)
ax.set_xlabel("Value", fontsize=14)
ax.set_ylabel("Square of Value", fontsize=14)
ax.tick_params(axis="both", labelsize=14)


plt.show()
```


    
![png](output_13_0.png)
    



```python
input_values = [1,2,3,4,5]
squares = [1,4,9,16,25]
fig, ax = plt.subplots()
ax.plot(input_values, squares, linewidth=3)
ax.set_title("Squares Number", fontsize=24)
ax.set_xlabel("Value", fontsize=14)
ax.set_ylabel("Square of Value", fontsize=14)
ax.tick_params(axis="both", labelsize=14)
plt.show()
```


    
![png](output_14_0.png)
    


*Using Built-in Styles*


```python
plt.style.available
```




    ['Solarize_Light2',
     '_classic_test_patch',
     '_mpl-gallery',
     '_mpl-gallery-nogrid',
     'bmh',
     'classic',
     'dark_background',
     'fast',
     'fivethirtyeight',
     'ggplot',
     'grayscale',
     'seaborn-v0_8',
     'seaborn-v0_8-bright',
     'seaborn-v0_8-colorblind',
     'seaborn-v0_8-dark',
     'seaborn-v0_8-dark-palette',
     'seaborn-v0_8-darkgrid',
     'seaborn-v0_8-deep',
     'seaborn-v0_8-muted',
     'seaborn-v0_8-notebook',
     'seaborn-v0_8-paper',
     'seaborn-v0_8-pastel',
     'seaborn-v0_8-poster',
     'seaborn-v0_8-talk',
     'seaborn-v0_8-ticks',
     'seaborn-v0_8-white',
     'seaborn-v0_8-whitegrid',
     'tableau-colorblind10']




```python

```


```python
input_values = [1,2,3,4,5]
squares = [1,4,9,16,25]
plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.plot(input_values, squares, linewidth=3)
ax.set_title("Squares Number", fontsize=24)
ax.set_xlabel("Value", fontsize=14)
ax.set_ylabel("Square of Value", fontsize=14)
ax.tick_params(axis="both", labelsize=14)
plt.show()
```


    
![png](output_18_0.png)
    



```python

```


```python
input_values = [1,2,3,4,5]
squares = [1,4,9,16,25]
plt.style.use('Solarize_Light2')
fig, ax = plt.subplots()
ax.plot(input_values, squares, linewidth=3)
ax.set_title("Squares Number", fontsize=24)
ax.set_xlabel("Value", fontsize=14)
ax.set_ylabel("Square of Value", fontsize=14)
ax.tick_params(axis="both", labelsize=14)
plt.show()
```


    
![png](output_20_0.png)
    



```python

```


```python
input_values = [1,2,3,4,5]
squares = [1,4,9,16,25]
plt.style.use('_mpl-gallery')
fig, ax = plt.subplots()
ax.plot(input_values, squares, linewidth=3)
ax.set_title("Squares Number", fontsize=24)
ax.set_xlabel("Value", fontsize=14)
ax.set_ylabel("Square of Value", fontsize=14)
ax.tick_params(axis="both", labelsize=14)
plt.show()
```


    
![png](output_22_0.png)
    



```python

```


```python
input_values = [1,2,3,4,5]
squares = [1,4,9,16,25]
plt.style.use('tableau-colorblind10')
fig, ax = plt.subplots()
ax.plot(input_values, squares, linewidth=3)
ax.set_title("Squares Number", fontsize=24)
ax.set_xlabel("Value", fontsize=14)
ax.set_ylabel("Square of Value", fontsize=14)
ax.tick_params(axis="both", labelsize=14)
plt.show()
```


    
![png](output_24_0.png)
    



```python

```


```python
input_values = [1,2,3,4,5]
squares = [1,4,9,16,25]
plt.style.use('tableau-colorblind10')
fig, ax = plt.subplots()
ax.plot(input_values, squares, linewidth=3)
ax.set_title("Squares Number")
ax.set_xlabel("Value")
ax.set_ylabel("Square of Value")
ax.tick_params(axis="both")
plt.show()
```


    
![png](output_26_0.png)
    



```python

```


```python
input_values = [1,2,3,4,5]
squares = [1,4,9,16,25]
plt.style.use('seaborn-v0_8-poster')
fig, ax = plt.subplots()
ax.plot(input_values, squares, linewidth=3)
ax.set_title("Squares Number", fontsize=24)
ax.set_xlabel("Value", fontsize=14)
ax.set_ylabel("Square of Value", fontsize=14)
ax.tick_params(axis="both", labelsize=14)
plt.show()
```


    
![png](output_28_0.png)
    



```python

```


```python
input_values = [1,2,3,4,5]
squares = [1,4,9,16,25]
plt.style.use('seaborn-v0_8-poster')
fig, ax = plt.subplots()
ax.plot(input_values, squares)
ax.set_title("Squares Number")
ax.set_xlabel("Value")
ax.set_ylabel("Square of Value")
ax.tick_params(axis="both")
plt.show()
```


    
![png](output_30_0.png)
    



```python

```


```python

```
