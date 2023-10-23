#!/usr/bin/env python3
import matplotlib.pyplot as plt
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

x = list(range(11))  # 0, 1, 2, ..., 10

plt.plot(x, y, 'r-')  
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Graph of y')
plt.show()
