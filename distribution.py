import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction

# Define the exact coefficients using Fraction
a = Fraction(-6)                   # exactly -6
b = Fraction(1182, 145)              # 1182/145
c = Fraction(1137, 1450)             # 1137/1450
d = Fraction(-1467, 725)             # -1467/725
e = Fraction(1323, 1450)             # 1323/1450

# Print the coefficients in fractional form (optional)
print("Normalized polynomial coefficients:")
print("a =", a)
print("b =", b)
print("c =", c)
print("d =", d)
print("e =", e)

# Define the normalized quartic polynomial g(x)
def g(x):
    # Convert fractions to float during evaluation
    return float(a)*x**4 + float(b)*x**3 + float(c)*x**2 + float(d)*x + float(e)

# Create an array of x-values over [0,1]
x_vals = np.linspace(0, 1, 500)
# Compute corresponding g(x) values
y_vals = [g(x) for x in x_vals]

# Define the key points according to our conditions:
# f(0)=1/2, f(0.3)=3/10, f(1)=1.
key_x = [0, 0.3, 1]
key_y = [g(0), g(0.3), g(1)]

# Print key point values (optional)
print("g(0) =", g(0))
print("g(0.3) =", g(0.3))
print("g(1) =", g(1))

# Plot the polynomial and key points
plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label="g(x) - Normalized Quartic")
plt.scatter(key_x, key_y, color='red', zorder=5, label="Key Points")
plt.xlabel("x")
plt.ylabel("g(x)")
plt.title("Visualization of Normalized Quartic Polynomial with Integral=1")
plt.legend()
plt.grid(True)
plt.show()
