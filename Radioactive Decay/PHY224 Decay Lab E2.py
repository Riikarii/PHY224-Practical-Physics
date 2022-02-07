import numpy as np
import pylab
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt

# Changing plotting aesthetics
# plt.style.use('seaborn')
mpl.rcParams['figure.figsize'] = (12, 8)
mpl.rcParams['axes.titleweight'] = 'bold'
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.labelsize'] = 13

red = '#ff4d4a'
green = '#2ffa7d'
blue = '#34bdf7'

# --- Beginning of Assignment ---

# defining model functions


def f(x, a, b):
    return a*x + b


def g(x, a, b):
    return b * np.exp(a * x)


# reading files in
cesium = np.loadtxt("Cesium-Barium_24012022.txt", dtype='float', skiprows=2,
                    delimiter='\t', unpack=True)
background = np.loadtxt("Background_20min_24012022.txt", dtype='float',
                        skiprows=2, delimiter='\t', unpack=True)

# subtracting the mean background radiation from the data
mean = np.mean(background[1])

data = cesium[1] - mean

# calculating std for each data point
standard = np.sqrt(cesium[1] + mean)

# error after transformation
error = abs(standard/data)
print("The uncertainty in the data is", np.mean(error))

# converting count data to rates

delta_t = 20  # seconds
count_rates = cesium[0] * delta_t

# performing linear regression on (x_i, log(y_i)) using f
# error on log(y_i)

popt, pcov = curve_fit(f, count_rates, np.log(data), absolute_sigma=True,
                       sigma=error, p0=[-1, 1])
print("a (linear) =", popt[0], "\nb (linear) =", popt[1])

# performing nonlinear regression on (x_i, y_i) using g

popt_two, pcov_two = curve_fit(g, count_rates, data, absolute_sigma=True,
                               sigma=standard, p0=[-0.01, 2000])
print("a (nonlinear) =", popt_two[0], "\nb (nonlinear) =", popt_two[1])


# outputting half-life values
hl1 = np.log(2)/popt[0]
print("The half-life calculated from linear curve fit parameters is:", hl1)

hl2 = np.log(2)/popt_two[0]
print("The half-life calculated from nonlinear curve fit parameters is:", hl2)


# plotting the errorbars and curves of best fit
tao = 156  # seconds. or 2.6 minutes
decay = np.log(2)/tao


# defining theoretical curve and applying curve fit
def theo(t, initial):
    return initial * np.exp(-t*decay)


popt_three, pcov_three = curve_fit(theo, count_rates, data,
                                   absolute_sigma=True, sigma=standard)

plt.figure(1)
plt.errorbar(count_rates, cesium[1], yerr=standard*2,
             linestyle='', marker='o', label='Measured data')
plt.plot(count_rates, np.exp(f(count_rates, popt[0], popt[1])),
         label='log(N) = ax + log(b)')
plt.plot(count_rates, g(count_rates, popt_two[0], popt_two[1]),
         label='N = b(e^ax)')
plt.plot(count_rates, theo(count_rates, popt_three), label='Theoretical')
plt.xlabel("Time (s)")
plt.ylabel("Number of Counts (count/sec)")
plt.title('Number of Counts vs. Time')
plt.legend()
plt.savefig("non-log e2.png")
plt.show()

# second plot but now with log axis
plt.figure(2)
plt.errorbar(count_rates, cesium[1], yerr=standard*2,
             linestyle='', marker='o', label='Measured data')
plt.plot(count_rates, np.exp(f(count_rates, popt[0], popt[1])),
         label='log(N) = ax + log(b)')
plt.plot(count_rates, g(count_rates, popt_two[0], popt_two[1]),
         label='N = b(e^ax)')
plt.plot(count_rates, theo(count_rates, popt_three), label='Theoretical')
pylab.yscale('log')
plt.xlabel('Time (s)')
plt.ylabel('log(Number of Counts) (count/second)')
plt.title('log(Number of Counts) vs. Time')
plt.legend()
plt.savefig("log e2.png")
plt.show()

# the nonlinear regression method gives a half-life closer to the expected
# half-life of 2.6 minutes. It is noticeable on the plot.

# calculating standard deviation of linear regressed half-life
a1_var = pcov[0, 0]
a1_std = np.sqrt(a1_var)
hl1_error = a1_std/(popt[0]**2)
print("The half-life calculated from the linear curve fit parameters with "
      "uncertainty is:", hl1, "+/-", hl1_error)

a2_var = pcov_two[0, 0]
a2_std = np.sqrt(a2_var)
hl2_error = a2_std/popt_two[0]**2
print("The half-life calculated from nonlinear curve fit parameters with "
      "uncertainty is:", hl2, "+/-", hl2_error)

# from the above, it is clear that the nominal half-life does not fall into
# the range.


# defining a function to calculate chi^2_red
def chi2r(measured, predicted, errors, num_of_parameters):
    return np.sum((measured-predicted)**2/errors**2) / \
           (measured.size - num_of_parameters)


print('Reduced Chi Squared of Linear Curve Fit is:',
      chi2r(np.log(data), f(count_rates, popt[0], popt[1]), error, 2))
print('Reduced Chi Squared of Nonlinear Curve Fit is:',
      chi2r(data, g(count_rates, popt_two[0], popt_two[1]), standard, 2))
print('Reduced Chi Squared of Theoretical Curve Fit is:',
      chi2r(data, theo(count_rates, popt_three), standard, 1))
