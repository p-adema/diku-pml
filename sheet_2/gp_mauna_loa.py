import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.optimize as opt
import scipy.spatial


def gaussian_kernel(X, Xprime, gamma=2):
    dists = scipy.spatial.distance.cdist(X, Xprime, metric="sqeuclidean")
    return np.exp(-gamma * dists)


def special_kernel(X, Xprime, eta):
    a = eta[0]
    b = eta[1]
    K = (1 + X @ Xprime.T) ** 2 + a * np.multiply.outer(
        np.sin(2 * np.pi * X.reshape(-1) + b),
        np.sin(2 * np.pi * Xprime.reshape(-1) + b),
    )
    return K


# load and normalize Mauna Loa data
data = np.genfromtxt("co2_mm_mlo.csv", delimiter=",")
# 10 years of data for learning
X = data[:120, 2] - 1958
y_raw = data[:120, 3]
y_mean = np.mean(y_raw)
y_std = np.sqrt(np.var(y_raw))
y = (y_raw - y_mean) / y_std
# the next 5 years for prediction
X_predict = data[120:180, 2] - 1958
y_predict = data[120:180, 3]


# B) todo: implement this
def negLogLikelihood(params, kernel):
    noise_y = params[0]
    eta = params[1:]
    # todo: calculate the negative loglikelihood (See section 6.3 in the lecture notes)
    return 0.0  # todo: return the negative loglikelihood


def optimize_params(ranges, kernel, Ngrid):
    opt_params = opt.brute(
        lambda params: negLogLikelihood(params, kernel), ranges, Ns=Ngrid, finish=None
    )
    noise_var = opt_params[0]
    eta = opt_params[1:]
    return noise_var, eta


# B) todo: implement the posterior distribution, i.e. the distribution of f^star
def conditional(X, y, noise_var, eta, kernel):
    # todo: Write the function...
    # See eq. 66 in the lecture notes. Note that there is a small error: Instead of (S) it should be K(S)
    return mustar, Sigmastar  # return mean and covariance matrix


# C) todo: adapt this
kernel = gaussian_kernel  # todo: change to new kernel
ranges = ((1.0e-4, 10), (1.0e-4, 10))  # todo: change to the new parameters

Ngrid = 10
noise_var, eta = optimize_params(ranges, kernel, Ngrid)
print("optimal params:", noise_var, eta)

# B) todo: use the learned GP to predict on the observations at X_predict
prediction_mean_gp, Sigma_gp = conditional("""TODO: Inset input """)
var_gp = np.diag(
    Sigma_gp
)  # We only need the diagonal term of the covariance matrix for the plots.

# plotting code for your convenience
plt.figure(dpi=400, figsize=(6, 3))
plt.plot(X + 1958, y_raw, color="blue", label="training data")
plt.plot(X_predict + 1958, y_predict, color="red", label="test data")
yout_m = prediction_mean_gp * y_std + y_mean
yout_v = var_gp * y_std**2
plt.plot(X_predict + 1958, yout_m, color="black", label="gp prediction")
plt.plot(
    X_predict + 1958, yout_m + 1.96 * yout_v**0.5, color="grey", label="GP uncertainty"
)
plt.plot(X_predict + 1958, yout_m - 1.96 * yout_v**0.5, color="grey")
plt.xlabel("year")
plt.ylabel("co2(ppm)")
plt.legend()
plt.tight_layout()
