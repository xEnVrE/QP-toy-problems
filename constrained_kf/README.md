# Constrained Kalman filtering

## Requirements
- [qpsolvers](https://github.com/stephane-caron/qpsolvers)
- `matplotlib`
- `numpy`

## Theory

Given a particle $[x, y]$ moving in 2D space along a circle of radius $r$, we want to estimate its state

$$
s = 
\begin{bmatrix}
x & \dot{x} & y & \dot{y}
\end{bmatrix}
$$

given noisy measurements

$$
z =
\begin{bmatrix}
x & y
\end{bmatrix} + \nu = 
H s + \nu
$$

with $\nu$ the measurement noise.

A constant velocity model is assumed for the particle. It is represented by the following model:

$$
s_{k} = F s_{k-1} + w
$$

with $w$ the process noise.

The estimation is obtained using a **constrained linear Kalman filter** implemented as follows.

### Prediction step

The standard KF prediction step:

$$
s_{k}^{-} = F s_{k-1},
$$

$$
P_{k}^{-} = F P_{k-1} F^{T} + Q
$$

where:
- $P$ is used to indicate the state covariance
- $Q$ is the process covariance noise associated to $w$

### Correction step via QP programming

The covariance correction step is the standard one:

$$
S = H P_{k}^{-} H^{T} + R,
$$

$$
K = P_{k}^{-} H^{T} S^{-1},
$$

$$
P_{k} = (I - KH) P_{k}^{-} (I - KH) + K R K^{T}.
$$

The state correction step is implemented using optimization as follows:

$$
\begin{split}
\begin{array}{ll}
s_{k} = 
\underset{s}{\mbox{min}}
    & (s_{k} - s_{k}^{-})^T (P_{k}^{-})^{-1} (s_{k} - s_{k}^{-}) + (z - Hs)^{T} R^{-1} (z - Hs) \\
\mbox{subject to}
    & g(s) = x^{2} + y^{2} - r^{2} = 0 \\
\end{array}
\end{split}
$$

The above program can be solved via QP programming:

$$
\begin{split}
\begin{array}{ll}
s_{k} = 
\underset{s}{\mbox{min}}
    & \frac{1}{2} s^T P_{qp} s + q^T s \\
\mbox{subject to}
    & A s = b \\
\end{array}
\end{split}
$$

by choosing the above matrices and vectors as follows:

$$
P_{qp} = (P_{k}^{-})^{-1} + H^{T} R^{-1} H
$$

$$
q = -(P_{k}^{-})^{-1} x_{k}^{-} - H^{T}R^{-1}z
$$

The constraint is implemented using the first-order differential law:

$$
\frac{\partial{g}}{\partial s} \dot{s} = -\lambda_{g} g(s),
$$

with $\lambda_{g}$ a tunable gain, which can be rewritten as $A s = b$ by choosing:

$$
A = 
\begin{bmatrix}
0 & 2x & 0 & 2y
\end{bmatrix},
$$

$$
b = - \lambda_{g} g(s).
$$

## Run as

```console
python constrained_kf/circular_tracking.py
```

The expected outcome is:

<img src="https://github.com/xEnVrE/QP-toy-problems/blob/master/constrained_kf/assets/example.png" width=1000></img>

The above plot shows the outcome of the filtering in terms of particle trajectory and estimation error when the constraint is enabled (central plot) or not (leftmost plot).
