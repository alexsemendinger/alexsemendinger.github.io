# Appendix

I sometimes find it helpful to write out tiny examples to see all the operations at once, in a handful rather than hundreds of dimensions. 

## One-layer toy example

To make this more concrete, let's run through a tiny example of each of these operations. Our dimensions will be the following:
* $$n = 3$$
* $$d_\text{vocab} = 10$$
* $$d_\text{model} = 5$$
* $$d_\text{head} = 2$$
* $$d_\text{mlp} = 10$$

We start with one-hot encodings of our tokens:

$$$
\begin{align*}
t_1^\top &= [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] \\
t_2^\top &= [0, 1, 0, 0, 0, 0, 0, 0, 0, 0] \\
t_3^\top &= [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
\end{align*}
$$$

Our embedding matrix is

$$$
\begin{align*}
W_E = \begin{bmatrix}
1 & -1 & 0 & 1 & -1 & 0 & 1 & -1 & 0 & 1\\
-1 & 0 & 1 & -1 & 0 & 1 & -1 & 0 & 1 & -1\\
0 & 1 & -1 & 0 & 1 & -1 & 0 & 1 & -1 & 0\\
1 & 0 & -1 & 1 & 0 & -1 & 1 & 0 & -1 & 1\\
-1 & 1 & 0 & -1 & 1 & 0 & -1 & 1 & 0 & -1
\end{bmatrix}
\end{align*}
$$$

Applying the embedding matrix, we get:

$$$
\begin{align*}
x^{(0)}_1 &= W_E t_1 = [1, -1, 0, 1, -1] \\
x^{(0)}_2 &= W_E t_2 = [-1, 0, 1, 0, 1] \\
x^{(0)}_3 &= W_E t_3 = [0, 1, -1, -1, 0]
\end{align*}
$$$

Next, we compute the values, keys, and queries using the matrices \(W_V\), \(W_K\), and \(W_Q\):

$$$
\begin{align*}
W_V &= \begin{bmatrix}
0.1 & 0.2 & 0.3 & 0.4 & 0.5 \\
0.2 & 0.3 & 0.4 & 0.5 & 0.6
\end{bmatrix} \\
W_K &= \begin{bmatrix}
0.1 & 0.2 & 0.3 & 0.4 & 0.5 \\
0.2 & 0.3 & 0.4 & 0.5 & 0.6
\end{bmatrix} \\
W_Q &= \begin{bmatrix}
0.1 & 0.2 & 0.3 & 0.4 & 0.5 \\
0.2 & 0.3 & 0.4 & 0.5 & 0.6
\end{bmatrix}
\end{align*}
$$$

We compute the values, keys, and query for the tokens:

$$$
\begin{align*}
v_1 &= W_V x^{(0)}_1 = [0.1, 0.2] \cdot [1, -1, 0, 1, -1] = [-0.1, -0.1] \\
v_2 &= W_V x^{(0)}_2 = [0.1, 0.2] \cdot [-1, 0, 1, 0, 1] = [0.1, 0.4] \\
v_3 &= W_V x^{(0)}_3 = [0.1, 0.2] \cdot [0, 1, -1, -1, 0] = [-0.1, -0.2]
\end{align*}
$$$

$$$
\begin{align*}
k_1 &= W_K x^{(0)}_1 = [0.1, 0.2] \cdot [1, -1, 0, 1, -1] = [-0.1, -0.1] \\
k_2 &= W_K x^{(0)}_2 = [0.1, 0.2] \cdot [-1, 0, 1, 0, 1] = [0.1, 0.4] \\
k_3 &= W_K x^{(0)}_3 = [0.1, 0.2] \cdot [0, 1, -1, -1, 0] = [-0.1, -0.2]
\end{align*}
$$$

$$$
q_3 = W_Q x^{(0)}_3 = [0.1, 0.2] \cdot [0, 1, -1, -1, 0] = [-0.1, -0.2]
$$$

Next, we compute the attention scores and weights:

$$$
\begin{align*}
a_1 &= \frac{q_3^\top k_1}{\sqrt{d_\text{head}}} = \frac{-0.1 \cdot -0.1 + -0.2 \cdot -0.1}{\sqrt{2}} = \frac{0.01 + 0.02}{\sqrt{2}} = \frac{0.03}{\sqrt{2}} \approx 0.021 \\
a_2 &= \frac{q_3^\top k_2}{\sqrt{d_\text{head}}} = \frac{-0.1 \cdot 0.1 + -0.2 \cdot 0.4}{\sqrt{2}} = \frac{-0.01 + -0.08}{\sqrt{2}} = \frac{-0.09}{\sqrt{2}} \approx -0.064 \\
a_3 &= \frac{q_3^\top k_3}{\sqrt{d_\text{head}}} = \frac{-0.1 \cdot -0.1 + -0.2 \cdot -0.2}{\sqrt{2}} = \frac{0.01 + 0.04}{\sqrt{2}} = \frac{0.05}{\sqrt{2}} \approx 0.035
\end{align*}
$$$

$$$
\begin{align*}
w_1 &= \frac{e^{a_1}}{e^{a_1} + e^{a_2} + e^{a_3}} = \frac{e^{0.021}}{e^{0.021} + e^{-0.064} + e^{0.035}} \approx 0.34 \\
w_2 &= \frac{e^{a_2}}{e^{a_1} + e^{a_2} + e^{a_3}} = \frac{e^{-0.064}}{e^{0.021} + e^{-0.064} + e^{0.035}} \approx 0.31 \\
w_3 &= \frac{e^{a_3}}{e^{a_1} + e^{a_2} + e^{a_3}} = \frac{e^{0.035}}{e^{0.021} + e^{-0.064} + e^{0.035}} \approx 0.35
\end{align*}
$$$
$$$
\begin{align*}
r_3 &= w_1 v_1 + w_2 v_2 + w_3 v_3 \\
&= 0.34 \cdot [-0.1, -0.1] + 0.31 \cdot [0.1, 0.4] + 0.35 \cdot [-0.1, -0.2] \\
&= [-0.038, 0.02]
\end{align*}
$$$

Finally, we project this result back to the residual stream using \(W_O\):

$$$
\begin{align*}
W_O &= \begin{bmatrix}
0.1 & 0.2 \\
0.2 & 0.3 \\
0.3 & 0.4 \\
0.4 & 0.5 \\
0.5 & 0.6
\end{bmatrix} \\
o_3 &= W_O r_3 = \begin{bmatrix}
0.1 & 0.2 \\
0.2 & 0.3 \\
0.3 & 0.4 \\
0.4 & 0.5 \\
0.5 & 0.6
\end{bmatrix} \cdot \begin{bmatrix}
-0.038 \\
0.02
\end{bmatrix} = \begin{bmatrix}
-0.0038 \\
-0.0056 \\
-0.0074 \\
-0.0092 \\
-0.011
\end{bmatrix}
\end{align*}
$$$
Now we add this to \(x_3^{(0)}\) to get \(x_3^{(1)}\):

$$
x_3^{(1)} = x_3^{(0)} + o_3 = [0, 1, -1, -1, 0] + [-0.0038, -0.0056, -0.0074, -0.0092, -0.011] = [-0.0038, 0.9944, -1.0074, -1.0092, -0.011]
$$