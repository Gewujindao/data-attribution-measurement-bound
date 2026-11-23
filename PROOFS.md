# Proofs

## Notation

- Training is a deterministic iterative process (fixed initialization and order); training points carry weights $`w\in\mathbb{R}^N`$. The exact attribution matrix $`Y\in\mathbb{R}^{k\times N}`$, $`Y_{ji}=\partial f_j(\theta_T(w))/\partial w_i`$ (the meta-gradient through the whole trajectory).
- LDS: random deletion subsets $`S`$ of fixed size $`n`$. The prediction of the estimated row $`\hat{y}_j`$ is $`q_j(S)=-\sum_{i\in S}\hat{y}_{ji}`$, the prediction of the exact row is $`p_j(S)`$,
  the measured response is $`a_j(S)=f_j(\text{retrain with } S \text{ deleted})-f_j(\text{full data})`$.
  $`\mathrm{LDS}_j=\mathrm{Spearman}_S(q_j,a_j)`$ (the metric used in practice); $`\mathrm{LDS}_{P,j}`$ is the Pearson version of the same. $`L^*_j`$ is the LDS of the exact row.
- $`R`$: each row of $`Y`$ minus its row mean, divided by its norm (row means and norms do not affect the LDS). $`\rho_j`$: the centered cosine between $`\hat{y}_j`$ and $`y_j`$. $`F(B)=(1/k)\sum_{i\le B}\sigma_i(R)^2`$.
- Measurement class $`\mathcal{M}_B`$: $`B_L`$ row measurements (one backward replay, test function $`\sum_j z_j f_j`$, answering $`z^\top Y\in\mathbb{R}^N`$) and $`B_R`$ column measurements (one forward replay, direction $`u`$, answering $`Yu\in\mathbb{R}^k`$), $`B_L+B_R\le B`$;
  they may be adaptive and randomized; the norm and mean of every row are given for free; no information about $`Y`$ other than these answers is used (no prior). The output is one estimate $`\hat{y}_j`$ per row.

The exact meta-gradient row of one test point $`y\in\mathbb{R}^N`$; delete a random subset $`S`$ of fixed size ($`|S|=n`$, mask $`x=\mathbf{1}_S`$), $`\gamma:=(n-1)/(N-2)`$ (at 1% deletion with $`N=1000`$: 0.009; at 10%: 0.099).

Linear prediction $`p(S)=-y^\top x`$; measured response $`a(S)=f(\text{retrain with } S \text{ deleted})-f(\text{full data})`$. $`g`$: the best linear datamodel of $`a`$ on the mask, $`a\approx c_0-g^\top x`$ (since $`\sum x=n`$ is fixed, $`g`$ is determined only up to a constant vector; only centered inner products are used below).

Projection-type estimates: $`W\subset\mathbf{1}^\perp`$ (dim $`B`$), $`u=P_W y`$, $`v=P_\perp y`$ ($`P_\perp`$ projects onto $`W^\perp\cap\mathbf{1}^\perp`$), $`y_c=y-\bar{y}\mathbf{1}=u+v`$, $`\rho=\lVert u\rVert/\lVert y_c\rVert`$.

```math
\begin{aligned}
&\beta_W=\frac{\langle u,g\rangle}{\lVert u\rVert^2},\qquad \beta_\perp=\frac{\langle v,g\rangle}{\lVert v\rVert^2},\\
&\text{total slope } \beta=\frac{\langle y_c,g\rangle}{\lVert y_c\rVert^2}=\rho^2\beta_W+(1-\rho^2)\beta_\perp;\qquad L_P=\mathrm{corr}(p,a).
\end{aligned}
```

$`\pi^W_i:=u_i y_{c,i}/\lVert u\rVert^2`$, $`\pi^\perp_i:=v_i y_{c,i}/\lVert v\rVert^2`$: two sets of (possibly signed) weights, each summing to 1 (because $`\langle u,v\rangle=0`$).

Single-point multiplier $`m_{ji}`$ = leave-one-out effect / meta-gradient; memorization score $`\mathrm{mem}_i`$ = how much point $`i`$'s own margin drops after it is deleted.

## Theorem A (cosine ceiling)

**Status:** proved

**Statement.** Let $`W`$ be the sum of "the span of the row-measurement answers with their component along $`\mathbf{1}`$ removed" and "the span of the projections of the column-measurement directions onto $`\mathbf{1}^\perp`$"; $`\dim W\le B`$, $`d=N-1-\dim W\ge 2`$.

For any orthogonal transformation $`Q`$ on $`W^\perp\cap\mathbf{1}^\perp`$, let $`r_j^Q=P_W r_j+Q\cdot P_\perp r_j`$ ($`P_\perp`$ is the projection onto $`W^\perp\cap\mathbf{1}^\perp`$), and let $`\mathcal{F}=\{R^Q\}`$ be the family.

(i) every member of $`\mathcal{F}`$ still has unit-length, centered rows, and gives exactly the same answers as $`R`$ to all measurements (including adaptively chosen ones);

(ii) averaging over $`Q`$ under the Haar measure, the output of any $`\mathcal{M}_B`$ method satisfies

```math
\begin{aligned}
\mathbb{E}_Q[\rho_j] &\le \lVert P_W r_j\rVert,\\
\mathbb{E}_Q[\rho_j^2] &\le \max\left(\lVert P_W r_j\rVert^2,\ \frac{1-\lVert P_W r_j\rVert^2}{d}\right);
\end{aligned}
```

(iii) hence for any weights $`w_j\ge 0`$ there exists a world $`R^Q`$ compatible with all answers such that

```math
\begin{aligned}
\frac{1}{k}\sum_j w_j\rho_j &\le \Phi_w(B):=\max_{\dim W=B,\ W\subset\mathbf{1}^\perp}\frac{1}{k}\sum_j w_j\lVert P_W r_j\rVert\\
&\le \sqrt{F(B)}\cdot\sqrt{\mathrm{mean}_j\,w_j^2},
\end{aligned}
```

and

```math
\mathrm{mean}_j\,\rho_j^2 \le F(B)+1/d;
```

(iv) equality: the SVD oracle ($`W`$ taken as the top $`B`$ right singular subspace, estimate $`P_W r_j`$) gives mean $`\rho^2=F(B)`$; the optimal subspace for $`\Phi_w`$ gives mean $`w\rho=\Phi_w(B)`$. Both require knowing $`R`$; they are oracles.

**Proof.** (i) $`\mathbf{1}^\perp=W\oplus(W^\perp\cap\mathbf{1}^\perp)`$ and $`r_j\in\mathbf{1}^\perp`$, so $`\lVert r_j^Q\rVert^2=\lVert P_W r_j\rVert^2+\lVert P_\perp r_j\rVert^2=1`$, and $`r_j^Q\perp\mathbf{1}`$.

The answer to a row measurement $`z`$ is $`v^\top+(z\cdot\mu)\mathbf{1}^\top`$, $`v=\sum_j z_j n_j r_j\in W`$ (by the definition of $`W`$), and $`\sum_j z_j n_j r_j^Q=P_W v+Q P_\perp v=v`$, so the answer is unchanged.

Column measurement $`u`$:

```math
\begin{aligned}
r_j^Q\cdot u&=r_j^Q\cdot u' \quad (u'=P_{\mathbf{1}^\perp}u\in W),\\
&=P_W r_j\cdot u'+(QP_\perp r_j)\cdot u'=r_j\cdot u',
\end{aligned}
```

because $`QP_\perp r_j\in W^\perp`$.

Adaptivity: run the method once on $`R`$ to obtain $`W`$; for any $`R^Q`$ the first query is the same, and if the first $`t`$ answers are the same then query $`t+1`$ is the same; by induction the whole sequence of queries and answers is the same. For a randomized method this holds for every seed, and hence also in expectation.

(ii) The method's output $`x_j`$ (without loss of generality centered and of unit length; centering does not change the centered cosine) is the same for all members of $`\mathcal{F}`$. $`\rho_j(Q)=x_j\cdot P_W r_j+(P_\perp x_j)\cdot Q(P_\perp r_j)`$.

Under the Haar measure $`Q`$ and $`-Q`$ have the same distribution, $`\mathbb{E}[Q]=0`$, so $`\mathbb{E}\rho_j=x_j\cdot P_W r_j\le\lVert P_W r_j\rVert`$. $`Qw`$ is uniform on the $`d`$-dimensional sphere, $`\mathbb{E}[((P_\perp x)\cdot Qw)^2]=\lVert P_\perp x\rVert^2\lVert w\rVert^2/d`$, and the cross term is 0:

```math
\begin{aligned}
\mathbb{E}\rho_j^2&=(x_j\cdot P_W r_j)^2+\lVert P_\perp x_j\rVert^2\frac{1-\lVert P_W r_j\rVert^2}{d}\\
&\le t\lVert P_W r_j\rVert^2+(1-t)\frac{1-\lVert P_W r_j\rVert^2}{d}\quad (t=\lVert P_W x_j\rVert^2),
\end{aligned}
```

which is linear in $`t`$, so the maximum is attained at an endpoint.

(iii) The average does not exceed the bound, so some $`Q`$ satisfies it; take the maximum over $`W`$; Cauchy–Schwarz and Ky Fan ($`\max_W \sum\lVert P_W r_j\rVert^2=\sum_{i\le B}\sigma_i^2`$) give the $`\sqrt{F}`$ step. ∎

## Auxiliary proposition (covariance on fixed-size subsets)

**Status:** proved

When the deletion set has fixed size, $`\mathrm{Cov}(a\cdot m, b\cdot m) \propto \sum_i (a_i-\bar{a})(b_i-\bar{b})`$, so the correlation between the estimated row and the exact row over random subsets is exactly the centered cosine $`\rho_j`$ [proved].

## Lemma B (identity for the Pearson version of LDS)

**Status:** proved (algebra)

**Statement.** For any estimated row ($`q`$ is a linear function of the deletion mask), over the distribution of random fixed-size subsets:

```math
\begin{aligned}
&\mathrm{LDS}_{P,j} = \rho_j\cdot L^*_{P,j} + \sqrt{1-\rho_j^2}\cdot\sqrt{1-(L^*_{P,j})^2}\cdot\kappa_j,\\
&\kappa_j = \text{the partial correlation of } q \text{ and } a \text{ given } p.
\end{aligned}
```

Moreover $`|\kappa_j| \le \sqrt{(L_{\mathrm{dm},j}^2-(L^*_{P,j})^2)/(1-(L^*_{P,j})^2)}`$, where $`L_{\mathrm{dm},j}`$ is the correlation of $`a_j`$ with its best linear prediction (the "datamodel"). Hence

```math
\begin{aligned}
\rho L^*_P - \sqrt{1-\rho^2}\sqrt{L_{\mathrm{dm}}^2-(L^*_P)^2} \le \mathrm{LDS}_P &\le \rho L^*_P + \sqrt{1-\rho^2}\sqrt{L_{\mathrm{dm}}^2-(L^*_P)^2}\\
&= L_{\mathrm{dm}}\cdot\cos\left(\arccos\rho - \arccos(L^*_P/L_{\mathrm{dm}})\right).
\end{aligned}
```

**Proof.** The correlations of three random variables satisfy

```math
r_{qa}=r_{qp}r_{pa}+\sqrt{1-r_{qp}^2}\sqrt{1-r_{pa}^2}\cdot r_{qa\cdot p}
```

(a rearrangement of the definition of the partial correlation); $`r_{qp}=\rho_j`$ is the auxiliary proposition (on fixed-size subsets the covariance of two linear predictions is proportional to their centered inner product).

The residual $`q^{\perp}`$ of $`q`$ after removing the $`p`$ direction is still a linear function of the mask, and the difference between $`a`$ and its best linear prediction $`a_{\mathrm{lin}}`$ is uncorrelated with every linear function,
so $`\mathrm{cov}(q^{\perp},a)=\mathrm{cov}(q^{\perp},a_{\mathrm{lin}}-\beta p)`$, $`|\mathrm{cov}|\le\mathrm{sd}(q^{\perp})\cdot\mathrm{sd}(a)\sqrt{L_{\mathrm{dm}}^2-(L^*_P)^2}`$; and $`\mathrm{sd}(a-\beta p)=\mathrm{sd}(a)\sqrt{1-(L^*_P)^2}`$. ∎
On a finite set of $`M`$ subsets, the same identity holds exactly for the sample correlations.

**Corollary.** (a) $`\mathrm{LDS}_{P,j} \le \rho_j L^*_{P,j}`$ if and only if $`\kappa_j\le 0`$ (necessary and sufficient, i.e. the weakest condition);

(b) if the exact meta-gradient row is, up to a multiple, the best linear datamodel ($`L_{\mathrm{dm}}=L^*_P`$), then $`\mathrm{LDS}_P=\rho L^*_P`$ holds with equality for every estimate;

(c) in general there is no constant $`c`$ such that $`\mathrm{LDS}_P\le c\cdot\rho\cdot L^*_P`$:

Counterexample (`ceiling_counterexamples.py` (1a)): true response = a linear function of the mask (datamodel $`g=y+0.8h`$, $`h\perp y`$) plus noise. The exact row has $`L^*_P=0.757`$; the estimate $`h`$ gives $`\rho=0.00`$ but $`\mathrm{LDS}_P=0.572`$; the estimate $`y+0.8h`$ gives $`\rho=0.79`$ but $`\mathrm{LDS}_P=0.958`$, higher than the exact row itself.

Meaning of condition (a): the error of the estimate must not carry precisely "the part of the linear response that the exact meta-gradient misses".

## Lemma C (switching to Spearman)

**Status:** proved under the stated condition; false without it

**Statement.** Condition S: there is an increasing, convex link function $`g`$ with $`g(0)=0`$, $`g(1)=1`$ such that both the estimated row and the exact row satisfy $`\mathrm{LDS}_S=g(\mathrm{LDS}_P)`$.

When $`(q,a)`$ and $`(p,a)`$ are bivariate normal, $`g(r)=(6/\pi)\arcsin(r/2)`$ (Pearson's classical formula), which satisfies S.

Under S and $`\kappa_j\le 0`$:

```math
\mathrm{LDS}_{S,j}=g(\mathrm{LDS}_{P,j}) \le g(\rho_j L^*_{P,j}) \le \rho_j\, g(L^*_{P,j}) = \rho_j L^*_{S,j}
```

(the second inequality follows from convexity and $`g(0)=0`$). When $`g`$ is strictly convex, equality holds if and only if $`\kappa_j=0`$ and $`\rho_j\in\{0,1\}`$.

Under normality with $`\kappa=0`$ there is also a lower bound: $`g(r)/r\in[3/\pi,1]`$, so

```math
0.955\cdot\rho L^*_S \le \mathrm{LDS}_S \le \rho L^*_S.
```

Unconditional counterexample (`ceiling_counterexamples.py` (1b), true response exactly linear, $`L^*=1`$, $`\kappa\equiv 0`$): one element $`H`$ in a row dominates the variance (heavy tail).

$`H=300`$: the estimate that removes the large element has $`\rho=0.10`$, $`\mathrm{LDS}_S=0.98`$ (ratio 9.6); the estimate that keeps only the large element has $`\rho=0.995`$, $`\mathrm{LDS}_S=0.17`$ (ratio 0.17).

That is, without a distributional condition, $`\rho`$ does not control the Spearman version of LDS in either direction.

## Sign condition (sign κ = sign(β_W − β_⊥))

**Status:** proved (algebra)

Notation as before: $`y`$ is the exact row of a test point; $`p`$, $`q`$, $`a`$ are respectively the predictions/response of the exact row, the estimate and the measurement on random deletion subsets;
$`g`$ is the best linear datamodel of the measured response (the least-squares linear part of $`a`$ on the deletion mask); $`\langle\cdot,\cdot\rangle`$ is the centered inner product over training points. $`\kappa=\text{partial correlation}(q,a\mid p)`$.

For estimates that "project $`y`$ onto some subspace $`W`$" (the SVD oracle, low-rank reconstructions such as SPELL, and leave-one-out projections are all of this type), the residual vector of $`q`$ after removing the $`p`$ direction is

```math
q^{\perp} = \frac{\lVert P_\perp y\rVert^2\cdot P_W y - \lVert P_W y\rVert^2\cdot P_\perp y}{\lVert y\rVert^2},
```

the part of $`a`$ orthogonal to all linear functions contributes no covariance, so

```math
\mathrm{cov}(q^{\perp},a) \propto \langle q^{\perp},g\rangle = \frac{\lVert P_W y\rVert^2\lVert P_\perp y\rVert^2(\beta_W-\beta_\perp)}{\lVert y\rVert^2},
```

where $`\beta_W = \langle P_W y, g\rangle/\lVert P_W y\rVert^2`$ (the amplification of the measured response along the "retained component"), $`\beta_\perp = \langle P_\perp y, g\rangle/\lVert P_\perp y\rVert^2`$ (the amplification along the "discarded component").

**Conclusion** [proved]: $`\mathrm{sign}\,\kappa_j = \mathrm{sign}(\beta_W - \beta_\perp)`$. $`\beta_W`$ and $`\beta_\perp`$ are exactly the regression slopes of $`a`$ on the two linear predictions $`q_W`$, $`q_\perp`$ (which are orthogonal over the subset distribution), so they can be estimated directly from retraining samples.

The total amplification $`\beta=\mathrm{cov}(a,p)/\mathrm{var}(p)`$ is their energy-weighted average; whether the "measured/predicted slope" is greater than 1 or less than 1 is unrelated to the sign of $`\kappa`$.

## Proposition 1 (exact decomposition of κ)

**Status:** proved

```math
\begin{aligned}
\kappa &= (\beta_W-\beta_\perp)\cdot\frac{\rho\sqrt{1-\rho^2}\cdot\mathrm{sd}(p)}{\mathrm{sd}(a)\sqrt{1-L_P^2}}\\
&= -\Delta\cdot\frac{\rho\sqrt{1-\rho^2}\cdot L_P}{\sqrt{1-L_P^2}},\qquad \Delta:=\frac{\beta_\perp-\beta_W}{\beta}.
\end{aligned}
```

**Proof.** On fixed-size subsets $`\mathrm{cov}(z_1^\top x,z_2^\top x)=c\cdot\langle z_1-\bar{z}_1,z_2-\bar{z}_2\rangle`$ ($`c=n(N-n)/(N(N-1))`$). $`q_W=-u^\top x`$, $`q_\perp=-v^\top x`$, $`p=q_W+q_\perp+\text{constant}`$, $`\mathrm{cov}(q_W,q_\perp)=c\langle u,v\rangle=0`$.

Write $`a=\beta_W q_W+\beta_\perp q_\perp+\varepsilon`$ ($`\varepsilon`$ uncorrelated with $`q_W`$ and $`q_\perp`$). The estimate is $`q\propto q_W`$, $`\kappa=\mathrm{corr}(q_W-\rho^2 p,\ a-\beta p)`$:

```math
\begin{aligned}
\mathrm{cov}(q_W-\rho^2 p,\ a)&=c\left[(1-\rho^2)\beta_W\lVert u\rVert^2-\rho^2\beta_\perp\lVert v\rVert^2\right]=c\lVert u\rVert^2\lVert v\rVert^2(\beta_W-\beta_\perp)/\lVert y_c\rVert^2;\\
\mathrm{var}(q_W-\rho^2 p)&=c\left[(1-\rho^2)^2\lVert u\rVert^2+\rho^4\lVert v\rVert^2\right]=c\lVert u\rVert^2\lVert v\rVert^2/\lVert y_c\rVert^2;\\
\mathrm{var}(a-\beta p)&=\mathrm{var}(a)(1-L_P^2).
\end{aligned}
```

Dividing and using $`\mathrm{sd}(p)=\sqrt{c}\,\lVert y_c\rVert`$, $`L_P=\beta\cdot\mathrm{sd}(p)/\mathrm{sd}(a)`$ gives the result. ∎

## Second-order structure: self terms and pairwise terms

**Status:** proved (algebra)

Every set function has the Möbius decomposition

```math
a(S)=\sum_{\emptyset\neq T\subseteq S} \mu(T):
```

$`\mu(\{i\})=A_i:=f(\mathbf{1}-e_i)-f(\mathbf{1})`$ (leave-one-out effect), $`\mu(\{i,l\})=I_{il}:=f(\mathbf{1}-e_i-e_l)-f(\mathbf{1}-e_i)-f(\mathbf{1}-e_l)+f(\mathbf{1})`$, and so on.

Under the fixed-size design, the projection of $`\mathbf{1}[T\subseteq S]`$ onto linear functions: for $`|T|=2`$ it is $`\gamma(x_i+x_l)+\text{constant}`$ (by symmetry and $`\sum x=n`$, coefficient $`=\mathrm{Cov}(x_i x_l,\ x_i+x_l)/\mathrm{Var}(x_i+x_l)=(n-1)/(N-2)`$); for $`|T|=3`$ the coefficient of each $`x_i`$ is $`(n-1)(n-2)/((N-2)(N-3))\approx\gamma^2`$. Hence

```math
g_i = -A_i - \gamma\cdot\sum_{l\neq i} I_{il} + O(\gamma^2\cdot\text{three-way terms}).
```

Define the single-point multiplier $`m_i:=-A_i/y_i`$ (the leave-one-out effect as a multiple of the meta-gradient; in general it varies with the test point); this gives the general expression for $`\beta`$:

```math
\begin{aligned}
\beta_W &= \sum_i m_i \pi^W_i - \gamma\frac{\langle u, I\cdot\mathbf{1}\rangle}{\lVert u\rVert^2} + O(\gamma^2),\\
\beta_\perp &= \sum_i m_i \pi^\perp_i - \gamma\frac{\langle v, I\cdot\mathbf{1}\rangle}{\lVert v\rVert^2} + O(\gamma^2).
\end{aligned}
```

In Taylor terms: of the second-order term $`\frac{1}{2}x^\top\mathbf{H}x`$ ($`\mathbf{H}`$ = meta-Hessian), the diagonal part goes into $`m_i-1=-(\frac{1}{2}\mathbf{H}_{ii}-\dots)/y_i`$ (all orders along a single coordinate), and the off-diagonal part goes into the pairwise term $`I_{il}=\mathbf{H}_{il}+O(3)`$.

## Proposition 2 (solvable case (a): kernel ridge regression / linear least squares, trained to optimum, finite deletion)

**Status:** proved

Let $`f(x)=\langle\varphi(x),\theta\rangle`$, with objective $`\frac{1}{2}\sum w_i(\varphi_i^\top\theta-t_i)^2+\frac{1}{2}\lambda\lVert\theta\rVert^2`$. $`H=\Phi^\top\Phi+\lambda I`$, residual $`r=\Phi\theta-t`$, $`c_{ji}=\varphi_j^\top H^{-1}\varphi_i`$ (the smoothing weight of test $`j`$ on training point $`i`$), hat matrix $`K=\Phi H^{-1}\Phi^\top`$, leverage $`h_i=K_{ii}\in[0,1)`$.

(i) Meta-gradient: $`\partial\theta/\partial w_i=-H^{-1}\varphi_i r_i`$, so $`y_{ji}=-c_{ji} r_i`$.

(ii) Exact response to deleting $`S`$: $`\theta_{-S}-\theta=H_{-S}^{-1}\Phi_S^\top r_S`$ (using $`H\theta=\Phi^\top t`$); Woodbury gives $`H_{-S}^{-1}\Phi_S^\top=H^{-1}\Phi_S^\top(I-K_{SS})^{-1}`$, hence

```math
a_j(S) = c_{jS}^\top (I-K_{SS})^{-1} r_S.
```

(iii) Single point: $`A_i=c_{ji} r_i/(1-h_i)=-y_{ji}/(1-h_i)`$, so $`m_i=1/(1-h_i)\ge 1`$, independent of the test point. It also equals "the point's leave-one-out effect on its own output / its meta-gradient on its own output" ($`y_{ii}=-h_i r_i`$).

(iv) Pairwise: write $`\alpha_i=1-h_i`$, $`e=K_{il}`$, $`\Delta_2=\alpha_i\alpha_l-e^2`$; inverting the $`2\times 2`$ matrix gives

```math
\begin{aligned}
I_{il} &= \frac{e(c_{ji} r_l+c_{jl} r_i) + e^2(c_{ji} r_i/\alpha_i + c_{jl} r_l/\alpha_l)}{\Delta_2}\\
&\approx K_{il} m_i m_l (c_{ji} r_l + c_{jl} r_i)\quad \text{(first order in } K_{il}\text{)}.
\end{aligned}
```

(v) Hence $`g_j = M y_j - \gamma\cdot I^{(j)}\mathbf{1} + O(\gamma^2)`$, $`M=\mathrm{diag}(1/(1-h_i))`$, $`\beta_W=\sum_i \pi^W_i/(1-h_i)-\gamma\langle u,I\mathbf{1}\rangle/\lVert u\rVert^2`$, and similarly for $`\beta_\perp`$.

## Correction for training duration (early stopping ≈ ridge)

**Status:** derived, spectral approximation

Correction for training duration [derived, spectral approximation]: for gradient-flow training of a linear model with squared loss for time $`T`$, the function $`\varphi_T(s)=(1-e^{-sT})/s`$ at both ends, $`s\to 0`$ and $`s\to\infty`$, equals $`1/(s+1/T)`$ (at $`sT=1`$ the difference is 26%), so finite-duration training ≈ ridge regression with $`\lambda=1/T`$ ("early stopping = ridge").

$`h_i(T)=[K(K+I/T)^{-1}]_{ii}`$ increases with $`T`$: for very short training $`h_i\approx T\cdot K_{ii}`$ is small and uniform, $`m\approx 1`$; the longer the training, the closer to 1 the leverage of points lying in small-eigenvalue directions.

## Proposition 3 (solvable case (b): linearized network + cross-entropy, finite-step gradient flow)

**Status:** (i), (iv) proved ((iv) for the self-driven model); (iii), (v), finite steps and the summary derived

(i) [proved] The training loss in the code (logsumexp − correct-class logit) is exactly $`\ell(z)=\log(1+e^{-z})`$, where $`z`$ is the margin $`l_y-\mathrm{LSE}(\text{others})`$ used by the test function. So cross-entropy is the logistic loss on the margin,

```math
-\ell'(z)=\sigma(-z)=:s(z).
```

(ii) Linearized model: margin $`z_i(\theta)=z_i^0+\varphi_i^\top(\theta-\theta_0)`$, weighted gradient flow $`\dot{z}=K W s(z)`$ ($`K_{il}=\varphi_i^\top\varphi_l`$; learning rate and $`1/N`$ absorbed into time), test margin $`z_j(T)=z_j^0+\int k_j^\top W s(z)\,du`$.

The contribution of training point $`i`$ to test $`j`$ is $`P_j(w)=\int_0^T \psi_{ji}(u)\cdot w\cdot s(z_i(u;w))\,du`$, where $`w`$ is its weight and $`\psi_{ji}(u)`$ is the transfer from "a push along $`\varphi_i`$ at time $`u`$" to $`z_j(T)`$ (the adjoint, including feedback from the other points).

Assumption L1: deleting one point does not change the transfers of the other points ($`\psi`$ is independent of $`w_i`$; the error is a second-order feedback of order $`K_{il}^2`$). Hence $`F(1)-F(0)=P_j(1)`$, $`y_{ji}=P_j'(1)`$, $`m_{ji}=P_j(1)/P_j'(1)`$.

(iii) [derived] If $`P_j`$ is concave on $`[0,1]`$ (the larger the weight, the faster the point is fitted and the more saturated its push $`w\cdot s(z_i)`$) and $`\psi_{ji}(u)`$ does not change sign, then $`P(1)=\int_0^1 P'(w)\,dw\ge P'(1)`$, i.e. $`m_{ji}\ge 1`$.

(iv) [proved, self-driven model] For a point with constant external drive $`D`$ and with $`\psi`$ constant over the relevant period: $`\dot{z}_i=D+\kappa w s(z_i)`$, $`\kappa=K_{ii}`$. Since $`\int w\cdot s(z_i)\,du=(z_i(T)-z_0-DT)/\kappa`$, $`P(w)=\psi(z_i(T;w)-z_0-DT)/\kappa`$, and $`z_i(T;0)=z_0+DT`$, so exactly

```math
m = \mathrm{mem}/\mathrm{SI},
```

$`\mathrm{mem}:=z_i(T;1)-z_i(T;0)`$ (how much the point's own margin drops when it is deleted, i.e. the "memorization score" in margin units), $`\mathrm{SI}:=\partial z_i(T)/\partial w_i`$ (its meta-gradient on its own margin).

For $`D=0`$, solving $`z+e^z=z_0+e^{z_0}+\kappa wu`$ gives $`z_i(T;w)=Z(C+\kappa wT)`$, where $`Z`$ is the inverse function of $`z+e^z`$ (increasing, concave); $`P`$ is concave, $`m\ge 1`$.

In the large-margin tail $`s\approx e^{-z}`$: $`\mathrm{mem}=\ln(1+A)`$, $`\mathrm{SI}=A/(1+A)`$, $`A=\kappa Te^{-z_0}`$, $`m=(1+A)\ln(1+A)/A=\Psi(\mathrm{mem})`$, $`\Psi(x)=x\cdot e^x/(e^x-1)`$ ($`\Psi(0)=1`$, $`\Psi(1)=1.58`$, $`\Psi(2)=2.31`$, $`\approx x`$ for large $`x`$).

[checked numerically, scipy integration] On 5 parameter sets (including $`D=0.2`$, $`1.0`$), $`m`$ and $`\mathrm{mem}/\mathrm{SI}`$ agree to 3 digits, and $`P`$ is concave on $`[0,1]`$;
$`\Psi(\mathrm{mem})`$ exceeds the exact value by 2%–17% (the exact logistic deviates from the $`e^{-z}`$ tail at small margins), and $`\mathrm{SI}`$ can be slightly larger than 1 (1.04), so $`\Psi`$ is only an approximation; $`m=\mathrm{mem}/\mathrm{SI}`$ is the exact formula.

Compare the same model with squared loss: $`m=(e^{\kappa T}-1)/(\kappa T)`$. Both losses give: $`m=1`$ if and only if there is no self-fitting; the stronger the self-fitting, the larger $`m`$; the longer the training, the larger $`m`$.

With external drive (points fitted together with their neighbors), $`s_i(u;w)`$ barely varies with $`w`$, and $`m\to 1`$.

Finite steps (discrete SGD, momentum $`\mu`$) [derived]: point $`i`$ is pushed only once per epoch, in the batch that contains it, so the self-driven dynamics becomes the iterated map $`z\leftarrow z+\kappa' w\cdot s(z)`$ ($`\kappa'\propto\eta/(1-\mu)`$);
as long as this map is monotone and saturating in $`w`$, $`P`$ is still concave, $`m\ge 1`$, and $`m=\mathrm{mem}/\mathrm{SI}`$ is still exact under constant external drive (same proof, with the integral replaced by a sum).

(v) Test-point independence requires assumption L2: during the period in which $`s_i(u;w)`$ varies with $`w`$, $`\psi_{ji}(u)`$ differs across $`j`$ only by a constant factor (the transfer is quasi-static). Under L1+L2, $`m_{ji}=m_i`$, and $`m_i`$ = the point's own $`\mathrm{mem}_i/\mathrm{SI}_i`$.

Summary [derived]: (a) exactly, and (b) under L1 and L2, both satisfy the "multiplier model" $`g_j\approx M y_j+O(\gamma)`$, $`M=\mathrm{diag}(m_i)`$, $`m_i\ge 1`$, where $`m_i`$ is a property of the training point (its degree of self-fitting), independent of the test point.

## Proposition 3′ (the multiplier varies with the time at which the influence arrives)

**Status:** derived

Under L1, $`y_{ji}=\int\omega_{ji}(u)\,du`$, $`\omega_{ji}(u)=\psi_{ji}(u)\cdot\partial_w[w s_i](u;1)`$; the leave-one-out effect is $`A_{ji}=\int\psi_{ji}(u)s_i(u;1)\,du=\int\omega_{ji}(u)\mu_i(u)\,du`$, $`\mu_i(u)=1/(1-\sigma(z_i(u))\,\partial_w z_i(u))`$. Hence

```math
m_{ji} = \text{the } \omega_{ji}\text{-weighted average of } \mu_i(u).
```

$`\mu_i(u)`$ equals 1 before point $`i`$ is fitted by itself, and increases afterwards. For the same training point, the multiplier is large for test points that "its influence reaches only late in training", and close to 1 for those it reaches early.

In the replay bookkeeping (the push at step $`t`$, together with all feedback after it, is credited to step $`t`$):
for test points the influence reaches late, early pushes are cancelled by the point's own later fitting (early contributions are negative or very small), the late share $`y_{\text{late}}/y`$ is large, and the multiplier is also large —
the two bookkeeping schemes give the same testable consequence: $`m_{ji}`$ increases with $`y_{\text{late},ji}/y_{ji}`$.

## Trajectory multiplier and endpoint RIF factor

**Status:** 1.1 proved (case (a)); 1.2 proved (self-driven model, large-margin tail), with numerical checks

1.1 [proved, case (a)] For kernel ridge regression trained to optimum, the two coincide: the single-point leave-one-out effect = influence function $`\times 1/(1-h_i)`$, and at the optimum the exact meta-gradient is the influence function, so $`m_{ji}=1/(1-h_i)`$, independent of the test point.

1.2 [proved, self-driven model, large-margin tail] For a point in cross-entropy (= logistic on the margin) gradient flow that relies only on self-fitting: $`m=\mathrm{mem}/\mathrm{SI}`$, $`\mathrm{SI}=A/(1+A)`$, $`\mathrm{mem}=\ln(1+A)`$, $`A=\kappa Te^{-z_0}`$.

The endpoint RIF factor of the same point: endpoint curvature $`\ell''(z_T)=\sigma(z_T)\sigma(-z_T)\approx e^{-z_T}=1/(e^{z_0}+\kappa T)`$; early stopping is equivalent to ridge with $`\lambda=1/T`$; the single-point generalized Gauss–Newton is $`\ell''\kappa`$,

```math
h=\frac{\ell''\kappa}{\ell''\kappa+1/T}=\frac{\sigma(-z_T)\kappa T}{1+\sigma(-z_T)\kappa T};
```

and $`\sigma(-z_T)\kappa T=\kappa T/(e^{z_0}+\kappa T)=A/(1+A)=\mathrm{SI}`$, so $`h=\mathrm{SI}/(1+\mathrm{SI})`$, $`\mathrm{RIF}=1/(1-h)=1+\mathrm{SI}`$.

So in this model: $`\mathrm{RIF}=1+\mathrm{SI} \in[1,2)`$, $`m=-\ln(1-\mathrm{SI})/\mathrm{SI} \in[1,\infty)`$. Both are increasing functions of $`\mathrm{SI}`$ and give the same ordering; numerically RIF is larger for weakly self-fitted points (at $`\mathrm{SI}=0.1`$, 1.10 vs 1.05), and is capped at 2 for strongly memorized points (at $`\mathrm{SI}=0.99`$, 1.99 vs 4.65).

Reason: endpoint leverage only looks at "how much curvature is left at the endpoint", and strongly memorized points are already well fitted at the endpoint, with exponentially small curvature; the multiplier looks at how much the point was fitted by itself over the whole training process.

[checked numerically, exact logistic, scipy integration] 5 parameter sets: with weak self-fitting RIF is smaller than $`1+\mathrm{SI}`$ ($`\kappa T=2`$: RIF 1.43, $`1+\mathrm{SI}`$ 1.62, $`m`$ 1.27); with strong memorization RIF is capped at about 2.0, while $`m=3.7`$–$`5.2`$.

So "$`\mathrm{RIF}=1+\mathrm{SI}`$" is exact only in the large-margin tail, while "RIF is capped, $`m`$ is not" holds over all parameters.

## Proposition 4 (sign theorem within the multiplier class)

**Status:** proved (algebra)

Let $`g=My`$ ($`\gamma\to 0`$, self terms only). Because $`\sum\pi^W=\sum\pi^\perp=1`$:

```math
\begin{aligned}
\beta_\perp-\beta_W &= \sum_i m_i(\pi^\perp_i-\pi^W_i) = \sum_i (m_i-\bar{m})(\pi^\perp_i-\pi^W_i) = \int D(t)\,dt,\\
D(t)&:=\sum_{i:\ m_i>t}(\pi^\perp_i-\pi^W_i)
\end{aligned}
```

(layered summation, using $`m_i\ge 0`$).

(i) $`\kappa\le 0 \iff \sum_i m_i(\pi^\perp_i-\pi^W_i)\ge 0`$: the weight of the discarded part falls on training points with larger multipliers.

(ii) Sufficient condition (dominance): if $`D(t)\ge 0`$ for all $`t`$ (above every threshold, the discarded part has at least as much weight as the retained part), then $`\kappa\le 0`$; when $`\pi`$ is nonnegative this says "the multiplier distribution of the discarded part first-order stochastically dominates".

(iii) Equality: when $`D\equiv 0`$ (for example, $`m`$ constant on the supports of the two sets of weights, or the two sets of weights having the same multiplier distribution), $`\beta_W=\beta_\perp`$ and $`\kappa=0`$. When all multipliers are equal, $`\kappa=0`$ for every $`W`$.

(iv) Reverse direction: if $`D(t)\le 0`$ for all $`t`$, then $`\kappa\ge 0`$.

Combined with Proposition 1:

```math
\kappa = -\frac{\sum(m_i-\bar{m})(\pi^\perp_i-\pi^W_i) + \gamma(\text{pairwise}_\perp-\text{pairwise}_W)}{\beta} \cdot \rho\sqrt{1-\rho^2} \cdot \frac{L_P}{\sqrt{1-L_P^2}}.
```

The proof is algebra; it turns "the sign of $`\kappa`$" into a measurable question: on which part of the row the high-multiplier (high self-fitting, high memorization) training points fall.

## A general inequality (kernel ridge optimum)

**Status:** proved (case (a))

A general inequality [proved, case (a)]: let the whitened features be $`\psi_i=H^{-1/2}\varphi_i`$; then $`c_{ji}=\langle\psi_j,\psi_i\rangle`$, $`h_i=\lVert\psi_i\rVert^2`$,

```math
|y_{ji}| = |r_i|\cdot|\langle\psi_j,\psi_i\rangle| \le |r_i|\cdot\lVert\psi_j\rVert\cdot\sqrt{h_i},
```

and at the optimum $`r_i=(1-h_i)\cdot r_i^{\mathrm{LOO}}`$, so

```math
|y_{ji}| \le |r_i^{\mathrm{LOO}}|\cdot\lVert\psi_j\rVert\cdot(1-h_i)\sqrt{h_i}.
```

That is: the large elements of a row can only fall on points with large $`\sqrt{h_i}\,|r_i|`$ (medium to high leverage; for fully memorized points with $`h\to 1`$ the meta-gradient instead tends to zero while the leave-one-out effect does not — this is one corner of "not meta-smooth").

This shows that high-multiplier points "can" carry the spikes of a row, but does not say that the SVD necessarily assigns them to the discarded side; that requires assumptions about the test distribution, see Proposition 5 (carrier model).

## Proposition 5 (carrier model)

**Status:** proved, under assumptions K1–K3; first-order version with intersecting carriers derived

Carrier model. Assume

- K1: case (a) (ridge or its finite-duration equivalent), $`\gamma\to 0`$;
- K2: the eigendirections $`\{e_a\}`$ of $`\Phi^\top\Phi`$ (eigenvalues $`s_a`$) each have a disjoint set of "carrier" training points, and all carriers have the same absolute loading $`\ell`$ in their direction (so $`s_a=n_a\ell^2`$, $`n_a`$ = number of carriers); $`|r_i|=r`$ is the same for all;
- K3: after row normalization, the test coefficients are uncorrelated across directions ($`\sum_j \beta_{ja}\beta_{jb}=0`$, $`a\neq b`$), and the mean component of the row is negligible.

Then (1) row $`j`$ restricted to the carriers of $`a`$ is $`\propto(\ell_i r_i)_{i\in a}\cdot\varphi_{ja}/(s_a+\lambda)`$, and the right singular vectors of $`R`$ are the carrier vectors of the directions, with energy $`E_a\propto\tau_a\cdot s_a/(s_a+\lambda)^2`$ ($`\tau_a`$ = the second moment of test points along $`e_a`$);

(2) the leverage of a carrier is $`h=\ell^2/(n_a\ell^2+\lambda)`$, and the multiplier $`m(n_a)=1/(1-\ell^2/(n_a\ell^2+\lambda))`$ is strictly decreasing in the number of carriers $`n_a`$;

(3) $`\pi^W`$ and $`\pi^\perp`$ are nonnegative and fall on the carriers of the directions inside and outside $`W`$ respectively: $`\beta_W`$ and $`\beta_\perp`$ are weighted averages of $`m(n_a)`$ over the directions inside and outside $`W`$;

(4) when the test distribution equals the training distribution ($`\tau_a\propto s_a`$), $`E_a\propto(s_a/(s_a+\lambda))^2=d_a^2`$, which is strictly increasing in $`n_a`$, so $`W`$ = the $`B`$ directions with the most carriers, and every direction in $`W`$ has $`n_a`$ no smaller than any direction outside $`W`$,
hence $`\beta_W\le\beta_\perp`$ and $`\kappa\le 0`$ row by row;
equality holds if and only if all directions involved in the row have the same number of carriers (then the $`E_a`$ are also equal and the spectrum is flat).

(5) Reversal: under covariate shift $`\tau_a=w_a\cdot s_a`$ ($`w_a`$ = test/training density ratio), $`E_a\propto w_a\cdot d_a^2`$; for well-fitted directions $`d_a\approx 1`$, and $`E_a`$ is determined by $`w_a`$ alone.
If the test set concentrates on directions with few carriers ($`w_a`$ large and $`n_a`$ small), $`W`$ consists of high-multiplier directions; when every direction in $`W`$ has fewer carriers than every direction outside $`W`$, $`\kappa\ge 0`$ (strict inequality unless the numbers of carriers are equal).

**Proof:** (1)(2) are direct computations; (3) follows from the carriers being disjoint, so that $`u`$ and $`v`$ have disjoint supports; (4)(5) follow from (2)(3) and the monotonicity of $`E_a`$. ∎

K2 is a strong assumption (real data points carry both class directions and their own idiosyncratic directions, so the carriers intersect). When carriers intersect, the first-order computation below gives the same reading.

First-order version with intersecting carriers [derived, first order in leverage, ignoring cross terms between directions]: let $`p_{ia}=U_{ia}^2`$ (the loading share of training point $`i`$ in the $`a`$-th eigendirection, $`\sum_i p_{ia}=1`$), $`d_a=s_a/(s_a+\lambda)`$; then $`h_i=\sum_a p_{ia} d_a`$, and the amplification along direction $`a`$ is

```math
\beta_a\approx 1+\sum_i p_{ia} h_i = 1+d_a\cdot\mathrm{IPR}_a+\sum_{b\neq a} d_b O_{ab},
```

$`\mathrm{IPR}_a=\sum_i p_{ia}^2`$ (the degree of localization of direction $`a`$ = 1/effective number of carriers), $`O_{ab}=\sum_i p_{ia} p_{ib}`$.

That is, the amplification of a direction = 1 + "how well each of the points carrying it is fitted". Delocalized directions (many carriers) $`\approx 1+\bar{h}`$; localized directions (carried by a few points, which are fitted) are markedly larger.

$`\kappa<0 \iff`$ the discarded directions are more localized than the retained directions (after weighting). For Gaussian-like (fully delocalized) loadings all directions have the same $`\beta_a`$ and $`\kappa\approx 0`$ (up to a positive $`O(1/N)`$ term coming from the fourth moments of $`h_i`$ and $`p_{ia}`$).

## Proposition 6 (pairwise terms: synergy, lifting the shared side as the deletion fraction grows)

**Status:** derived; two sub-cases, (i) and (ii), proved

(i) [proved, case (a), first order] $`I_{il}\approx K_{il} m_i m_l (c_{ji} r_l+c_{jl} r_i)`$. For two similar points ($`K_{il}>0`$) that act on test $`j`$ with the same sign, $`I_{il}`$ has the same sign as $`A_i`$: deleting them together has a larger effect than the sum of deleting them separately (synergy).

(ii) [proved, cluster model of case (b)] $`n'`$ identical points pushing together: $`z_T(n')=Z(C+\kappa n'T)`$, concave in $`n'`$, so the effect of deleting $`t`$ points is convex in $`t`$, i.e. synergy.

(iii) [derived] Together, $`g_i`$ gains an extra $`-\gamma\sum_l I_{il}`$, with the same sign as $`m_i y_i`$ and relative size $`\approx 2\gamma\cdot\bar{m}\cdot\chi_i`$, $`\chi_i:=\sum_{\text{same-direction neighbors } l} K_{il}`$ (the part of $`i`$'s smoothing weight that falls on same-direction neighbors, $`\le 1-h_i`$).

Shared (clustered) points have $`\chi\approx 1-h`$ and isolated (idiosyncratic) points have small $`\chi`$, so $`\beta_W-\beta_\perp`$ gains about $`2\gamma\cdot\bar{m}\cdot(\mathbb{E}_{\pi^W}[\chi]-\mathbb{E}_{\pi^\perp}[\chi])>0`$: the larger the deletion fraction, the further $`\kappa`$ moves toward positive values.

In the carrier model $`\chi_i=d_a(1-1/n_a)`$ exactly. There is a critical fraction $`\gamma^*\approx\Delta_{\text{self}}\cdot\beta/(2\bar{m}(\bar{\chi}_W-\bar{\chi}_\perp))`$ (first order); for $`\gamma>\gamma^*`$, $`\kappa>0`$; for large $`\gamma`$ the $`O(\gamma^2)`$ terms are not negligible, so $`\gamma^*`$ is only an order of magnitude.

## Curvature lemma (curvature of the response in the amount deleted)

**Status:** proved

A general lemma [proved]: if along some direction the response to deleting a set $`S`$ can be written as $`a(S)=-F(\sum_{i\in S}u_i)`$ (all $`u_i`$ of the same sign, $`F(0)=0`$, $`F'(0)>0`$), then the single-point multiplier is $`m_i=F(u_i)/(F'(0)u_i)`$, and the difference between the group and the sum of single points is $`\sum_i F(u_i)-F(\sum u_i)`$.

$`F`$ concave (the response saturates with the amount deleted) $`\Rightarrow`$ every $`m_i\le 1`$ and groups are subadditive; $`F`$ convex (accelerating) $`\Rightarrow`$ $`m_i\ge 1`$ and superadditive.

That is, "multiplier $`<1`$" and "subadditive" are determined by the same sign: the curvature of the response in the amount deleted. In the kernel model (ridge regression) $`F`$ is convex (Woodbury amplification), so $`m\ge 1`$ and similar points are superadditive.

## Proposition 7 (two-layer product model)

**Status:** proved (checked numerically)

Proposition 7 [proved, two-layer "product" model]: $`f(x)=abx`$ (one shared feature, $`x`$ is a scalar feature), squared loss, gradient flow, balanced initialization $`a=b=\sqrt{c_0}`$. $`c=ab`$ satisfies $`\dot{c}=2c(s-qc)`$, $`s=\sum_i w_i x_i y_i`$, $`q=\sum_i w_i x_i^2`$, with solution

```math
c(T)=\frac{c^*}{1+(c^*/c_0-1)e^{-2sT}},\qquad c^*=s/q.
```

Deleting point $`i`$ decreases $`s`$ by $`u_i=x_i y_i`$ and $`q`$ by $`x_i^2`$.

(i) Before takeoff ($`c(T)\ll c^*`$, exponential growth phase): $`c\approx c_0 e^{2sT}`$, and the multiplier of a supporter ($`u_i>0`$) is $`m_i=(1-e^{-2Tu_i})/(2Tu_i)<1`$; when two supporters are deleted as a group,

```math
\text{sum of single points} - \text{group} = c(1-e^{-2Tu_i})(1-e^{-2Tu_l})>0,
```

exactly subadditive.

(ii) After saturation ($`c(T)\approx c^*`$): least squares, $`m_i=1/(1-x_i^2/q)\ge 1`$, and points with same-sign residuals are superadditive as a group.

(iii) Near the inflection point ($`c=c^*/2`$) the two regimes alternate.
[checked numerically, scipy] 100 points, $`T`$ = 0.3/0.7/1.0/1.5/4 times the inflection time: median supporter multipliers 0.984, 0.967, 0.997, 1.023, 1.011; pairwise (sum of single points − group)/|sum of single points| is +0.009 to +0.024 before takeoff (subadditive) and negative after saturation (superadditive).
The same model with cross-entropy: after separation it enters a logarithmic growth phase, $`m`$ = 1.007–1.03, superadditive.

Magnitude: $`1-m\approx Tu_i\le\ln(c^*/c_0)\cdot(u_i/s)/2`$, i.e. "the point's share of the support for this feature × the logarithm of the growth factor". When a feature has $`n`$ supporters, $`1-m`$ is about $`\ln(c^*/c_0)/(2n)`$; obtaining multipliers like 0.7–0.9 on digits requires features with few supporters ($`n\approx 10`$) and a large growth factor.
