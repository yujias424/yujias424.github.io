---
title: XGBoost
tags: Machine-Learning-Review
mathjax: true
---

### XGBoost: A Scalable Tree Boosting System

#### Introduction

*   Designs introduced scalability to XGBoost
    *   A novel tree learning algorithm is for handling sparse data.
    *   A theoretically justified weighted quantile sketch procedure enables handling instance weights in approximate tree learning.

#### Tree Boosting In a Nutshell

*   Regularized Learning Objective
    *   For a givem data set with $n$ examples and $m$ features $D = \{(x_i, y_i)\} (\lvert D\lvert = n, x_i \in \mathbb{R}^m, y_i\in \mathbb{R})$

        $$
        \hat{y}_i = \phi(x_i) = \sum_{k=1}^K f_k(x_i), \quad f_k\in\mathcal{F}
        $$

        where $\mathcal{F} = \{f(x) = w_{q(x)}\} (q: \mathbb{R}^m \rightarrow T, w\in\mathbb{R}^T)$ is the space of regression trees.

    *   Minimize the regularized objective  

        $$
        \mathcal{L}(\phi) = \sum_i l(\hat{y}_i, y_i) + \sum_k \Omega(f_k)
        $$

        where $\Omega(f) = \gamma T + \frac{1}{2}\lambda\lvert\lvert w \lvert\lvert^2$.

*   Gradient Tree Boosting

    *   Gradient Tree
        Noticed that the GTB model is trained in an additive manner, we thus add $f_t$ to minimize the objective function

        $$
        \mathcal{L}^{t} = \sum_{i=1}^n l(y_i, \hat{y}_i^{t-1} + f_t(x_i)) + \Omega(f_t)
        $$

        To accelerate the optimization, the above function can be approximated in second-order.

        $$
        \mathcal{L}^{(t)} \simeq \sum_{i=1}^n[l(y_i, \hat{y}^{t-1} + \partial_{\hat{y}^{t-1}}l(y_i, \hat{y}^{t-1}) f_t(x_i) + \frac{1}{2} \partial^2_{\hat{y}^{t-1}}l(y_i, \hat{y}^{t-1}) f_t^2(x_i))] + \Omega(f_t)
        $$

        Notice that $\partial_{\hat{y}^{t-1}}l(y_i, \hat{y}^{t-1})$ and $\partial^2_{\hat{y}^{t-1}}l(y_i, \hat{y}^{t-1})$ are first and second order gradient statistics on the loss function. And $\mathcal{L}^{(t)}$ can be further simplified to 

        $$
        \tilde{\mathcal{L}}^{t} = \sum_{i=1}^n [\partial_{\hat{y}^{t-1}}l(y_i, \hat{y}^{t-1}) f_t(x_i) + \frac{1}{2} \partial^2_{\hat{y}^{t-1}}l(y_i, \hat{y}^{t-1}) f_t^2(x_i))] + \Omega(f_t)
        $$

        To simplify the equation, we may use $g_i = \partial_{\hat{y}^{t-1}}l(y_i, \hat{y}^{t-1})$ and $h_i = \partial^2_{\hat{y}^{t-1}}l(y_i, \hat{y}^{t-1})$. 

        Define $I_j = \{i\lvert q(x_i) = j\}$ as the instance set of leaf j, we can further expand $\Omega$

        $$
        \begin{aligned}
            \mathcal{L}^{(t)} 
            &= \sum_{i=1}^n[g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i))] + \gamma T + \frac{1}{2}\lambda\sum_{j=1}^T w_j^2 \\
            &= \sum_{j=1}^T[(\sum_{i\in I_j} g_i)w_j] + \frac{1}{2}\sum_{j=1}^T[\sum_{i\in I_j} h_i]w_j^2 + \frac{1}{2}\lambda\sum_{j=1}^Tw_k^2 + \gamma T\\
            &= \sum_{j=1}^T[(\sum_{i\in I_j} g_i)w_j + \frac{1}{2}(\sum_{i\in I_j}h_i + \lambda)w_j^2] + \gamma T
        \end{aligned}
        $$

        To derive that, we need to notice that $f_t(x_i)$ is actually associated with $w$, as mentioned in the paper, the final prediction ($\hat{y}_i = \sum_{t=1}^Tf_t(x_i)$) by summing up the score in the corresponding leaves (given by w). In other word, we have $w_j = \sum_{i\in I_j}f_t(x_i)$.

        Given a fixed structure $q(x)$, we have optimal weight $w^*_j$ of leaf $j$ by (Simply let $\frac{\partial \mathcal{L}^{(t)}}{\partial w_j} = 0$ and derive the $w_j$).

        $$
        w_j^* = \frac{\sum_{i\in I_j} g_i}{\sum_{i\in I_j}h_i + \lambda}
        $$

        Thus, we have **optimal value that can be used to meaure the quality of a tree structure $q$**.
        
        $$
        \tilde{\mathcal{L}}^{(t)}(q) = -\frac{1}{2}\sum_{j=1}^T\frac{(\sum_{i\in I_j} g_i)^2}{\sum_{i\in I_j} h_i+\lambda} + \gamma T
        $$

    *   Greedy algorithm to obtain the tree structure.

        **Core idea:**
        Starts from a single leaf and iteratively adds branches to the tree 

        Assume that $I_L$ and $I_R$ are the instance sets of left and right nodes after the split. Letting $I = I_L \cup I_R$, then **the loss reduction after the split** is given by

        $$
        \mathcal{L}_{split} = \frac{1}{2}[\frac{(\sum_{i\in I_L}g_i)^2}{\sum_{i\in I_L}h_i + \lambda} + \frac{(\sum_{i\in I_R}g_i)^2}{\sum_{i\in I_R}h_i + \lambda} - \frac{(\sum_{i\in I}g_i)^2}{\sum_{i\in I}h_i + \lambda}] -\gamma
        $$

#### Split finding algorithm

*   Basic Exact Greedy Algorithm
    *   **Algorithm 1**: Exact Greedy Algorithm for Split Finding 
        **Input**: $I$, instance set of current node  
        **Input**: $d$, feature dimension  
        gain $\leftarrow 0$  
        $G\leftarrow \sum_{i\in I}g_i$, $H\leftarrow \sum_{i\in I}hi$  
        **for** $k=1$ **to** $m$ **do**  
        $\quad G_L\leftarrow 0$, $H_L\leftarrow 0$  
        $\quad$**for** $j\,\,in\,\,sorted(I, \,\, by\,x_{jk})$ **do**     
        $\qquad G_L\leftarrow + g_j$, $H_L\leftarrow H_L + h_j$  
        $\qquad G_R\leftarrow G - G_L$, $H_R\leftarrow H - H_L$  
        $\qquad score\leftarrow max(score, \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda})$ 
        $\quad$**end**  
        **end**  
        **Output**: Split with max score
*   Approximate Algorithm
    *   **Algorithm 2**: Approximate Algorithm for Split Finding  
        **for** $k=1$ **to** $m$ **do**
        $\quad$Propose $S_k=\{s_{k1}, s_{k2}, \ldots, s_{kl}\}$ by percentiles on feature $k$.
        $\quad$Proposal can be done per tree (global), or per split(local).
        **end**  
        **for** $k=1$ **to** $m$ **do**  
        $\quad G_{kv}\leftarrow = \sum_{j\in \{j \lvert s_{k,v}\geq x_{jk} > s_{k,v-1}\}}g_j$
        $\quad H_{kv}\leftarrow = \sum_{j\in \{j \lvert s_{k,v} \geq x_{jk} > s_{k,v-1}\}}h_j$
        **end**  
*   Sparsity-aware Split Finding
    *   **Algorithm 3**: Sparsity-aware Split Finding
        **Input**: $I$, instance set of current node  
        **Input**: $I_k = \{i\in I\lvert x_{ik} \neq missing\}$  
        **Input**: $d$, feature dimension  
        Also applies to the approximate setting, only collect statistics of non-missing entries into buckets gain $\leftarrow 0$.  
        $G \leftarrow \sum_{i\in I}, g_i, H\leftarrow \sum_{i\in I}h_i$  
        **for** $k=1$ **to** $m$ **do**  
        $\quad$ // enumerate missing value goto right
        $G_L\leftarrow 0, H_L\leftarrow 0$  
        $\quad$**for** $j\,\,in\,\,sorted(I_k,\,\,ascent\,\,order\,\,by\,\,x_{jk})$ **do**
        $\qquad G_L\leftarrow G_L+g_j, \,H_l\leftarrow H_L + h_j$  
        $\qquad G_R\leftarrow G-G_L,   \,H_R\leftarrow H-H_L$
        $\qquad score\leftarrow max(score, \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda})$
        $\quad$**end**
        $\quad$// enumerate missing value goto left
        $\quad G_R\leftarrow 0, H_R\leftarrow 0$
        $\quad$**for** $j\,\,in\,\,sorted(I_k,\,\,ascent\,\,order\,\,by\,\,x_{jk})$ **do**
        $\qquad G_R\leftarrow G_R+g_j, \,H_R\leftarrow H_R + h_j$  
        $\qquad G_L\leftarrow G-G_R,   \,H_L\leftarrow H-H_R$
        $\qquad score\leftarrow max(score, \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda})$
        $\quad$**end**
        **end**
