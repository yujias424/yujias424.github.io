---
title: Regularization
tags: Machine_Learning
mathjax: true
---

### Regularization

##### Overview
**Definition**  
*   In general: Any method to prevent overfitting or help the optimization.
*   Specifically: Additional terms in the training optimization objective to prevent overfitting or help the optimization.

**Overfitting**  
*   Key: empirical loss and expected loss are different.
*   Smaller the data set, larger the difference between empirical loss and expected loss.
*   Larger the hypothesis class, easier to find a hypothesis that fits the difference between the two type of loss.
    *   hus has small training error but large test error (overfitting)
*   Larger data set helps 
*   Throwing away useless hypotheses also helps (**regularization**)

##### Different views of regularization
**Regularization as hard constraint**  
*   When $\Omega$ measured by some quantity $R$  
    
    $$
    \begin{aligned}
    \min_\theta &\hat{L}(\theta) = \frac{1}{n}\sum_{i=1}^nl(\theta,x_i,y_i) \\
    &s.t. \quad R(\theta) \leq r
    \end{aligned}
    $$

*   For example, $l_2$ regularization. $R(\theta) \leq r$ can be expressed as $||\theta||^2_2\leq r^2$.  

**Regularization as soft constraint**
*   The hard-constraint optimization is equivalent to soft-constraint  
    
    $$
    \begin{aligned}
    \min_\theta &\hat{L}(\theta) = \frac{1}{n}\sum_{i=1}^nl(\theta,x_i,y_i) + \lambda^* R(\theta)
    \end{aligned}
    $$

    For some $\lambda^* > 0$  
*   Lagrangian multiplier method.  
    
    $$
    \mathcal{L}(\theta. \lambda) := \hat{L}(\theta) + \lambda[R(\theta)-r]
    $$

*   Suppose $\theta^* \,$ is the optimal for hard-constraint optimization.  
    
    $$
    \theta^* = \arg\min_\theta \max_{\lambda\geq0}\mathcal{L}(\theta, \lambda) := \hat{L}(\theta) + \lambda[R(\theta)-r]
    $$

*   Suppose $\lambda^* \,$ is the corresponding optimal for max  
    
    $$
    \theta^* = \arg\min_\theta \mathcal{L}(\theta, \lambda^* ) := \hat{L}(\theta) + \lambda^* [R(\theta)-r]
    $$

**Regularization as soft constraint**
*   Bayesian View: Everything will be regarded as a distribution.
*   Prior over the hypotheses: $p(\theta)$
*   Posterior over the hypotheses: $p(\theta| \{x_i, y_i\})$
*   Likelihood: $p(\{x_i, y_i\}|\theta)$
*   Bayesian Rule:  
    
    $$
    p(\theta|\{x_i, y_i\}) = \frac{p(\theta)p(\{x_i, y_i\}|\theta)}{p(\{x_i, y_i\})}
    $$

*   Maximum A Posteriori (MAP):  
    
    $$
    \max_\theta\log p(\theta|\{x_i, y_i\}) = \max_\theta\log p(\theta) + \log p(\{x_i, y_i\}|\theta)
    $$

    To be noticed, $\log p(\theta)$ is the **regularization** term and $\log p(\{x_i, y_i\}|\theta)$ is the **MLE loss** term.  
*   Example: $l_2$ loss with $l_2$ regularization is correspond to a normal likelihood $p(x,y|\theta)$ and a normal prior $p(\theta)$.  

**Three Views**  
1.   Typical choice for optimization: soft-constraint.
2.   Hard-constraint preferred if
     *   Know the explicit bound $R(\theta) \leq r$.
     *   Soft-constraint causes trapped in a local minima while projection back to feasible set leads to stability.
3.   Bayesian view preferred if
     *   Domain knowledge easy to represent as a prior

##### Examples of Regularization
**Classical Regularization**  
**Overview**  
*   Norm penalty  
    *   $l_2$ regularization  
    *   $l_1$ regularization  
*   Robustness to noise
    *   Noise to the input
    *   Noise to the weights

****

**$l_2$ Regularization**  

$$
\min_\theta \hat{L}_R(\theta) = \hat{L}(\theta) + \frac{\alpha}{2}||\theta||^2_2
$$

*   Effect on (stochastic) gradient descent 
    *   Gradient of regularized objective
        
        $$
        \nabla \hat{L_R}(\theta) = \nabla \hat{L}(\theta) + \alpha\theta
        $$

    *   Gradient descent update
        
        $$
        \begin{aligned}
        \theta &= \theta - \eta\nabla\hat{L_R}(\theta) \\
        &= \theta - \eta\nabla\hat{L}(\theta) - \eta\alpha\theta \\
        &= (1-\eta\alpha)\theta - \eta\nabla\hat{L}(\theta)
        \end{aligned}
        $$

    *   Terminology: **weight decay**  
*   Effect on the optimal solution  
    *   Consider a quadratic approximation ($f(x) \approx f(x_0) + f'(x_0)(x-x_0) + \frac{f''(x_0)}{2}(x-x_0)^2 \,\,(x\approx x_0)$) around $\theta^* \,$. Notice that $H$ is the Hessian Matrix, $[\frac{\partial^2\mathcal{L}}{\partial w_i \partial w_j}]$
        
        $$
        \hat{L}(\theta) \approx \hat{L}(\theta^* ) + (\theta - \theta^* )^T \nabla \hat{L}(\theta^* ) + \frac{1}{2}(\theta - \theta^* )^TH(\theta - \theta^*    )
        $$

    *   Since $\theta^* \,$ is optimal, $\nabla\hat{L}(\theta^* )=0$, we have,
        
        $$
        \begin{aligned}
        \hat{L}(\theta) \approx \hat{L}(&\theta^* ) + \frac{1}{2}(\theta - \theta^* )^TH(\theta - \theta^* ) \\
        &\nabla\hat{L}(\theta) \approx H(\theta - \theta^* )
        \end{aligned}
        $$

    *   Gradient of regularized objective ($\alpha\theta$ comes from the derivitive of the $l_2$ term, $\frac{\alpha}{2}||\theta||^2_2$)
        
        $$
        \nabla\hat{L_R}(\theta) \approx H(\theta - \theta^* ) + \alpha\theta
        $$

    *   On the optimal $\theta_R^* \,$
        
        $$
        \begin{aligned}
        0 = \nabla\hat{L_R}&(\theta_R^* ) \approx H(\theta_R^* - \theta^* ) + \alpha\theta_R^* \\
        &\theta_R^* \approx (H+\alpha I)^{-1}H\theta^* 
        \end{aligned}
        $$

    *   Suppose $H$ has eigen-decomposition $H = QΛQ^T$
        
        $$
        \theta_R^* \approx (H+\alpha I)^{-1}H\theta^* = Q(Λ + \alpha I)^{-1}ΛQ^T\theta^*
        $$

    *   Effect: rescale along eigenvectors of $H$

****

**$l_1$ Regularization**  

$$
\min_\theta \hat{L}_R(\theta) = \hat{L}(\theta) + \alpha||\theta||_1
$$

*   Effect on (stochastic) gradient descent 
    *   Gradient of regularized objective, and the sign will be applied to each element in $\theta$
        
        $$
        \nabla \hat{L_R}(\theta) = \nabla \hat{L}(\theta) + \alpha\cdot sign(\theta)
        $$

    *   Gradient descent update
    
        $$
        \begin{aligned}
        \theta &= \theta - \eta\nabla\hat{L_R}(\theta) \\
        &= \theta - \eta\nabla\hat{L}(\theta) - \eta\alpha\cdot sign(\theta)
        \end{aligned}
        $$

*   Effect on the optimal solution
    *   Consider a quadratic approximation around $\theta^* \,$. Notice that $H$ is the Hessian Matrix, $[\frac{\partial^2\mathcal{L}}{\partial w_i \partial w_j}]$
        
        $$
        \hat{L}(\theta) \approx \hat{L}(\theta^* ) + (\theta - \theta^* )^T \nabla \hat{L}(\theta^* ) + \frac{1}{2}(\theta - \theta^* )^TH(\theta - \theta^*    )
        $$

    *   Since $\theta^* \,$ is optimal, $\nabla\hat{L}(\theta^* )=0$, we have,
        
        $$
        \begin{aligned}
        \hat{L}(\theta) \approx \hat{L}(&\theta^* ) + \frac{1}{2}(\theta - \theta^* )^TH(\theta - \theta^* )
        \end{aligned}
        $$

    *   Further assume that $H$ is diagonal and positive ($H_{ii} > 0, \forall_i$)
    *   The regularized objective is (ignoring constants)
        
        $$
        \hat{L_R}(\theta) \approx \sum_i\frac{1}{2}H_{ii}(\theta_i - \theta_i^* )^2 + \alpha|\theta_i|
        $$

    *   The optimal $\theta_R^* \,$ approximate to,
        
        $$
        (\theta_R^* )_ i \approx
        \begin{cases}
        \max\{\theta_i^* - \frac{\alpha}{H_{ii}, 0}\}\quad if\,\,\theta_i^* \geq 0 \\
        \min\{\theta_i^* + \frac{\alpha}{H_{ii}, 0}\}\quad if\,\,\theta_i^* \leq 0
        \end{cases}
        $$

    *   Effect: induce sparsity
    *   Further assume that $H$ is diagonal
    *   Compact expression for the optimal $\theta_R^* \,$
        
        $$(\theta_R^* )_ i \approx sign(\theta_i^* )\max\{|\theta_i^* | - \frac{\alpha}{H_{ii}},0\}$$
    
**Bayesian View**  
Gaussian likelihood → squared error.  
Gaussian prior → L2-regularization.  
Laplace prior → L1-regularization.  
Laplace likelihood → absolute error.  
Sigmoid likelihood → logistic loss.  

**Equivalence to weight decay**  
*   Suppose the hypothesis is $f(x) = w^Tx$, noise is $\epsilon \sim N(0,\lambda I)$  
*   After adding noise, the loss is
    
    $$
    \begin{aligned}
    &\quad\quad\,\,\,\, L(f) = \mathbb{E}_{x,y,\epsilon}[f(x+\epsilon)-y]^2 = \mathbb{E}_{x,y,\epsilon}[f(x) + w^T\epsilon -y]^2\\
    &\implies L(f) = \mathbb{E}_{x,y,\epsilon}[f(x)-y]^2 + 2\mathbb{E}_{x,y,\epsilon}[w^T\epsilon(f(x)-y)]^2 + \mathbb{E}_{x,y,\epsilon}[w^T\epsilon]^2\\
    &\implies L(f) = \mathbb{E}_{x,y,\epsilon}[f(x)-y]^2 + \lambda||w||^2
    \end{aligned}
    $$

**Add noise to the weights**  
*   For the loss on each data point, add a noise term to the weights before computing the prediction
    
    $$
    \epsilon\sim N(0, \eta I),\,\,w' = w + \epsilon
    $$

*   Prediction: $f_{w'}(x)$ instead of $f_w(x)$
*   Loss becomes
    
    $$
    L(f) = \mathbb{E}_{x,y,\epsilon}[f_{w+\epsilon}(x)-y]^2
    $$

*   We use <u>Taylor expansion</u> to simplify the function term.

    $$
    f_{w+\epsilon}(x) \approx f_w(x) + \epsilon^T\nabla f_w(x)
    $$

*   Plug in loss function. 
    
    $$
    L(f) \approx \mathbb{E}_{x,y,\epsilon}[f_{w}(x)-y]^2 + 2\mathbb{E}_{x,y,\epsilon}[(f_{w}(x)-y)\epsilon^T\nabla f_w(x)] + \eta\mathbb{E}||\nabla f_w(x)||^2
    $$

    *   Regularization term: $\eta\mathbb{E}||\nabla f_w(x)||^2$
    *   Expectation of $\epsilon^T$ equals 0.
    
##### Other types of regularization
**Data augmentation**  
*   Adding noise to the input: a special kind of augmentation
*   Be careful about the transformation applied:
    *   Example: classifying ‘b’ and ‘d’
    *   Example: classifying ‘6’ and ‘9’

**Early stopping**  
*   Idea: don’t train the network to too small training error
    *  Recall overfitting: Larger the hypothesis class, easier to find a hypothesis that fits the difference between the two
    *  Prevent overfitting: do not push the hypothesis too much; use validation error to decide when to stop
*   When training, also output validation error 
*   Every time validation error improved, store a copy of the weights 
*   When validation error not improved for some time, stop 
*   Return the copy of the weights stored
*   hyperparameter selection: training step is the hyperparameter
*   Advantage:
    *   Efficient: along with training; only store an extra copy of weights
    *   Simple: no change to the model/algo
*   Disadvantage: need validation data
*   Strategy to heuristically mitigate the disadvantage
    *   After early stopping of the first run, train a second run and reuse validation data
*   How to heuristically reuse validation data
    1.  Start fresh, train with both training data and validation data up to the previous number of epochs 
    2.  Start from the weights in the first run, train with both training data and validation data until the validation loss is smaller than the training loss at the early stopping point


**Dropout**  
*   Randomly select weights to update
*   More precisely, in each update step
    *   Dropout probability p, or present probability 1-p
    *   Randomly sample a different binary mask to all the input and hidden units 
    *   Multiple the mask bits with the units and do the update as usual
*   During test time: all units present; multiply weight by 1-p
*   Typical dropout probability: *<u>0.2 for input and 0.5 for hidden units</u>*

**Batch Normalization**  
**Algorithm**  
*<u>Input:</u>*
*   Values of $x$ over a mini-batch: $\mathcal{B} = \{x_{1,...,m}\}$;
*   Parameters to be learned: $\gamma$, $\beta$;

*<u>Output:</u>*  
$\{y_i =  BN_{\gamma, \beta}(x_i)\}$  
$\mu_\mathcal{B} \leftarrow \frac{1}{m}\sum_{i=1}^m x_i$ $\quad$//mini-batch mean  
$\sigma_\mathcal{B} \leftarrow \frac{1}{m}\sum_{i=1}^m (x_i-\mu_\mathcal{B})^2$ $\quad$//mini-batch variance  
$\hat{x_i}\leftarrow \frac{x_i-\mu_{\mathcal{B}}}{\sqrt{\sigma_\mathcal{B}^2+\epsilon}}$ $\quad$//normalize  
$y_i\leftarrow \gamma\hat{x_i} + \beta \equiv BN_{\gamma, \beta}(x_i)$ $\quad$//scale and shift  

**Comments**  
* First three steps are just like standardization of input data, but with respect to only the data in mini-batch. Can take derivative and incorporate the learning of last step parameters into backpropagation.  
* Note last step can completely un-do previous 3 steps.  
* But if so this un-doing is driven by the later layers, not the earlier layers; later layers get to “choose” whether they want standard normal inputs or not.  
