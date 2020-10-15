---
title: SHapley Additive exPlanations (SHAP)
tags: ML-Evaluation
mathjax: true
---

### A Unified Approach to Interpreting Model Predictions

#### Introduction

*   Explanation model
    Viewing any explanation of a model's prediction as a model itself.
*   Application of game theory
    Game theory results guaranteeing a unique solution apply to the entire class of additive feature attribution methods and propose SHAP values as a unified measure of feature importance that various methods approximate.

#### Addidtive Feature Attribution Methods

*   **Definition 1**  
    **Additive feature attribution methods** have an explanation model that is a linear function of binary variables:
    
    $$
    g(z') = \phi_0 + \sum_{i=1}^M\phi_iz'_i
    $$

    where  $z'\in\{0, 1\}^M$, $M$ is the number of simplified input features, and $\phi_i\in \mathbb{R}$.  
    Noticed that methods with explanation models matching aforementioned definition will **attribute an effect $\phi_i$ to each feature**, and **summing the effects of all feature attributions approximates the output $f(x)$ of the original model.**

*   **LIME**  
    LIME interprets individual model predictions based on locally approximating the model around a given prediction, and it refers to simplified inputs $x'$ as "interpretable inputs", and the mapping $x=h_x(x')$ converts a binary vector of interpretable inputs into the original input space.
    To find $\phi$, LIME will minimize the following objective function:

    $$
    \xi = \argmin_{g\in\mathcal{G}} L(f, g, \pi_{x'}) + \Omega(g)
    $$

    Faithfulness of the explanation model $g(z')$ to the original model $f(h_x(z'))$ is enforced through the loss $L$ over a set of samples in the simplified input space weighted by the local kernel $\pi_{x'}$, and the $\Omega$ is used to penalize the complexity of $g$.
*   DeepLIFT
*   Layer-Wise Relevance Propagation
*   **Classic Shapley Value Estimation**
    *   Shapley regression values  
        *   Definition:  
            Feature importances for linear models in the presence of multicollinearity.  
        *   Algorithm:  
            *   An importance value will be assigned to each feature to represent the effect on the model prediction of including that feature.  
                1.  Train two models with and without feature $i$ respectively, $f_S$ and $f_{S\cup\{i\}}$.
                2.  Compare the prediction of two models using current input
                    
                    $$
                    f_{S\cup\{i\}}(x_{S\cup\{i\}}) - f_S(x_S)
                    $$

                    where $x_S$ represents the values of the input features in the set $S$ and $x_{S\cup\{i\}}$ represents the values of the input features in the set $S\cup\{i\}$.
                3.  Calculate the Shapley values $\phi_i$, which is a weighted average of all possible differences.

                    $$
                    \phi_i = \sum_{S\subseteq F \setminus \{i\}} \frac{\lvert S\lvert!(\lvert F\lvert - \lvert S\lvert - 1)!}{\lvert F\lvert!} [f_{S\cup\{i\}}(x_{S\cup\{i\}}) - f_S(x_S)]
                    $$

    *   Shapley sampling values
        1. Applying sampling approximations to aforementioned $\phi_i$ formula.
        2. Approximating the effect of removing a variable from the model by integrating over samples from the training dataset.

#### Simple Properties Uniquely Determine Additive Feature Attributions

*   Property 1 (**Local accuracy**)

    $$
    f(x) = g(x') = \phi_0 + \sum_{i=1}^M\phi_ix_i'
    $$

    The explanation model $g(x')$ matches the original model $f(x)$ when $x=h_x(x')$, where $\phi_0 = f(h_x(0))$ represents the model output with all simplified inputs toggled off (i.e. missing) 
*   Property 2 (**Missingness**)

    $$
    x'_i = 0 \Rightarrow \phi_i=0
    $$

    Missingness constrains features where $x'_i = 0$ to have no attributed impact.

*   Property 3 (**Consistency**)

    Let $f_x(z') = f(h_x(z'))$ and $z'\setminus i$ denote setting $z'_i = 0$. For any two models $f$ and $f'$, if

    $$
    f'_x(z') - f_x'(z'\setminus i) \geq f_x(z') - f_x(z'\setminus i) 
    $$

    for all inputs $z' \in \{0,1\}^M$, then $\phi_i(f',x) \geq \phi_i(f,x)$.
*   Theorem 1  
    Only one possible explanation model $g$ follows three properties and aforementioned definition 1.

    $$
    \phi_i(f,x) = \sum_{z'\subseteq x'} \frac{\lvert z'\lvert!(M-\lvert z'\lvert-1)!}{M!}[f_x(z') - f_x(z'\setminus i)]
    $$

    where $\lvert z'\lvert$ is the number of non-zero entries in $z'$, and $z'\subseteq x'$ represents all $z'$ vectors where the non-zero entries are a subset of the non-zero entries in x'.



