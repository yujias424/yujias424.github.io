---
title: Ensemble Method
tags: Machine Learning Review
mathjax: true
---

### Ensemble Method  

#### Ensemble  
* Definition:
  A set of learned models whose individual decisions are combined in some way to make predictions for new instances.
* When ensemble does a better job?
  The errors made by the individual predictors are (somewhat) uncorrelated, and the predictors' error rates are better than guessing (< 0.5 for 2-class problem)
* Diverse classifiers
  Encourage diversity in their errors by
  * choosing a variety of learning algorithms
  * choosing a variety of settings (e.g. # hidden units in neural nets) for the learning algorithm
  * choosing different subsamples of the training set (bagging)
  * using different probability distributions over the training instances (boosting, skewing)
  * choosing different features and subsamples (random forests)

#### Bagging 
**(Bootstrap Aggregation)**  
* Algorithm:
  * Learning:
    * Given: 
      Learner $L$, training set $D = \{(x_1,y_1) \ldots (x_m,y_m)\}$
    * Action:
      for $i \leftarrow 1$ to $T$:  
      $\quad\quad$ $D^{(i)} \leftarrow m$ instances randomly drawn <u>with replacement</u> from $D$.  
      $\quad\quad$ $h_i \leftarrow$ model learned using $L$ on $D^{(i)}$
  * Predition:
    * Classification:
      * Given:
        Test instance $x$
      * Action:
        predict $y \leftarrow$ plurality_vote($h_1(x), \ldots h_T(x)$)
    * Regression:
      * Given:
        Test instance $x_t$
      * Action:
        predict $y \leftarrow$ plurality_vote($h_1(x), \ldots h_T(x)$)
* Comments
  * each sampled training set is a bootstrap replicate
    * contains m instances (the same as the original training set)
    * on average it includes 63.2% of the original training set
    * some instances appear multiple times
  * can be used with any base learner
  * works best with *<u>unstable</u>* learning methods: those for which small changes in D result in relatively large changes in learned models, i.e., those that tend to **overfit** training data.

#### Boosting
* Intuition
  * Boosting came out of the PAC learning community.
  * A weak PAC learning algorithm is one that cannot PAC learn for arbitrary ε and δ, but it can for some: its hypotheses are at least slightly better than random guessing.
  * Suppose we have a weak PAC learning algorithm L for a concept class C. Can we use L as a subroutine to create a (strong) PAC learner for C?
    * Yes, by boosting! [Schapire, Machine Learning 1990]
    * The original boosting algorithm was of theoretical interest, but assumed an unbounded source of training instances.
  * A later boosting algorithm, AdaBoost, has had notable practical success.
* Algorithm of AdaBoost
  * Given:
    Learner $L$, # stages $T$ (or say iteration number), training set $D = \{(x_1,y_1) \ldots (x_m,y_m)\}$
  * Pseudocode: ($i$ denotes the $i^{th}$ instance)  
    for all $i: w_1(i) \leftarrow \frac{1}{m}$  $\quad$// initialize instance weights  
    for $t\leftarrow 1$ to $T$ do  
    $\quad\quad$for all $i: p_t(i) \leftarrow \frac{w_t(i)}{\sum_jw_t(j)} \quad$// normalize weights  
    $\quad\quad h_t \leftarrow$ model learned using $L$ on $D$ and $p_t$  
    $\quad\quad$ $\epsilon_t \leftarrow \sum_ip_t(i)(1-\delta(h_t(x_i), y_i)$   $\quad$// calculate weighted error  
    $\quad\quad$if $\epsilon_t>0.5$ then  
    $\quad\quad \quad\quad T\leftarrow t-1$  
    $\quad\quad \quad\quad$break  
    $\quad\quad \beta_t \leftarrow \frac{\epsilon_t}{1-\epsilon_t} \quad$ // lower error, smaller $\beta_t$  
    $\quad\quad$for all $i$ where $h_t(x_i) = y_i \quad$ // downweight correct examples  
    $\quad\quad$$\quad\quad w_{t+l}(i) \leftarrow w_t(i)\beta_t$  
    return:  
    $\quad\quad$ $h(x) = \arg\max_y\sum_{t=1}^T(\log{\frac{1}{\beta_t}})\delta(h_t(x), y)$  
  * Comments:
    * $\delta()$ is the error function, $E(h(x), i, y) = e^{-y_ih(x_i)}$
    * Implementing weighted instances with AdaBoost
      * AdaBoost calls the base learner $L$ with probability distribution $p_t$ specified by weights on the instances.
      * There are two ways to handle this
        1. Adapt $L$ to learn from weighted instances; straightforward for decision trees and naïve Bayes, among others
        2. Sample a large (>> m) unweighted set of instances according to $p_t$; run $L$ in the ordinary manner

#### Bagging vs. Boosting
* Bagging almost always better than single decision tree or neural net
* Boosting can be much better than bagging
* Boosting can sometimes reduce accuracy (too much emphasis on outliers?)

#### Random forests
* Algorithm:
   * Learning:
     * Given: 
       Candidate feature splits $F$, training set $D = \{(x_1,y_1) \ldots (x_m,y_m)\}$
     * Action:
       for $i \leftarrow 1$ to $T$:  
       $\quad\quad D^{(i)} \leftarrow m$ instances randomly drawn <u>with replacement</u> from $D$.  
       $\quad\quad h_i \leftarrow$ *<u>randomized</u>* decision tree learned with $F, D^{(i)}$
   * Randomized decision tree learning:
     * To select a split at a node  
     $\quad\quad R ←$ randomly select (without replacement) $f$ feature splits from $F$ (where $f \approx \sqrt{|F|}$)  
     $\quad\quad$ choose the best feature split in $R$.
     * do not prune trees.
   * Predition:
     * Classification/Regression:  
       As in bagging