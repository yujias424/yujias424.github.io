---
title: Decision Tree
tags: Machine_Learning
mathjax: true
---

### Decision Tree

##### Overview
**Simple Structure**  
Each internal node tests one features $x_i$.  
Each branch from an internal node represents one outcome of the test.  
Each leaf predicts $y$ or $P(y|x)$  

##### Top-down decision tree learning
***Pseudocode***  
**MakeSubtree**(set of training instances $D$)  
$\quad\quad C$ = **DetermineCandidateSplits**($D$)  
$\quad\quad$if stopping criteria met:  
$\quad\quad\quad\quad$make a leaf node $N$ determine class label/probabilities for $N$  
$\quad\quad$else:  
$\quad\quad\quad\quad$make an internal node $N$  
$\quad\quad S$ = **FindBestSplit**($D$, $C$)  
$\quad\quad\quad\quad$for each outcome $k$ of $S$:  
$\quad\quad D_k$ = subset of instances that have outcome $k$  
$\quad\quad\quad\quad k^{th}$ child of $N$ = **MakeSubtree**($D_k$)  
$\quad\quad$return subtree rooted 

***Candidate Splits***
1.  Splits on nominal features have one branch per value.
2.  Splits on numeric features use a threshold.

***Pseudocode of Candidate splits on numeric features***  
**\#** *<u>Run this subroutine for each numeric feature at each node of DT induction</u>*
**DetermineCandidateNumericSplits**(set of training instances $D$, feature $X_i$):  
$\quad\quad C$ = { } **\#** *<u>initialize set of candidate splits for feature $X_i$</u>*  
$\quad\quad S$ = partition instances in $D$ into sets $s_1 \cdots s_V$ where the instances in each set have the same value for $X_i$  
$\quad\quad$let $v_j$ denote the value of $X_i$ for set $s_j$   
$\quad\quad$sort the sets in $S$ using $v_j$ as the key for each $s_j$   
$\quad\quad$for each pair of adjacent sets $s_j$, $s_{j+1}$ in sorted $S$:  
$\quad\quad \quad\quad$if $s_j$ and $s_{j+1}$ contain a pair of instances with different class labels:  
$\quad\quad \quad\quad$**\#** *<u>assume we’re using midpoints for splits </u>*  
$\quad\quad \quad\quad$add candidate split $X_i ≤ (v_j + v_{j+1})/2$ to $C$  
$\quad\quad$return $C$

##### Finding the best splits
**Key Hypothesis**  
The simpliest  tree that classifies the training instances accuractely will work well on previously unseen instances.  

**Information-theoretic heuristic**  
Owing to the fact that finding the smallest possible decision tree that accurately classifies the training set is a NP-hard problem, an information-theoretic heuristic approach will be applied to greedily choose splits.  

**Entropy**  
Entropy is a measure of uncertainty associated with a random variable, which is defined as the expected number of bits required to communicate the value of the variable.  
$$
H(Y) = - \sum_{y\in values(Y)}p(y)\log_2P(y)
$$

**Conditional Entropy**  
$$
H(Y|X) = \sum_{x\in values(X)}P(X=x)H(Y|X=x)
$$  
where,  
$$
H(Y|X=x) = - \sum_{y\in values(Y)}p(Y=y|X=x)\log_2P(Y=y|X=x)
$$

**Information Gain (a.k.a mutual information)**  

* choosing splits in ID3: **select the split S that most reduces the conditional entropy of Y for training set D**  
$$
InfoGain(D,S) = H_D(Y) - H_D(Y|S)
$$  
$D$ indicates that we’re calculating probabilities using the specific sample $D$

**Gain Ratio**  
Information gain will be biased towards tests with many outcomes, and **Gain ratio** is introduced to address this limitation.

$$
GainRatio(D,S) = \frac{InfoGain(D,S)}{H_D(S)} = \frac{H_D(Y)-H_D(Y|S)}{H_D(S)}
$$

Gain ratio normalizes the information gain by the entropy of the split being considered.

##### Overfitting

**Overview**  
**Causes:**  
1.  Noise in the training data
2.  Limited size of the training data, causing difference from the true distribution.
3.  Larger the hypothesis class, easier to find a hypothesis that fits the difference between the training data and the true distribution.

**Solutions:**  
1.  Clean training data.
2.  Collect more training data.
3.  Occam's Razor, get rid of the unnecessary hypotheses.

**Avoiding overfitting in Decision Tree learning**  
1.  **Early stopping**: Stop if further splitting not justified by a statistical test.
    *    ID3
2.  **Post-pruning**: Grow a large tree, then prune back some nodes.
    *    More robust to myopia of greedy tree learning
    *    Example: C4.5 pruning
            1.  Split given data into training and validation(tuning) sets.
            2.  Grow a complete tree.
            3.  Do until further pruning is harmful
                *   Evaluate impact on tuning-set accuracy of pruning each node.
                *   Greedily remove the one that most improves tuning-set accuracy.
                
**Validation Sets (a.k.a tuning set)**  
Subset of the training set that is held aside
* Not used for primary training process.
* Used to select among models.

##### Variants

**Regression Trees**  
**Overview**  
Leaves have functions that predict numeric values instead of class labels.

**CART**

* CART does least squares regression which tries to minimize  
$$
\begin{aligned}
&\,\,\,\,\,\,\,\,\,\,\,\sum_{i=1}^{|D|}(y^{(i)} - \hat{y}^{(i)})^2 \\
&= \sum_{L\in leaves}\sum_{i\in L}(y^{(i)} - \hat{y}^{(i)})^2
\end{aligned}
$$  
* At each internal node, CART chooses the split that most reduces this quantity.

**Probability estimation trees**  
**Overview**  
Leaves estimate the probability of each class.

**m-of-n splits**  
Found via a hill-climbing search
* Initial state: best 1-of-1 (ordinary) binary split
* Evaluation function: Information gain
* Operators:
    *   m-of-n $\rightarrow$ m-of-(n+1)
    *   m-of-n $\rightarrow$ (m+1)-of-(n+1)
    
**Lookahead Algorithm**  
**Overview**  
Aims to alleviate a shortcoming of the hill-climbing search, that is myopia. The meaning of myopia is that an important feature may not appear to be informative until used in conjunction with other features.

**Pseudocode**  
**OrdinaryFindBestSplit**(set of training instances $D$, set of candidate splits $C$):  
$\quad\quad$maxgain = -$∞$  
$\quad\quad$for each split $S$ in $C$  
$\quad\quad \quad\quad$gain = **InfoGain**($D$, $S$)   
$\quad\quad \quad\quad$if gain > maxgain  
$\quad\quad \quad\quad$maxgain = gain  
$\quad\quad S_{best}$ = $S$  
$\quad\quad$return $S_{best}$  

**LookaheadFindBestSplit**(set of training instances $D$, set of candidate splits $C$):  
$\quad\quad$maxgain = -$∞$  
$\quad\quad$for each split $S$ in $C$  
$\quad\quad \quad\quad$gain = **EvaluateSplit**($D$, $S$)   
$\quad\quad \quad\quad$if gain > maxgain  
$\quad\quad \quad\quad$maxgain = gain   
$\quad\quad S_{best} = S$  
$\quad\quad$return $S_{best}$

**EvaluateSplit**($D$, $C$, $S$):  
$\quad\quad$if a split on $S$ separates instances by class (i.e. $H_D(Y|S) = 0$):  
$\quad\quad \quad\quad$**\#** <u>*no need to split further*</u>  
$\quad\quad \quad\quad$ return $H_D(Y) - H_D(Y|S)$  
$\quad\quad$else:  
$\quad\quad \quad\quad$for each outcome $k$ of $S$:  
$\quad\quad \quad\quad$**\#** <u>*see what the splits at the next level would be*</u>  
$\quad\quad D_k$ = subset of instances that have outcome k     
$\quad\quad S_k$ = **OrdinaryFindBestSplit**($D_k$, $C$ – $S$)  
$\quad\quad \quad\quad$**\#** *<u>return information gain that would result from this 2-level subtree</u>*  
$\quad\quad \quad\quad$return $H_D(Y)-(\sum_k\frac{|D_k|}{|D|}H_{D_k}(Y|S=k, S_k))$  

##### Comments
1.  Feature will not be reused.