#### Recursive partitioning for heterogeneous causal effects

##### Introduction
*   Two focused problems
    *   Estimating heterogeneity by covariates or features in causal effects in experimental or observational studies.
    *   Conducting inference about the magnitude of the differences in treatment effects across subjects of the population.
*   Definition of **Causal Effect**  
    *Comparsions between outcomes we observe and counterfactual outcomes we would have observed under a different regime or treatment.*
*   Definition of **Honesty**  
    *A model is "**Honesty**" if it does not use the same information for selecting the model structure (in paper's setting, the partition of the covariate space) as for estimatino given a model structure.*
*   How to acheieve "Honesty"?
    Splitting the training sample into two parts, one for constructing the tree (including the cross-validation step) and a second for estimating treatment effects within leaves of the tree.

##### The Problem
*   **Setup**
    *Given*:
    *   **$N$ units**, indexed by $i=1,\ldots,N$. Each unit has a pair of potential outcomes $(Y_i(0), Y_i(1))$.
    *   **Unit-level causal effect ($\tau_i$)**: $\tau_i = Y_i(1) - Y_i(0)$.
    *   $W_i\in\{0,1\}$ be the **binary indicator for the treatment**
        *   $W_i=0$ indicating that unit $i$ received the control treatment.
        *   $W_i=1$ indicating that unit $i$ received the active treatment.
    *   The realized outcome for unit $i$ is the potential outcome corresponding to the treatment received
        $$
        Y_i^{obs} = Y_i(W_i) = 
        \begin{cases}
            &Y_i(0)\quad \text{if}\,\,W_i=0, \\
            &Y_i(1)\quad \text{if}\,\,W_i=1, \\
        \end{cases}
        $$
    *   $X_i$ be a $K$-component vector of features, covariates, or pre-treatment variables, known not to be affected by the treatment.  

    In summary, we are given i.i.d data from a large population consist of the triple $(Y_i^{obs}, W_i, X_i)$, for $i = 1, \ldots, N$.
    *   Definition of **"propensity score"**  
        Conditional treatment probability:
        $$
        e(x) = pr(W_i=1\lvert X_i=x)
        $$
*   **Unconfoundedness**
    *   Assumption 1 (Unconfoundedness)
        $$
        W_i \perp \!\!\!\perp (Y_i(0), Y_i(1))\lvert X_i
        $$
        Using symbol $\perp \!\!\!\perp$ to denote (conditional) independence of two random variables. This assumption is **satisfied** in a randomized experiment without conditioning on covariates but also may be justified in observational studies ***if the researcher is able to observe all of the variables that affect the unit’s receipt of treatment and are associated with the potential outcomes***.
    *   Conditional Average Treatment Effects and Partitioning
        $$
        \tau(x) \equiv\mathbb{E}[Y_i(1) - Y_i(0)\lvert X_i=x]
        $$

##### Honest Inference for Population Averages
What's the difference between causal tree and other two approaches - conventional classification and regression trees (CART)?
1.  Focus on estimating conditional average treatment effects rather than predicting outcomes.
2.  Impose a separation between constructing the partition and estimating effects with leaves of the partition, using separate samples for the two tasks, in what is referred to as **honest estimation**
*   Setup
    *   Defining key concepts and functions
        *   A tree or partitioning $\Pi$ corresponds to a partitioning of the feature space $\mathbb{X}$, with $\#(\Pi)$ the number of elements in the partition.  
            $$
            \Pi = \{l_1, \ldots, l_{\#(\Pi)}\}, \,\,\text{with}\,\,\cup_{j=1}^{\#(\Pi)}l_j = \mathbb{X}
            $$
        *   $\mathbb{P}$ denote the space of partitions.  
        *   $l(x;\Pi)$ denote the leaf $l\in\Pi$ such that $x\in l$.  
        *   $\mathbb{S}$ denote the space of data samples from a population.  
        *   $\pi: \mathbb{S} \mapsto \mathbb{P}$ be an algorithm that on the basis of a sample $\mathcal{S}\in\mathbb{S}$ constructs a partition. 
        *   Conditional mean function $\mu(x; \Pi)$
            $$
            \mu(x;\Pi) \equiv \mathbb{E}[Y_i\lvert X_i\in l(x;\Pi)] = \mathbb{E}[\mu(X_i)\lvert X_i\in l(x;\Pi)]
            $$
            which can be viewed as a **step-function approximation to $\mu(x)$**.
        *   Given a sample $\mathcal{S}$, the estimated counterpart of conditional mean function is,
            $$
            \hat{\mu}(x; \mathcal{S}, \Pi) \equiv \frac{1}{\#(i\in\mathcal{S}: X_i\in l(x;\Pi))}\sum_{i\in\mathcal{S}:X_i\in l(x;\Pi)} Y_i
            $$
            which is unbiased for $\mu(x;\Pi)$.
*   The Honest Target
    Using **MSE criteria** to compare alternative estimators.
    *   Definition of MSE given a partition $\Pi$: average over a test sample $\mathcal{S}^{te}$ and the conditional mean is estimated on an estimation sample $S^{est}$.
        $$
        \text{MSE}_{\mu}(\mathcal{S}^{te}, \mathcal{S}^{est}, \Pi) \equiv \frac{1}{\#(\mathcal{S}^{te})} \sum_{i\in\mathcal{S}^{te}}\{(Y_i-\hat{\mu}(X_i;\mathcal{S}^{est}, \Pi))^2 - Y_i^2\}
        $$
    *   Adjusted expected MSE: expectation of $MSE_{\mu}(\mathcal{S}^{te}, \mathcal{S}^{est}, \Pi)$ over test and estimation samples.
        $$
        \text{EMSE}_{\mu}(\Pi) \equiv\mathbb{E}_{\mathcal{S}^{te}, \mathcal{S}^{est}}[\text{MSE}_\mu(\mathcal{S}^{te}, \mathcal{S}^{est}, \Pi)]
        $$
    The ultimate goal is to construct and assess algorithms $\pi(\cdot)$ that maximize the honest criterion
    $$
    Q^H(\pi) \equiv -\mathbb{E}_{\mathcal{S}^{te}, \mathcal{S}^{est}, \mathcal{S}^{tr}}[\text{MSE}_\mu(\mathcal{S}^{te}, \mathcal{S}^{est}, \pi(\mathcal{S}^{tr}))]
    $$

*   The Adaptive Target
    *   Conventional criterion in CART, referred as adaptive
        $$
        Q^{C}(\pi)\equiv -\mathbb{E}_{\mathcal{S}^{te}, \mathcal{S}^{tr}}[\text{MSE}_\mu(\mathcal{S}^{te}, \mathcal{S}^{tr}, \pi(\mathcal{S}^{tr}))]
        $$
    *   Difference between adaptive target and honest target
        *   Different samples $\mathcal{S}^{tr}$ and $\mathcal{S}^{est}$ are used for construction of the tree and estimation of the conditional means.
        *   

*   The Implementation of CART
*   Honest Splitting
*   