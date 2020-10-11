#### Generating survival times to simulate Cox proportional hazards models

###### Introduction
*   The Cox proportional hazards model
    
    $$
    h(t|x) = h_0(t)exp(\beta'x)
    $$

    *   t: time
    *   x: vector of covariates
    *   $\beta$: the vector of regression coefficients
    *   $h_0(t)$: baseline hazard function
*   Distribution used in survival time estimation
    *   Distribution that share the assumption of proportional hazards with the Cox regression model:
        *   Exponential distribution
        *   Weibull  distribution
        *   Gompertz distribution
    *   Not:
        *   Gamma distribution
        *   Lognormal distribution
        *   log-logistic distribution

###### Simulation Survival Times
*   General considerations
    *   Survival function of the Cox proportional hazards model
        $$
        S(t|x) = \exp[-H_0(t)\exp(\beta'x)]
        $$
        where
        $$
        H_0(t) = \int_0^th_0(u)du
        $$
        is the cumulative baseline hazard function.
    *   The distribution function of the Cox model
        $$
        F(t|x) = 1 - exp[-H_0(t)\exp(\beta'x)]
        $$
    *   Derive the survival time $T$ of the Cox model   
        (**Inverse transform method**)
        1.  Let $Y$ be a r.v. with distribution $F$, then $U=F(Y)$ follows a uniform distribution on the interval from 0 to 1. In the mean time, $(1-U)\sim \mathbf{U}[0,1]$.
        2.  Let $T$ be the survival time of the Cox model,
            $$
            U = \exp[-H_0(T)\exp(\beta'x)]\sim\mathbf{U}[0,1]
            $$
            If $h_0(t) > 0$ for all $t$, then $H_0$ can be inverted and the survival time $T$ of the Cox model can be expressed as
            $$
            T = H_0^{-1}[-\log(U)\exp(-\beta'x)]
            $$
            where $U$ is a r.v. with $U \sim \mathbf{U}[0,1]$.
*   Proportional hazards models with other distributions
    *   Formulas for the survival time and the hazard function of Cox models using the exponential, the Weibull and the Gompertz distribution.
        | Characteristics | Cox-exponential | Cox-Weibull | Cox-Gompertz |
        | --------------- | --------------- | ----------- | ------------ |
        | Survival time   | $T = -\frac{\log(U)}{\lambda\exp(\beta'x)}$ |  $T = (-\frac{\log(U)}{\lambda\exp(\beta'x)})^{\frac{1}{v}}$ | $T = \frac{1}{\alpha}\log[1 - \frac{\alpha\log(U)}{\lambda\exp(\beta'x)}]$ |
        | Hazard function | $h(t\lvert x) = \lambda\exp(\beta'x)$ | $h(t\lvert x) = \lambda\exp(\beta'x)vt^{v-1}$ | $h(t) = \lambda\exp(\beta'x)\exp(\alpha t)$ |      
