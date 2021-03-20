# Solving Hyperbolic Equations via Reinforcement Learninig

## Motivation
Numerically solving a PDE that contains the "shock" solution suffers from osilliation. 
The two animation below show the numerical solutions of Burger's Equation over time. 
 
- The result when applying the trivial weights to 3rd order polynomials
![trivial-animation](./assets/trivial-animation.gif)

- The result when applying the WENO scheme
![weno-animation](./assets/weno-animation.gif)

## References
[0] Yufei Wang, Ziju Shen, Zichao Long & Bin Dong. (2020). Learning to Discretize: Solving 1D Scalar Conservation Laws via Deep Reinforcement Learning. Communications in Computational Physics. 28 (5). 2158-2179. doi:[10.4208/cicp.OA-2020-0194](https://global-sci.org/intro/article_detail/cicp/18408.html). <br/>
[0] Farahmand, Amir-massoud & Nabi, Saleh & Grover, Piyush & Nikovski, Daniel. (2016). Learning to Control Partial Differential Equations: Regularized Fitted Q-Iteration Approach. [10.1109/CDC.2016.7798966](https://ieeexplore.ieee.org/document/7798966). <br/>
[0] Farahmand, Amir-massoud & Nabi, Saleh & Nikovski, Daniel. (2017). Deep reinforcement learning for partial differential equation control. 3120-3127. [10.23919/ACC.2017.7963427](https://ieeexplore.ieee.org/document/7963427). <br/>

