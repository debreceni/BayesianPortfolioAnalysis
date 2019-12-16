## Bayesian Portfolio Construction and Error Minimization
##### Author: David Debreceni
##### Advisor: William Rubens, PhD

### Problem Statement
##### Can a portfolio constructed using a Bayesian process vs a Linear Regression produce a smaller estimate error?

### High Level Process
1. Using Fama French 3 Factor Model and Momentum build Linear regression over 20 years of data
2. Record the portfolio, weights, expected returns and actual returns for each period
3. Calculate the ERROR in the return
4. Run the process again for Bayesian Regression using PYMMC3 library
5. Compare the ERROR estimates for each return period and report

# Files
There are 2 diffent analysis saved here.  
Files labeled _v3 process the regressions using all known data at that time.
Files labled _v4 process the regression comparing security returns [1:t] vs factors [0:t-1] with predictions done on factors t

Core code is stored in files named FinalProject, all outputs are saved after each analysis.
Files named WorkingWithResults are built to work with the outputs from each of the analyses.
