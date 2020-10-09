# FiniteElementProject
Project for TMA4220. This version is part one of the project. 

This group members for this project are: 
 * Håkon Noren 
 * Alexander Johan Arntzen 

The source files for the solver and quadrature methods are placed in the `part1/` directory. 
The tests and figure generating methods are placed in `test/` directory. 

To run all the test run: 

```
python3 -m unittest
```
from in the top level directory. 

Alternatively  append `test.test_quadrature.TestQuadratureMethods`, `test.test_solver.TestHomogeneousDirichlet`,  `test.test_meshing.TestMeshing`
or `test.test_solver.TestSolverNeumann` to run the test corresponding to each task. 
For example: 
```
python3 -m unittest test.test_quadrature.TestQuadratureMethods
```
Then the figures will be generated into the `figures/` directory 
