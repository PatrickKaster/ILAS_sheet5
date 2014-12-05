Exercise 5
==========

This is going to be the solution for exercise 5 of the
[Machine Learning Lecture][machine_learning] at the University of Bonn.

Contributors:
* Timm Behner
* Philipp Bruckschen
* Patrick Kaster
* Markus Schwalb

style file: [HMC Mathematics Homework Class]

[machine_learning]: http://www-kd.iai.uni-bonn.de/index.php?page=teaching_details&id=83
[HMC Mathematics Homework Class]: https://www.math.hmc.edu/computing/support/tex/classes/hmcpset/

Implementation of k-Nearest-Neighbor and Leave-One-Out
--------------------------------------------------------
None of the group members was willing to do this by hand so we implemented the
k-Nearest-Neighbor algorithm and the Leave-One-Out cross validation in python.
The algorithms are contained in knn.py and leaveoneout.py. Both depend on the
Example class and the ExampleSet class.
To be able to import the other files create an empty file caled __init__.py in
the same folder.
