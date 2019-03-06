# Stochastic Allocation Simulation (SAS)

Using `cloudsim` to simulate a stochastic resource allocation scheme.
See the full paper for more information:

Liran Funaro, Orna Agmon Ben-Yehuda, and Assaf Schuster. 2019. Stochastic Resource Allocation. In _Proceedings of the 15th ACM SIGPLAN/SIGOPS International Conference on Virtual Execution Environments (VEE ’19), April 14, 2019, Providence, RI, USA_. ACM, New York, NY, USA, 15 pages. https://doi.org/10.1145/3313808.3313815


# Install (beta)
Install `g++-8`:
```bash
apt-get install g++-8 
```

Download and install dependencies by cloning the following repositories:

 * vecfunc: https://bitbucket.org/funaro/vecfunc
 * cloudsim: https://bitbucket.org/funaro/cloudsim
 
Follow the `README.md` file in each of these repositories to install them properly.
 
Finally, install the package in developer mode:
```bash
python setup.py develop --user
```


# Usage
The notebooks in the [notbooks](notebooks) folder are used to produce the results seen in the paper. 


# License
[GPL](LICENSE.txt)