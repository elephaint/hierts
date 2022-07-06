# Installation #
We recommend installation of hierTS in a (virtual) Python environment via PyPi.

Don't know how to setup a Python environment and/or access a terminal? We recommend you start by installing [Anaconda](https://docs.anaconda.com/anaconda/install/index.html). This allows you to create Python virtual environments and access them via a terminal. 

##  Installing via PyPi ##
Execute the following commands in a terminal after you have activated your Python (virtual) environment.

`pip install hierts`
  
## Installing from source ##
Clone the repository, build the package and run `pip install .` in the newly created directory `hierts`, i.e. run the following code from a terminal within a Python (virtual) environment of your choice:

  ```
  git clone https://github.com/elephaint/hierts.git
  cd hierts
  py -m build
  pip install .
  ```

## Dependencies ##
* [`pandas>=1.3.5`](https://pandas.pydata.org/getting_started.html)
* [`numba>=0.53.1`](https://numba.readthedocs.io/en/stable/user/installing.html) 

Installing via PyPi and via source as listed above will install the dependencies automatically.

## Verification ##
To verify, download & run an example from the examples folder to verify the installation is correct:
* Run [this example](https://github.com/elephaint/pgbm/blob/main/examples/example_tourism.py) to verify on the tourism dataset.
* Run [this example](https://github.com/elephaint/pgbm/blob/main/examples/example_reconciliation.py) to verify on the prison dataset.