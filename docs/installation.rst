.. _installation:


Installation
===================================

TREASMO is written in Python and is available from PyPI. We recommend to use Python 3.10.

Install from PyPI
---------------------

Please run the following command. Creating new environment using conda is recommended.

.. code-block:: bash
    
        pip3 install treasmo

        # For all the functions, please also install leidenalg and minisom
        pip3 install leidenalg
        pip3 install minisom


If the above method failed, please try create a new environment using conda first then re-install the three packages above.  
  
If things still doesn't work, try the method below.

Use the package locally
-------------------------

If there is conflicts with other packages or something unexpected happened, try download the package and use it locally in a new conda environment.

.. code-block:: bash

    # Create a new conda environment
    conda create -n treasmo numpy=1.22
    conda activate treasmo

    # Setting up kernels for Jupyter Notebook if needed
    conda install ipykernel
    ipython kernel install --user --name=treasmo

    # Install dependencies
    conda install -c conda-forge scanpy python-igraph leidenalg
    conda install -c conda-forge libpysal
    conda install -c conda-forge hdf5plugin
    conda install -c conda-forge muon
    conda install -c conda-forge fa2
    conda install -c conda-forge louvain

    # Option 1 to install minisom
    git clone https://github.com/JustGlowing/minisom.git
    cd minisom
    python setup.py install

    # Option 2 to install minisom
    pip3 install minisom

    # install Homer if needed




  





