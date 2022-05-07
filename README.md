# MOCLE (Multi-objective Clustering Ensemble)

**Coded by:** Benjamin M. Sainz-Tinajero.  

Source code of **MOCLE** [1],[2], a state-of-the-art Pareto-based evolutionary clustering algorithm by Katti Faceli, Marcilio C.P. de Souto, Daniel S.A. de Araújo, and André C.P.L.F. de Carvalho. The algorithm is based upon the search strategy of NSGA-II, and for this implementation we used the source code available in [3]. MOCLE starts by creating an initial population with conceptually diverse clustering algorithms using varying hyper-parameters and cluster numbers in the range of *k* to *2k*. The initial population is constituted using k-means, Single and Average link Agglomerative Clustering from the Scikit-learn library [4], and Shared Nearest Neighbors clustering method with diverse parameters [5]. The crossover operator for recombination selected randomly the number of clusters of the resulting child from the cluster interval of the parents and is performed by Strehl and Ghosh’s Meta-clustering Algorithm using Kultzak’s version [6], avoiding clones. MOCLE returns a set of solutions representing each region of the Pareto front obtained from the optimization of two complementary objective functions to be minimized: deviation and connectivity.

MOCLE is available in this repository in a parallelized Python implementation.

# Data Preparation
This implementation requires one mandatory file to perform clustering. A ``.csv`` file named ``iris_X.csv``, for instance, must contain the features of a dataset with no header, one column per attribute and one row per object. A second optional file could contain ground truth labels in case you're running a benchmark and need to compute the Adjusted RAND Index of a solution against a reference mask or partition. This second file must be named ``iris_y.csv`` (for this example) and must have one column with the same number of objects as the ``iris_X.csv`` file (no header), placing each object into one group. Our algorithm automatically searches for both of this files in the ``\data`` path and computes the Adjusted RAND Index only if it finds ground truth labels in a file as mentioned before. We include 40 publicly available datasets complying with our implementation's required data format. 

# Hyper-parameter Setting
``data``: a string with the name of the dataset to be retrieved without the ``_X.csv`` or ``_y.csv`` suffixes.   
``n_clusters``: integer with the number of required clusters. As an alternative, setting this argument as ``'auto'`` will set the number of clusters found in the ground truth file as ``n_clusters`` (this feature only works if there is a ``_y.csv`` ground truth file in the ``\data`` directory).  
``runs`` (default = 10): independent runs of the algorithm.  
``max_gens`` (default = 50): maximum generations of the evolutionary process.    

For more information on the hyper-parameters and their influence in the evolutionary process, we refer the user to the article in Ref.[1].  

# Setup and Run using Python
Open your preferred Python interface and follow these commands to generate a clustering using MOCLE. We will continue using the ``iris`` dataset as an example.  

``>>> from gen import *``  
``>>> run_mocle(data='iris', n_clusters=3, runs=10, max_gens=50)``

Running these commands will execute MOCLE using the ``iris`` dataset's features with 3 clusters, 50 generations, and 10 independent runs, and will compute the Adjusted RAND Index between the solutions and the reference labels in the ``iris_y.csv`` file. A ``.csv`` file with the clustering and the results is stored in the ``/mocle-out`` path.

An ``example.py`` file is provided with this example for a more straight-forward approach to using the algorithm.  

**Important**: You will need to have previously installed some basic data science packages such as NumPy, Pandas, Metis, Tables, and Scikit-learn.

I hope our implementation to this great method is useful for your clustering tasks,

Benjamin  
**LinkedIn:** https://www.linkedin.com/in/benjaminmariosainztinajero/  
**Email:** sainz@tec.mx, bm.sainz@gmail.com

# References
[1] K. Faceli, A. C. De Carvalho, and M. C. De Souto, “Multi-objective clustering ensemble,” International Journal of Hybrid Intelligent Systems, vol. 4, no. 3, pp. 145–156, 2007.  
[2] K. Faceli, M. C. de Souto, D. S. de Araujo, and A. C. de Carvalho, “Multi-objective clustering ensemble for gene expression data analysis,” Neurocomputing, vol. 72, no. 13-15, pp. 2763–2774, 2009.  
[3] H. A. Khan, “Nsga-ii,” https://github.com/haris989/NSGA-II, 2017.  
[4] F.Pedregosa,G.Varoquaux,A.Gramfort,V.Michel,B.Thirion,O.Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Pas- sos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, “Scikit- learn: Machine learning in Python,” Journal of Machine Learning Re- search, vol. 12, pp. 2825–2830, 2011.  
[5] A. Espin, “Snn-clustering,” https://github.com/albert-espin/snn-clustering, 2019.  
[6] A. Kultzak, “Mcla,” https://github.com/kultzak/MCLA, 2019.
