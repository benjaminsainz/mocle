# MOCLE
Multi-objective Clustering Ensemble

Source code of **MOCLE** [1],[2], a state-of-the-art Pareto-based evolutionary clustering algorithm by Katti Faceli, Marcilio C.P. de Souto, Daniel S.A. de Araújo, and André C.P.L.F. de Carvalho. The algorithm is based upon the search strategy of NSGA-II, and for this implementation we used the source code available in [3]. MOCLE starts by creating an initial population with conceptually diverse clustering algorithms using varying hyper-parameters and cluster numbers in the range of *k* to *2k*. The initial population is constituted using k-means, Single and Average link Agglomerative Clustering from the Scikit-learn library [4], and Shared Nearest Neighbors clustering method with diverse parameters [5]. The crossover operator for recombination selected randomly the number of clusters of the resulting child from the cluster interval of the parents and is performed by Strehl and Ghosh’s Meta-clustering Algorithm using Kultzak’s version [6], avoiding clones. MOCLE returns a set of solutions representing each region of the Pareto front obtained from the optimization of two complementary objective functions to be minimized: deviation and connectivity.

MOCLE is available in this repository in a Python implementation.

# Algorithm hyper-parameters
``X``: an array containing the dataset features with no header. Each row must belong to one object with one column per feature.  
``n_clusters``: int with the number of desired clusters.  
``runs`` (default = 10): independent runs of the algorithm.  
``data``: a string with the name of the dataset used for printing the algorithm initialization and naming the output file.  
``y`` (default = None): one-dimensional array with the ground truth cluster labels if available.  
``max_gens`` (default = 50): maximum generations in the evolutionary process.   
``k_range`` (default = True): boolean to generate clusters in the range *k* to *2k* or perform clustering exclusively with the n_clusters hyper-parameter provided.  
``representation`` (default = 'label'): we include a label-based representation and is the only available option in this implementation.  
``pareto_plot`` (default = False): display the pareto plot with color-coded fronts along the evolutionary process.

### Optional data retrieval function
An additional data retrieval function is included for easy access and generation of the parameters X, clusters and data along with multiple datasets ready to be clustered, which can be used as a reference for preparing your data. The function will use the datasets included in the path ``/data`` and returns the data string, the X features, and the dataset's number of reference classes (n_clusters). The only parameter for this function is a string with a dataset name from the options. To run it on Python and get the information of the *wine* dataset, run these commands in the interface.     
``>>> from retr import *``  
``>>> data, n_clusters, X, y = data_retrieval('wine')``  

Label files are included for every dataset for any desired benchmarking tests.

# Setup and run using Python
Open your preferred Python interface and follow these commands to cluster a dataset using MOCLE. To execute it, just import the functions in *gen.py* and run ``run_mocle()`` with all of its hyper-parameters. See the example code below, which follows the data, n_clusters, X, and y variables set previously for the *wine* dataset.  
**Important**: You will need to have previously installed some basic data science packages such as numpy, pandas, matplotlib, seaborn, and Sci-kit Learn).

``>>> from gen import *``  
``>>> run_mocle(X, n_clusters, runs=10, data=data, y=y, max_gens=50, k_range=True, representation='label', pareto_plot=False)``  

Running these commands will execute MOCLE using the wine dataset's features, 3 clusters, with 50 generation for 10 independent runs, and will compute the adjusted RAND index between the solutions and the provided y array. A .csv file with the clustering and the results is stored in the ``/mocle-out`` path.

A test.py file is provided for a more straight-forward approach to using the algorithm.  

I hope this implementation to this great method is useful for your clustering tasks,

Benjamin  
**LinkedIn:** https://www.linkedin.com/in/benjaminmariosainztinajero/

# References
[1] K. Faceli, A. C. De Carvalho, and M. C. De Souto, “Multi-objective clustering ensemble,” International Journal of Hybrid Intelligent Systems, vol. 4, no. 3, pp. 145–156, 2007.  
[2] K. Faceli, M. C. de Souto, D. S. de Araujo, and A. C. de Carvalho, “Multi-objective clustering ensemble for gene expression data analysis,” Neurocomputing, vol. 72, no. 13-15, pp. 2763–2774, 2009.  
[3] H. A. Khan, “Nsga-ii,” https://github.com/haris989/NSGA-II, 2017.  
[4] F.Pedregosa,G.Varoquaux,A.Gramfort,V.Michel,B.Thirion,O.Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Pas- sos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, “Scikit- learn: Machine learning in Python,” Journal of Machine Learning Re- search, vol. 12, pp. 2825–2830, 2011.  
[5] A. Espin, “Snn-clustering,” https://github.com/albert-espin/snn-clustering, 2019.  
[6] A. Kultzak, “Mcla,” https://github.com/kultzak/MCLA, 2019.
