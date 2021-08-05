# Multiple Kernel Transfer Clustering (MKTC)

The MKTC method clusters a data set by utilizing weakly supervised information provided on a multi-instance subset of the data set.

The MKTC method can be described in terms of two tasks:

- A _Source Task_: Where a multiple kernel metric is learnt while clustering the weakly supervised multi-instance subset.

- A _Target Task_: Where the learnt multiple kernel metric is used to cluster the original data set.

![mktc_overview](https://raw.githubusercontent.com/Avisek20/MKTC/master/figs2.png)

MKTC has a computation complexity _linear_ in the size of the dataset, making them suitable for the clustering of large datasets.

**demo.ipynb** contains an example of using MKTC on the sklearn digits data set.

**mktc.py** contains the implemetation of MKTC.

