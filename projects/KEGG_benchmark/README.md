The scripts in this folder allow to apply the response logic on a synthetic data set that is generated from the ensemble of human [KEGG](https://www.kegg.jp/) pathways.  

First, you need to download and parse the KEGG pathways. This can be done by running the `KEGG_parsing.py` script. 
Then `first.py` will generate the synthetic data sets, infer networks from them and score them. The fraction of missing data as well as the fraction of misclassified data can be set in lines 271 and 272. Since the inference goes over the entire ensemble of human KEGG pathways it can take several hours to run, depending on the chosen number of cores in line 273 (every core will requires approximately 1.5 GiB Memory). Increase number of repeats in line 247 and line 270 for better statistics (but longer runtimes).  
Finally `second.py` can be used to visualize the results.




