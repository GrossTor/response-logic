To apply the response logic approach to the DREAM3Challenge4 and DREAM4challenge2 network challenges download the according data sets from
[GeneNetWeaver](http://gnw.sourceforge.net/dreamchallenge.html)
or via direct links: [DREAM4](http://gnw.sourceforge.net/resources/DREAM4%20in%20silico%20challenge.zip) / [DREAM3](http://gnw.sourceforge.net/resources/DREAM3%20in%20silico%20challenge.zip).  
Extract the zip files into the DREAM3 and DREAM4 subfolders respectively.
To run the inference call the `first.py` script within a Python 3.6 session.
You can score the inferred networks with the official [DREAMTools package](https://dreamtools.readthedocs.io/en/latest/). As of now (Jan. 2019) the tool does not support Python 3.6. Therefore install DREAMTools in an Python 2.7 environment and from it call the `DREAM3_scoring_py27.py` or the `DREAM4_scoring_py27.py` scripts to generate a scores-csv file.


