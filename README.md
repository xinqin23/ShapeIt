# ShapeIt
A tool for mining specifications of cyber-physical systems (CPS) from their real-valued behaviors.


#### Installation:
Command line to setup the tool:

##### Setup virtualenv
````
cd ShapeIt
virtualenv -p python3 venv 
source venv/bin/activate
pip3 install -r requirements.txt
````


#### Use:
To use the tool, just run the main file (main.py).
In the last line of this file, specify the data type (one between 'meat', 'fish', 'wine'), 
the maximum WSCC error and the maximum error in the segmentation.

For example:
````
if __name__ == '__main__':
    #main(data, max_delta, max_mse)
    main('meat', 10, 0.05)

````

#### Cite the Tool:   [![DOI](https://zenodo.org/badge/238563183.svg)](https://zenodo.org/badge/latestdoi/238563183)

