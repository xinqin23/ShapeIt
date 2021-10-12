# ShapeIt
The tool.



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

