# Time Series Classification Based on Temporal Features

Along with the widespread application of Internet of things technology, time series classification have been becoming a research hotspot in the field of data mining for massive sensing devices generate time series all the time. However, how to accurately classify time series based on intuitively interpretable features is still a huge challenge. For this, we proposed a new Time Series Classification method based on Temporal Features (TSC-TF). TSC-TF firstly generates some temporal feature candidates through time series segmentation. And then,  TSC-TF selects temporal feature according the importance measures with the help of a random forest.  Finally, TSC-TF trains a fully convolutional network to obtain high accuracy. Experiments on various datasets from the  UCR time series classification archive demonstrate the  superiority of our method.

## Code
The code for our method is in package **src**
## Experiements
The code for experiments is in packages **experiments**.
## Transformed data
The transformed data is in fold **trans**. Please unzip the rar file before using these files.
## Results
The experimental results are in fold **result**.