# Time Series Classification Based on Temporal Features

**This work has been accepted by "Applied Softe Computing".**  [paper link](https://www.sciencedirect.com/science/article/abs/pii/S1568494622005889)

Along with the widespread application of Internet of things technology, time series classification have been becoming a research hotspot in the field of data mining for massive sensing devices generate time series all the time. However, how to accurately classify time series based on intuitively interpretable features is still a huge challenge. For this, we proposed a new Time Series Classification method based on Temporal Features (TSC-TF). TSC-TF firstly generates some temporal feature candidates through time series segmentation. And then,  TSC-TF selects temporal feature according the importance measures with the help of a random forest.  Finally, TSC-TF trains a fully convolutional network to obtain high accuracy. Experiments on various datasets from the  UCR time series classification archive demonstrate the  superiority of our method.

## Code
The code for our method is in package **src**
## Experiements
The code for experiments is in packages **experiments**.
## Transformed data
The transformed data is in fold **trans**. Please unzip the rar file before using these files.
## Results
The experimental results are in fold **result**.

## Acknowledgements
This work was supported by the Innovation Methods Work Special Project under Grant 2020IM020100, and the Natural Science Foundation of Shandong Province under Grant ZR2020QF112.

We would like to thank Eamonn Keogh and his team, Tony Bagnall and his team for the UEA/UCR time series classification repository.
