# second_stage
During the second stage we were challenged by the problem of identification of floors and columns for each window. The task was quiet tricky as houses were of different scale and on different images and some images were under fish eye distortion.
To overcome these issues we:
1. substituted 3 types of windows with the only type to increase accuracy of bounding box prediction.
2. trained YOLOv8 on updated dataset.
3. obtained prediction for each window form dataset and derived (x,y) coordinates for each window from each window
4. we splitted (x,y) into x and y groups. Thus x-values represented floors and y-values represented columns.
5. finally, we applied clusterization algorithms: hierarchical clustering, affine propagation, optics, to x-values and y-values separately to identify floors and columns. After that we reduced too big numeber of classes as they exceeded real number of floors and columns in houses.

Here is nice visualisation of our outcomes.

![alt text](https://github.com/REDISKA3000/urbancode_2023/blob/1e269bf530fbf07d182ca02fe0cf67caf7fd455a/second_stage/results/inf_st2.jpg)
![alt text](https://github.com/REDISKA3000/urbancode_2023/blob/1e269bf530fbf07d182ca02fe0cf67caf7fd455a/second_stage/results/inf_stg2_3.jpg)
