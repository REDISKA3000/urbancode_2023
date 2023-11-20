# first_stage
The task for the first stage was to make an algorithm that would be able to detect 3 types of windows from CCTV camera photos:
- $\color{#2899ED}{windows-without-brickwork}$
- $\color{#EDC928}{windows-with-brickwork}$
- $\color{#FF5733}{glazed-window}$ <br/>
![alt text](https://github.com/REDISKA3000/urbancode_2023/blob/04953dedef63a8cd11c8281e4f37fc05b4bb8f2d/first_stage/sample_data/sample_house.jpg) <br/>

To solve the problem we made 4 crucial steps:
1. we exended the dataset from 250 up to 400 unique images
2. applied severall augmentation techniques
3. trained YOLOv8
4. optimized hyperparameters to maximize MAP score <br/>

Ultimately, we achieved MAP is equal to 0.83, what was 3rd best score in leaderboard
![alt text](https://github.com/REDISKA3000/urbancode_2023/blob/6287391fd32b8515c307fbec21a23198a081dd70/first_stage/results/21_14_12_06_790792-2023-08-31_34382.jpg)
![alt text](https://github.com/REDISKA3000/urbancode_2023/blob/6287391fd32b8515c307fbec21a23198a081dd70/first_stage/results/43_04_09_57_227749-2023-09-06_34977.jpg)
