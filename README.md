# Azul-Board-Game

*Repository under construction*

## Board Detection 

<img src="img/detection.png" width=600 alt="full pipeline"/>

Game boards are detected in two stages:

1. Image segmentation of the board using YOLO v8
2. Perspective undistort with ORB feature matching against a board pattern

<img src="img/feature_matching.png" width=600 alt="feature matching"/>

## Game UI

<img src="img/ui.png" width=600 alt="ui"/>
