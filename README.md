# Concept Bottleneck Models for Time Series with Applications to Human Activity Recognition

This code is supplementary to our paper 'Concept Bottleneck Models for Time Series with Applications to Human Activity Recognition' [1]. The paper presents a methodology for developing Concept Bottleneck Models [2] for time series. We propose desiderata for concept definition, a model architecture and mitigate the learning of spurious correlations by the model through 'sensor masking'. Sensor masking restricts the prediction of each concept to data from physically relevant sensors. Additionally, we propose validation methods for concept validity using data augmentation. 

This code presents the validation of our methodology through application to a use case from Human Activity Recognition (HAR) using the MHEALTH dataset [3,4]. In our paper we proposed a set of human-interpretable concepts at a class-level (see ```Concept Annotation.ipynb```). In our leave-one-subject-out experiments we found competitive predictive performance for the CBMs (85.26% accuracy with sensor masking, and 91.01% without), compared to the black box models (86.89%). Note, whether spurious correlations are truly absent requires further evaluation. However, the current results support our CBMs as potential alternative for black box models for safety-critical applications.

The architecture of our CBM is as follows:

<img width="600" alt="architecture" src="https://github.com/user-attachments/assets/17967fbd-3af9-4235-b043-8170de97cf8a" />

## Data
- The ```dataset/``` folder contains the sensor data with the corresponding activity label per subject
- The ```concepts/``` folder contains the concepts folders each with the corresponding concept labeling per subject
- The ```results/``` folder contains the results of the leave-one-subject-out cross validation and the augmentation experiments

## Jupyter notebooks
- ```Concept Annotation.ipynb``` is used to create the concept ```.csv``` files in the ```concepts``` folder
- ```Concept Bottleneck Model.ipynb``` is used to create and show the results of the concept bottleneck model
- ```Concept Bottleneck Model - No Sensor Masking.ipynb``` is used to create and show the results of the concept bottleneck model without sensor masking
- ```Black Box Model.ipynb``` is used to create and show the results of the black box model
- ```Visualize Augmentation Experiment.ipynb``` is used to visualize the results from the data augmentation experiments obtained in ```Concept Bottleneck Model.ipynb```.

## Scripts
- ```loso_cv.py``` is used to run the leave-one-subject-out cross validation experiments.
- ```augmentation_helpers.py``` contains functions used to support the data augmentation experiments in ```Concept Bottleneck Model.ipynb```.

## Requirements
- ```Python 3.11```
- ```requirements.txt```

## References
[1] TBD

[2] Koh, P.W., Nguyen, T., Tang, Y.S., Mussmann, S., Pierson, E., Kim, B., Liang, P.: Concept Bottleneck Models. In: Proceedings of the 37th International Conference on Machine Learning. pp. 5338–5348. PMLR (Nov 2020)

[3] Banos, O., Garcia, R., Holgado-Terriza, J.A., Damas, M., Pomares, H., Rojas, I., Saez, A., Villalonga, C.: mHealthDroid: A Novel Framework for Agile Development of Mobile Health Applications. In: International workshop on ambient assisted living. pp. 91–98. Springer (2014). https://doi.org/10.1007/978-3-319-13105-4_14

[4] Banos, O., Villalonga, C., Garcia, R., Saez, A., Damas, M., Holgado-Terriza, J.A., Lee, S., Pomares, H., Rojas, I.: Design, implementation and validation of a novel open framework for agile development of mobile health applications. Biomedical engineering online 14(Suppl 2), S6 (2015). https://doi.org/10.1186/1475-925X-14-S2-S6
