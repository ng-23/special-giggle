# core
This subpackage contains the main modules and runnable scripts.

## Approaches
### Alpha (`alpha.py`)
Alpha relies solely on the timeseries precipitation data. We perform extensive feature engineering on the precipitation data to include more relevant features (season, monthly precipitation, etc.) but make no use of the images. 

We introduce 3 types of features: local, regional, and global. Local features are specific to an individual location (event_id). To derive regional features, we use K-Means clustering where the centroids are the barycenters of the Dynamic Time Warping (DTW) between time series. Lastly are global features, which are computed across all locations/regions. With these 3 types of features, we hope to capture trends in precipitation at varying granularities.

Urban floods are inherently rare events, which results in the dataset being highly imbalanced. To combat this, we calculate class and sample weights.

This is by far our simplest, fastest, and cheapest (computationally) approach, though it may not yield the best performance since the image data is not considered by the classifier.

## Helpful Resources
- [fusilli - Fusion Model Explanation](https://fusilli.readthedocs.io/en/latest/fusion_model_explanations.html)