# core
This subpackage contains the main modules and runnable scripts.

## Approaches
We came up with multiple general approaches to the problem, attempting to leverage both the timeseries and image data for optimal results. We used the Optuna Python library to aid in optimal hyperparameter discovery and a seed of <> to control randomness. All code was ran on a machine with <> GB of RAM, a <> processor, and a <> GPU. 

### Alpha (`alpha.py`)
Alpha relies solely on the timeseries precipitation data. We perform extensive feature engineering on the precipitation data to include more relevant features (season, monthly precipitation, etc.) but make no use of the images. This is by far the simplest, fastest, and cheapest (computationally) approach, though may not yield the best performance since the image data is not considered by the classifier.

### Beta (`beta.py`)
Beta extends Alpha, leveraging not just the feature-engineered precipitation data but also multi-band composite images. We create these composite images by combining the spectral bands and computing spectral indices, so each time step has multiple images associated with it. These composite images each emphasize different relevant information, such as vegetation presence/health, soil bareness, and more.

To handle the multimodal nature of the input data, we first pass the composite images through a pretrained DCNN, which serves as a feature extractor when discarding the last (classification) layer. We then apply PCA to each composite image's feature vector to reduce the dimensionality. Finally, we concatenate each image's feature vector with the precipitation data, combining the different modalities into one dataset on which a classifier can be trained. This approach is much more complex and computationally expensive than Alpha, though may yield better peformance because of the additional features obtained from the composite images.

### Charlie (`charlie.py`)
Charlie is a hybrid of Beta. The feature-engineered precipitation data and composite images are both still used as inputs. Unlike Beta, we train 2 classifiers, one for each data modality. 

We frame the problem as binary classification (flood or no flood) for the precipitation data model. Because the precipitation data is a timeseries, the spatiotemporal features are already explicitly present in the input data, so no special handling is needed. For the image data model, the images themselves do not explicitly contain any spatiotemporal data, which is important information for the problem. We thus frame the problem as a multi-task learning problem, wherein the model contains classifiers for the presence of a flood, the geographical location (spatial), and the day of the year (temporal, 0 to 364 for both years). To prevent data leakage and ensure temporal structure during training/validation, we perform k-fold cross-validation with a sliding window. Each fold has its own window with 1 month excluded from training/validation splits, and subsequent folds include that month while excluding others.

Finally, we leverage decision fusion to combine the logits from both models and produce the final probability of a flood event occurring at a given location and on a given time. This is arguably the most complex approach and would likely require the most computational resources since 2 models must be trained and used for inference.