# TrackGANN

TrackGANN is a PyTorch-based GANN package developed by St. Omelianchuk Savelii. It is a neural network designed for track reconstruction of events in the SPD experiment in 2025. The TrackGANN model is a Graph Attention Neural Network consisting of an Encoder and an Edges Classifier. You can find a detailed description in the "ArticleLink".

This package implements logic for working with CSV files in the following format:
"x_cord, y_cord, z_cord, trackID, eventID, leftTimeBound, rightTimeBound"

The network is trained hierarchically using the "time_resolution" parameter, which controls the temporal resolution of the detector. The pre-trained models for different "time_resolution" values were trained sequentially using an average of 15 events per time slice. The models, datasets, and quality metrics are available in the "SPD_15EVENT" and "ML_15EVENT" folders.

You can find examples of training and evaluation for TrackGANN in the "Examples" folder. Configuration file examples for training and testing are available in the "configs" folder.

# Training

You can start training or evaluation using the commands:

`python Training.py/Evaluate.py -d Path\to\configs_folder -c ConfigName.yaml`

Here, `ConfigName` should be set to `TrainConfig` for training settings and `TestConfig` for evaluation settings.
