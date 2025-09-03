TrackGANN is a GANN PyTorch-based package which adopted by St. Omelianchuk Savelii. It is NN for tracks by events in the SPD experiment in 2025 year. TrackGANN model is the Graph Attention Neural Network with an Encoder and an Edges Classifier. You can read comprehensive in the "ArticleLink"

This packege is realized logic for working with CSV files with the format:
"x_cord, y_cord, z_cord, trackID, eventID, leftTimeBound, rightTimeBound"

The network is trained hierarchically using the "time_resolution" parameter. This parameter controls the temporal resolution of the detector. The pre-trained models for different "time_resolution" values were trained sequentially for an average of 15 events per time slice. The models, datasets, and quality metrics are available in the "SPD_15EVENT" and "ML_15EVENT" folders.

You can find the examples of training and evaluating for TrackGANN in the folder "Examples". Examples for configuration files for testing and training you can find in the folder "configs". 


