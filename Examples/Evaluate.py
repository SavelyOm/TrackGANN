import torch
import time
from pathlib import Path
import yaml
import pandas as pd
import os
from collections import defaultdict
import csv

from TrackGANN.src.model import TracksNN
from TrackGANN.src.dataset import SPDdataset, TrackMLdataset
from TrackGANN.src.config import ImportConfig
TESTCONFIG, configArgs = ImportConfig()

from TrackGANN.utils.loader import model_loader
from TrackGANN.utils.evaluation_functions import full_evaluate




' CONFIG PARAMETERS '
'_________________________________________________________________________'

# Log information
#=======================================================================#
name_of_directory = TESTCONFIG['dataset']['name_of_directory']
data_name = TESTCONFIG['dataset']['dataset_name']
log_label = TESTCONFIG['Log_parms']['log_label']
#=======================================================================#


# Detector parameters
#=======================================================================#
r_min = TESTCONFIG['detector']['r_min']
r_max = TESTCONFIG['detector']['r_max']
n_stations = TESTCONFIG['detector']['n_stations']
det_eff = TESTCONFIG['detector']['detector_eff']
#=======================================================================#


# Timeslice parameters
#=======================================================================#
time_resolution = TESTCONFIG['dataset']['time_resolution']
n_samples = TESTCONFIG['dataset']['n_samples']
mean_event = TESTCONFIG['timeslice']['mean_events']
max_tracks = TESTCONFIG['timeslice']['max_tracks']
#=======================================================================#

# Loader and DATASET generation
#=======================================================================#
model_dirrectory = TESTCONFIG['loader']['model_dirrectory']
load_model_name = TESTCONFIG['loader']['model_name']
trainMLfolder = TESTCONFIG['dataset']['MLtrackPath']
#=======================================================================#


# Model Parameters 
#=======================================================================#
hidden_linear_layers = TESTCONFIG['model']['encoder']['hidden_linear_layers']
out_linear_layer = TESTCONFIG['model']['encoder']['out_linear_layer']
encoder_convolutions = TESTCONFIG['model']['encoder']['n_convolutions']

edge_weight_layers = TESTCONFIG['model']['classifier']['edge_weight_layers']
edge_update_layers = TESTCONFIG['model']['classifier']['edge_update_layers']
node_update_layers = TESTCONFIG['model']['classifier']['node_update_layers']
classifier_convolutions = TESTCONFIG['model']['classifier']['n_convolutions']
#=======================================================================#
'_________________________________________________________________________'


# DATASET GENERATION 
#=======================================================================#
if data_name == 'SPD':
    dataset = SPDdataset(name_of_directory)
    dataset.SetDetectorParm(n_stations,
                                  det_eff,
                                  r_min,
                                  r_max)
if data_name == 'TrackML':
    dataset = TrackMLdataset(name_of_directory)
    dataset.SetFolders(trainMLfolder)
    
dataset.test_data_generation(n_samples, mean_event, max_tracks, time_resolution, name_of_directory)
graphs = dataset.convert_to_graph(name_of_directory)

#=======================================================================#



                



# Config logit
#=======================================================================#
config_path = Path(configArgs.config_dir) / Path(configArgs.config_name)
with open(config_path, "r") as f:
    config = yaml.safe_load(f)
with open(Path(name_of_directory) / configArgs.config_name, "w") as f:
    yaml.dump(config, f, sort_keys=False, default_flow_style=False)
#=======================================================================#



# MODEL INITIALIZATION
#=======================================================================#
device = torch.device(f'cuda')
model = TracksNN(hidden_linear_layers, out_linear_layer, encoder_convolutions,
                 edge_weight_layers, edge_update_layers, node_update_layers, classifier_convolutions).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True, weight_decay=1e-3) 
model, optimizer, new_epoch = model_loader(model, optimizer, model_dirrectory, load_model_name)
#=======================================================================#



# EVALUATING
#=======================================================================#
start_time = time.time()
result = full_evaluate(model, graphs, device, name_of_directory, log_label)
end_time = time.time()

print(f'time: {end_time-start_time} sec')
#=======================================================================#
