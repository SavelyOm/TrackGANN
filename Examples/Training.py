import torch 
import time
from pathlib import Path
import yaml 

from TrackGANN.src.model import TracksNN
from TrackGANN.src.dataset import SPDdataset, TrackMLdataset
from TrackGANN.src.config import ImportConfig
TRAINCONFIG, configArgs = ImportConfig()

from TrackGANN.utils.train_functions import train, evaluate, balanced_focal_loss
from TrackGANN.utils.saver import save_model
from TrackGANN.utils.loader import model_loader
from TrackGANN.utils.visualization import plot_metrics

'Config parameters'
'_____________________________________________________________________'
# Directory name
#=======================================================================#
name_of_directory = TRAINCONFIG['name_of_directory']
#=======================================================================#

# Detector parameters
#=======================================================================#
r_min = TRAINCONFIG['detector']['r_min']
r_max = TRAINCONFIG['detector']['r_max']
n_stations = TRAINCONFIG['detector']['n_stations']
det_eff = TRAINCONFIG['detector']['detector_eff']
#=======================================================================#


# Parameters for training dataset
#=======================================================================#
train_data_name = TRAINCONFIG['Train_dataset']['dataset_name']
train_directory = TRAINCONFIG['Train_dataset']['dataset_directory']
train_time_resolution = TRAINCONFIG['Train_dataset']['time_resolution']
train_mean_event = TRAINCONFIG['Train_dataset']['mean_events']
train_max_tracks = TRAINCONFIG['Train_dataset']['max_tracks']
train_samples = TRAINCONFIG['Train_dataset']['n_samples']
trainMLfolder = TRAINCONFIG['Train_dataset']['MLtrackPath']
#=======================================================================#


# Parameters for testing datasets
#=======================================================================#
test_data_name = TRAINCONFIG['Test_dataset']['dataset_name']
test_directory = TRAINCONFIG['Test_dataset']['dataset_directory']
test_time_resolution = TRAINCONFIG['Test_dataset']['time_resolution']
test_mean_event = TRAINCONFIG['Test_dataset']['mean_events']
test_max_tracks = TRAINCONFIG['Test_dataset']['max_tracks']
test_samples = TRAINCONFIG['Test_dataset']['n_samples']
test_log_label = TRAINCONFIG['Test_dataset']['log_label']
testMLfolder = TRAINCONFIG['Train_dataset']['MLtrackPath']
#=======================================================================#


# For other tests datasets 
#=======================================================================#
test_data_name1 = TRAINCONFIG['Test_dataset1']['dataset_name']
test_directory1 = TRAINCONFIG['Test_dataset1']['dataset_directory']
test_time_resolution1 = TRAINCONFIG['Test_dataset1']['time_resolution']
test_mean_event1 = TRAINCONFIG['Test_dataset1']['mean_events']
test_max_tracks1 = TRAINCONFIG['Test_dataset1']['max_tracks']
test_samples1 = TRAINCONFIG['Test_dataset1']['n_samples']
test_log_label1 = TRAINCONFIG['Test_dataset1']['log_label']
testMLfolder1 = TRAINCONFIG['Train_dataset']['MLtrackPath']
#=======================================================================#


# Parameters for learning process
#=======================================================================#
pos_weight = TRAINCONFIG['training']['pos_weight']
neg_weight = TRAINCONFIG['training']['neg_weight']
gamma = TRAINCONFIG['training']['gamma']
n_epochs = TRAINCONFIG['training']['n_epochs']
#=======================================================================#


# save/load model settings
#=======================================================================#
save_model_name = TRAINCONFIG['saver']['model_name']
save_step = TRAINCONFIG['saver']['save_step']
load_model = TRAINCONFIG['loader']['load_model']
model_dirrectory = TRAINCONFIG['loader']['model_dirrectory']
load_model_name = TRAINCONFIG['loader']['model_name']
#=======================================================================#


# Loss Function 
#=======================================================================#
pos_weight = TRAINCONFIG['training']['pos_weight']
neg_weight = TRAINCONFIG['training']['neg_weight']
gamma = TRAINCONFIG['training']['gamma']
#=======================================================================#


# Model Parameters 
#=======================================================================#
hidden_linear_layers = TRAINCONFIG['model']['encoder']['hidden_linear_layers']
out_linear_layer = TRAINCONFIG['model']['encoder']['out_linear_layer']
encoder_convolutions = TRAINCONFIG['model']['encoder']['n_convolutions']

edge_weight_layers = TRAINCONFIG['model']['classifier']['edge_weight_layers']
edge_update_layers = TRAINCONFIG['model']['classifier']['edge_update_layers']
node_update_layers = TRAINCONFIG['model']['classifier']['node_update_layers']
classifier_convolutions = TRAINCONFIG['model']['classifier']['n_convolutions']
#=======================================================================#

'_____________________________________________________________________'





# TrainDATASET GENERATION 
#=======================================================================#
if train_data_name == 'SPD':
    train_dataset = SPDdataset(name_of_directory)
    train_dataset.SetDetectorParm(n_stations,
                                  det_eff,
                                  r_min,
                                  r_max)
if train_data_name == 'TrackML':
    train_dataset = TrackMLdataset(name_of_directory)
    train_dataset.SetFolders(trainMLfolder)

train_dataset.train_data_generation(train_samples, train_mean_event, 
                                    train_max_tracks, train_time_resolution, train_directory)
train_data = train_dataset.convert_to_graph(train_directory)
#=======================================================================#



# TestDATASET GENERATION
#=======================================================================#
if test_data_name == 'SPD':
    test_dataset = SPDdataset(name_of_directory, remove=False)
    test_dataset.SetDetectorParm(n_stations,
                                  det_eff,
                                  r_min,
                                  r_max)
if test_data_name == 'TrackML':
    test_dataset = TrackMLdataset(name_of_directory, remove=False)
    test_dataset.SetFolders(testMLfolder)

test_dataset.test_data_generation(test_samples, test_mean_event, 
                                  test_max_tracks, test_time_resolution, test_directory)
test_data = test_dataset.convert_to_graph(test_directory)

log_header = "Train_Loss,Test_loss,Acceracy,Preciion,Recall"
log_filename = Path(name_of_directory) / f'{test_log_label}.csv'
path_to_csv = Path(name_of_directory) / f'{test_log_label}'
#=======================================================================#
    

# For other tests datasets 
#=======================================================================#
if test_data_name1 == 'SPD':
    test_dataset1 = SPDdataset(name_of_directory, remove=False)
    test_dataset1.SetDetectorParm(n_stations,
                                  det_eff,
                                  r_min,
                                  r_max)
if test_data_name1 == 'TrackML':
    test_dataset1 = TrackMLdataset(name_of_directory, remove=False)
    test_dataset1.SetFolders(testMLfolder1)

test_dataset1.test_data_generation(test_samples1, test_mean_event1, 
                                   test_max_tracks1, test_time_resolution1, test_directory1)
test_data1 = test_dataset1.convert_to_graph(test_directory1)


log_header1 = "Test_loss,Acceracy,Preciion,Recall"
log_filename1 = Path(name_of_directory) / f'{test_log_label1}.csv'
path_to_csv1 = Path(name_of_directory) / f'{test_log_label1}'
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
start_epoch = 0

device = torch.device(f'cuda')
model = TracksNN(hidden_linear_layers, out_linear_layer, encoder_convolutions,
                 edge_weight_layers, edge_update_layers, node_update_layers, classifier_convolutions).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, amsgrad=True, weight_decay=1e-3)
if load_model == True:       
    model, optimizer, new_epoch = model_loader(model, optimizer, model_dirrectory, load_model_name)
    start_epoch += new_epoch
    n_epochs += new_epoch
#=======================================================================#


# TRAINING LOOP
#=======================================================================#
start_time = time.time()
criterion = lambda pred, label: balanced_focal_loss(pred, label, pos_weight, neg_weight, gamma)

for epoch in range(start_epoch, n_epochs):
        train_loss = train(model=model, loader=train_data, optimizer=optimizer, criterion=criterion, device=device, 
      log_filename=log_filename, log_header=log_header)
        
        # Evaluation on testDataset
        test_loss, accuracy, purity, efficiency = evaluate(model=model, loader=test_data, criterion=criterion, device=device, threshold=0.5, 
      log_filename=log_filename, log_header=log_header)

        status = f'Epoch {epoch+1}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss:.5f}, Accuracy: {accuracy:.4f}, Purity: {purity:.4f}, Efficiency: {efficiency:.4f}'
        print(status)

        # Evaluation on testDataset1
        test_loss1, accuracy1, purity1, efficiency1 = evaluate(model=model, loader=test_data1, criterion=criterion, device=device, threshold=0.5, 
      log_filename=log_filename1, log_header=log_header1)
        status = f'Epoch {epoch+1}, Train Loss: {train_loss:.5f}, Test Loss: {test_loss1:.5f}, Accuracy: {accuracy1:.4f}, Purity: {purity1:.4f}, Efficiency: {efficiency1:.4f}'
        print(status)

        print(f"Spent time: {time.time() - start_time:.3f} s")
        if epoch != 0 and epoch % save_step == 0:
            save_model(model, optimizer, epoch, name_of_directory, save_model_name)
#=======================================================================#




# vizualisation 
#=======================================================================#
plot_metrics(path_to_csv)
plot_metrics(path_to_csv1)
#=======================================================================#