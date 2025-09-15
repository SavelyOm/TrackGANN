from pathlib import Path
import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
from abc import ABC, abstractmethod

from .TrackML_sampler import TrackMLsampler
from .data_generation import SPDEventGenerator

from TrackGANN.utils.saver import save_graph_to_npz
from TrackGANN.utils.loader import load_npz_to_pyg
from TrackGANN.utils.convertors import csv_to_graph


def remove_folder(path, ask_confirmation=True):
    ''' Deletes a folder after confirming that the user really wants to delete it. '''

    path = Path(path)
    if path.exists() and path.is_dir():
        if ask_confirmation:
            answer = input(f"Delete the folder located at the '{path}'? [y/n]: ")
            if answer.lower() != 'y':
                return False
            else:
                shutil.rmtree(path)


class TimeslicesDataset(ABC):
    ''' Interface for a dataset containing a variable number of timeslices. 
    This interface defines the methods for generating timeslices as well 
    as for converting them into a graph data type. '''

    
    def __init__(self, dir_name, remove):
        self.remove = remove
        self.dir_name = Path(dir_name)
        if self.remove == True:
            remove_folder(dir_name) 
            self.dir_name.mkdir(parents=True, exist_ok=True)
            (self.dir_name / "models").mkdir()
    
    @abstractmethod
    def train_data_generation(self,
                            n_samples,
                            mean_event,
                            max_tracks,
                            time_resolution,
                            timeslice_dir
                            ):
        pass

    @abstractmethod
    def test_data_generation(self,
                            n_samples,
                            mean_event,
                            max_tracks,
                            time_resolution,
                            timeslice_dir
                            ):
        pass
        
    def convert_to_graph(self):
        raise NotImplementedError()
 




class SPDdataset(TimeslicesDataset):
    ''' This class defines methods for creating a dataset in the configuration of a loose SPD simulation. 
    The class itself allows working with already generated data, the generation algorithm for which is 
    described in data_generation.py located in the src folder. Within this class, methods test_data_generation() and 
    train_data_generation() are defined. A method for generating drift time, drift_time_generation(), has been added. 
    There is also a method for converting data into a graph data type convert_to_graph(), 
    and a method for saving timeslices to a file. '''


    def __init__(self, dir_name, remove=True):
        super().__init__(dir_name, remove)
        self.dir_name = Path(dir_name)
    

    def generation(self, n_samples, mean_event, max_tracks):
        self.n_samples = n_samples
        self.mean_event = mean_event
        self.max_tracks = max_tracks
        
        spdgen = SPDEventGenerator(max_event_tracks = self.max_tracks,
                                   mean_events_timeslice = self.mean_event,
                                   n_stations = self.n_stations,
                                   detector_eff = self.det_eff,
                                   r_coord_range=(self.r_min, self.r_max))
        
        timeslice = spdgen.generate_time_slice()
        self.df = pd.DataFrame({
            'x': timeslice.hits[:, 0],
            'y': timeslice.hits[:, 1],
            'z': timeslice.hits[:, 2],
            'track_id': timeslice.track_ids,
            'event_id': timeslice.event_ids
        })

        return self.df    

    def SetDetectorParm(self, n_stations, det_eff, r_min, r_max):
        self.n_stations = n_stations
        self.det_eff = det_eff
        self.r_min = r_min
        self.r_max = r_max


    def drift_time_generation(self, time_resolution):

        unique_events = sorted(self.df['event_id'].unique())
        num_events = len(unique_events)
        event_time_centers = {event: event_id / (num_events - 1) for event_id, event in enumerate(unique_events)}
        if (time_resolution+1) < max(unique_events):
            DeltaT = lambda s: abs((event_time_centers[s] - event_time_centers[0])/2)  
        else: 
            DeltaT = lambda s: abs((event_time_centers[max(unique_events)] - event_time_centers[0])/2)

        track_time_data = {}
        for track_id, group in self.df.groupby('track_id'):
            event_id = group['event_id'].iloc[0]  
            t_center = event_time_centers[event_id]
            
            t_noise = DeltaT(time_resolution+1)-DeltaT(time_resolution+1)/1000
            t_left = max(0, t_center-t_noise)
            t_right = min(t_center+t_noise,1)
            
            track_time_data[track_id] = {
                'time_left': t_left,
                'time_right': t_right
            }

        
        self.df['time_left'] = self.df['track_id'].map(lambda x: track_time_data[x]['time_left'])
        self.df['time_right'] = self.df['track_id'].map(lambda x: track_time_data[x]['time_right'])

        return self.df
    
    def timeslice_shuffle(self):

        unique_tracks = self.df['track_id'].unique()
        np.random.shuffle(unique_tracks) 
        self.df = pd.concat(
            [self.df[self.df['track_id'] == track_id] for track_id in unique_tracks],
            ignore_index=True
        )
        track_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_tracks)}
        self.df['track_id'] = self.df['track_id'].map(track_mapping)
    
    def make_csv(self, timeslice_directory, i):
        timeslices_dir = self.dir_name / timeslice_directory / "Timeslices"
        if timeslices_dir.exists():
            pass
        else:
            (self.dir_name / timeslice_directory).mkdir()
            (self.dir_name / timeslice_directory / "Timeslices").mkdir()

        file_path = timeslices_dir / f"timeslice_{i}.csv"
        self.df.to_csv(file_path, sep=' ', index=False)
        return self.df


    def convert_to_graph(self, data_directory): 
        

        self.graphs = []
        folder_path = Path(self.dir_name / data_directory / "Timeslices")
        n_files = len([f for f in folder_path.iterdir() if f.is_file()])

        for idx in tqdm(range(n_files), desc="Convertation", unit="nzp_files"):
            csv_path = self.dir_name / data_directory / "Timeslices" / f"timeslice_{idx}.csv"
            
            graph = csv_to_graph(csv_file=str(csv_path))
            if (self.dir_name / data_directory / "Nbfiles").exists():
                pass
            else:
                (self.dir_name / data_directory / "Nbfiles").mkdir()

            npz_path = self.dir_name / data_directory / "Nbfiles" / f"file_{idx}.npz"

            save_graph_to_npz(graph, str(npz_path))
            self.graphs.append(load_npz_to_pyg(str(npz_path)))

        return self.graphs

    def train_data_generation(self,
                              n_samples,
                              mean_event,
                              max_tracks,
                              time_resolution,
                              timeslice_dir
                              ):
        for i in tqdm(range(n_samples), desc="Train_Generation", unit="timeslice"):
            self.generation(n_samples, mean_event, max_tracks)
            self.drift_time_generation(time_resolution)
            self.timeslice_shuffle()
            self.make_csv(timeslice_dir, i)
        
    def test_data_generation(self,
                              n_samples,
                              mean_event,
                              max_tracks,
                              time_resolution,
                              timeslice_dir
                              ):
        for i in tqdm(range(n_samples), desc="Test_Generation", unit="timeslice"):
            self.generation(n_samples, mean_event, max_tracks)
            self.drift_time_generation(time_resolution)
            self.timeslice_shuffle()
            self.make_csv(timeslice_dir, i)

    



class TrackMLdataset(TimeslicesDataset):
    ''' This class is designed for creating a timeslice dataset with tracks from the open TrackML database. 
    The class defines methods test_data_generation() and train_data_generation(). 
    It also includes a method for converting data into a graph type, convert_to_graph(). 
    The class itself does not implement the algorithms for generating timeslices based on tracks from TrackML. 
    The timeslice generation methods are described in the TrackML_sampler.py file. '''
    
    def __init__(self, dir_name, remove=True):
        super().__init__(dir_name, remove)
        self.dir_name = Path(dir_name)

    def SetFolders(self, ML_data_folder):
        self.data_folder = ML_data_folder
        
    def train_data_generation(self,
                              n_samples,
                              mean_event,
                              max_tracks,
                              time_resolution,
                              timeslice_dir
                              ):
        for i in tqdm(range(n_samples), desc="Train_Generation", unit="timeslice"):
            timeslice = TrackMLsampler()
            timeslice.timeslece_generator(
                mean_event=mean_event,
                mean_tracks=max_tracks,
                data_folder= self.data_folder,
            )
            timeslice.drift_time_generation(time_resolution=time_resolution)
            timeslice.timeslice_shuffle()
            timeslice_path = Path(self.dir_name / timeslice_dir)
            timeslice.create_csv(file_path=timeslice_path, file_note=i)

    def test_data_generation(self,
                              n_samples,
                              mean_event,
                              max_tracks,
                              time_resolution,
                              timeslice_dir):
        for i in tqdm(range(n_samples), desc="Test_Generation", unit="timeslice"):
            timeslice = TrackMLsampler()
            timeslice.timeslece_generator(
                mean_event=mean_event,
                mean_tracks=max_tracks,
                data_folder=self.data_folder,
            )
            timeslice.drift_time_generation(time_resolution=time_resolution)
            timeslice.timeslice_shuffle()
            timeslice_path = Path(self.dir_name / timeslice_dir)
            timeslice.create_csv(file_path=timeslice_path, file_note=i)

    def convert_to_graph(self, data_directory): 
        
        self.graphs = []
        folder_path = Path(self.dir_name / data_directory / "Timeslices")
        n_files = len([f for f in folder_path.iterdir() if f.is_file()])

        for idx in tqdm(range(n_files), desc="Convertation", unit="nzp_files"):
            csv_path = self.dir_name / data_directory / "Timeslices" / f"timeslice_{idx}.csv"
            
            graph = csv_to_graph(csv_file=str(csv_path))
            if (self.dir_name / data_directory / "Nbfiles").exists():
                pass
            else:
                (self.dir_name / data_directory / "Nbfiles").mkdir()

            npz_path = self.dir_name / data_directory / "Nbfiles" / f"file_{idx}.npz"

            save_graph_to_npz(graph, str(npz_path))
            self.graphs.append(load_npz_to_pyg(str(npz_path)))

        return self.graphs

    
    
