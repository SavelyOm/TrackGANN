from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings


warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('mode.chained_assignment', None)

class TrackMLsampler():

    def __init__(self):
        self.non=2

    def event_generator(self, data_path,
                        mean_tracks):
        
        self.data_path = data_path
        self.mean_tracks = mean_tracks

        flag = True

        while flag:
            random_normal = np.random.randint(2,mean_tracks)
            hits = pd.read_csv(self.data_path + '-hits.csv')
            truth = pd.read_csv(self.data_path + '-truth.csv')
            merged = pd.merge(hits, truth, on='hit_id', how='left')

            merged = merged[merged['particle_id'] != 0]
            merged['tr_id'] = merged['particle_id']
            track_notes = merged['tr_id'].unique()

            selected_tracks = np.random.choice(track_notes, size=int(random_normal), replace=False)
            unsorted_event = merged[merged['tr_id'].isin(selected_tracks)]

            unsorted_event['R'] = np.sqrt(unsorted_event['x']**2 + unsorted_event['y']**2 + unsorted_event['z']**2)
             
            event_sorted = unsorted_event.sort_values(['particle_id','R'])
            self.event = event_sorted[['x', 'y', 'z', 'tr_id']]

            x_coords_dict = {tid: self.event[self.event['tr_id'] == tid]['x'].values for tid in selected_tracks}
            num_hits = [len(x_coords_dict[selected_tracks[i]]) for i in range(len(selected_tracks))] 
            if num_hits:
                min_hits = min(num_hits)
            else:
                min_hits = 1
            
            if  min_hits < 3:
                continue
            else:
                flag = False

        return self.event

    def timeslece_generator(self,
                            mean_event: int,
                            mean_tracks: int,
                            data_folder: str,
                            ):
        
        self.data_folder = data_folder
        self.mean_event = mean_event
        self.mean_tracks = mean_tracks


        folder = Path(self.data_folder)
        first_file = next((f.name for f in folder.iterdir() if f.is_file()), None)
        parts = first_file.split('-')
        number_part = int(parts[0][5:])
        

        num_event = np.random.poisson(self.mean_event,size=1)
        self.timeslice = pd.DataFrame(columns=['x', 'y', 'z', 'track_id', 'event_id'])

        flag = True
        while flag:
            for n in range(int(num_event)):
                event = self.event_generator(data_path=f'{self.data_folder}/event00000{number_part+n}', mean_tracks=self.mean_tracks)
                event = event.assign(ev_id=[int(n)] * len(event['x']))
                vz_range = np.random.uniform(-350,350)
                vy_range = np.random.uniform(-20,20)
                vx_range = np.random.uniform(-20,20)
                event['z']=event['z'].apply(lambda x: x+vz_range)
                event['y']=event['y'].apply(lambda x: x+vy_range)
                event['x']=event['x'].apply(lambda x: x+vx_range)
                #print(pd.DataFrame(event.values, columns=timeslice.columns))
                timeslice_data = pd.DataFrame(event.values, columns=self.timeslice.columns).dropna(how='all', axis=1)
                self.timeslice = pd.concat([self.timeslice, timeslice_data], ignore_index=True)
            
            unique_particles = self.timeslice['track_id'].unique()
            particle_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_particles)}
            self.timeslice['track_id'] = self.timeslice['track_id'].map(particle_id_map)
            self.timeslice['event_id'] = self.timeslice['event_id'].astype(int)
            if len(self.timeslice['track_id'].unique()) != max(self.timeslice['track_id'])+1:
                continue
            else: 
                flag = False

        return self.timeslice
        
    def drift_time_generation(self, time_resolution):
        ''' For timeslices with format DataFrame(['x', 'y', 'z', 'track_id', 'event_id']) 
        after timeslece_generator() '''

        self.time_resolution = time_resolution

        unique_events = sorted(self.timeslice['event_id'].unique())
        num_events = len(unique_events)

        
        event_time_centers = {event: event_id / (num_events - 1) for event_id, event in enumerate(unique_events)}
        if (self.time_resolution+1) < max(unique_events):
            DeltaT = lambda s: abs((event_time_centers[s] - event_time_centers[0])/2)  
        
        else: 
            DeltaT = lambda s: abs((event_time_centers[max(unique_events)] - event_time_centers[0])/2)

        
        track_time_data = {}

        
        for track_id, group in self.timeslice.groupby('track_id'):
            event_id = group['event_id'].iloc[0]  
            t_center = event_time_centers[event_id]
            
            
            t_noise = DeltaT(self.time_resolution+1)-DeltaT(self.time_resolution+1)/1000
                
            t_left = max(0, t_center-t_noise)
            t_right = min(t_center+t_noise,1)
            
            track_time_data[track_id] = {
                'time_left': t_left,
                'time_right': t_right
            }
            
        self.timeslice['time_left'] = self.timeslice['track_id'].map(lambda x: track_time_data[x]['time_left'])
        self.timeslice['time_right'] = self.timeslice['track_id'].map(lambda x: track_time_data[x]['time_right'])

        return self.timeslice
    
    def timeslice_shuffle(self):
        ''' For timeslices with format DataFrame(['x', 'y', 'z', 'track_id', 'event_id']) 
        after timeslece_generator() '''

        unique_tracks = self.timeslice['track_id'].unique()
        np.random.shuffle(unique_tracks)

       
        self.timeslice = pd.concat(
            [self.timeslice[self.timeslice['track_id'] == track_id] for track_id in unique_tracks],
            ignore_index=True
        )
        track_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_tracks)}

    
        self.timeslice['track_id'] = self.timeslice['track_id'].map(track_mapping)
         
        return self.timeslice
    
    def create_csv(self, file_path, file_note):
        timeslices_dir = file_path / "Timeslices"
        if timeslices_dir.exists():
            pass
        else:
            (file_path).mkdir()
            (file_path / "Timeslices").mkdir()

        timeslice_path = timeslices_dir / f"timeslice_{file_note}.csv"
        self.timeslice.to_csv(timeslice_path, sep=' ', index=False)
        return self.timeslice

    def plot_timeslice(self):

        timeslice_df=self.timeslice

        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        
        unique_events = timeslice_df['event_id'].unique()
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_events)))
        
        event_colors = {event: colors[i] for i, event in enumerate(unique_events)}
        grouped = timeslice_df.groupby(['event_id', 'track_id'])
        
        for (event_id, track_id), group in grouped:
            ax.plot(
                group['x'], 
                group['y'], 
                group['z'],
                color=event_colors[event_id],
                marker='o',
                markersize=3,
                linewidth=1,
                alpha=0.7,
                label=f'Event {event_id}' if track_id == 0 else ""  
            )
        
        
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_zlabel('Z coordinate')
        ax.set_title('Timeslice Visualization with Event Coloring')
        
       
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title='Events', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()

