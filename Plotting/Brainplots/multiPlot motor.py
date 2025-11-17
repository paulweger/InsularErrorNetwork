''' 
###### Plot motor relevant electrodes across participants ######
''' 
from pathlib import Path
import numpy as np
import brainplots
import csv
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
# Local import
from functions import *


def example():
    #################
    participants = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
    save_plot = True           # False = view plot, True = save plot     
    onesided = True
    HFA = False
    #################

    # Define paths & initialize mni surface
    path_data = Path(__file__).resolve().parents[2] / 'data'
    file_left_hemisphere  = path_data/'MNI_surface'/'lh_pial.mat'
    file_right_hemisphere = path_data/'MNI_surface'/'rh_pial.mat'
    path_cortical_regions = path_data/'MNI_surface'/'label'/'gyri'
    path_subcortical_regions = path_data/'MNI_surface'/'subcortical'
    path_hfa = Path(__file__).resolve().parents[1] / 'Plotting_values' / 'motorHFAchannels.pkl'
    path_lfa = Path(__file__).resolve().parents[1] / 'Plotting_values' / 'motorLFAchannels.pkl'
    if onesided: brain = brainplots.Brain(id_='MNI', file_right=file_right_hemisphere)
    else: brain = brainplots.Brain(id_= 'MNI', file_left=file_left_hemisphere, file_right=file_right_hemisphere)
    
    # Load error electrodes to highlight
    if HFA: path_highlight = path_hfa
    else: path_highlight = path_lfa
    with open(path_highlight, 'rb') as f:
        highlightChannels = pickle.load(f)
    minimum_tval, maximum_tval = get_min_max(highlightChannels)   # Get global min/max absolute t-values
    print(f'Min abs t-value: {minimum_tval:.2f}, Max abs t-value: {maximum_tval:.2f}')
    minimum_tval, maximum_tval = 2, 32          # Hardcode to equalize hfa & lfa scaling  
    
    # Define colors
    color_green = np.array((44, 160, 44))
    color_blue = np.array((31, 119, 180))
    color_turkoise = np.array([81, 224, 224])    
    hot_r = ListedColormap(plt.cm.hot_r(np.linspace(0, 1, 256))[:, :3])
    
    # Loop over participants
    contacts_list = [] 
    for ppt_id in participants:
        
        # Load electrode coordinates and locations
        path_to_coordinates = path_data/'Electrode_coordinates'/f'coords_P{ppt_id:02d}.mat'
        contacts = brainplots.Contacts(path_to_coordinates)       
        if onesided: contacts.xyz[:, 0] = np.abs(contacts.xyz[:, 0])  
        path_to_locations = path_data/'Electrode_locations'/f'electrode_locations_P{ppt_id:02d}.csv'
        with open(path_to_locations, newline='') as f:
            reader = csv.DictReader(f)
            electrode_names, locations, roi = zip(*[(r['electrode_name_1'], r['location'], r['ROI']) for r in reader]) 
            roi_map = {name: int(r) for name, r in zip(electrode_names, roi)}
             
        # Get tvalues per patient & map to size/color
        highlight_electrodes = highlightChannels[ppt_id]['channels']
        highlight_tvals = highlightChannels[ppt_id]['tvalues']
        highlight_sizes = tvals_to_sizes(highlight_tvals, minimum_tval, maximum_tval, min_size=2, max_size=4)
        highlight_colors = tvals_to_color(highlight_tvals, minimum_tval, maximum_tval)
        highlight_colors = hot_r(highlight_colors)[:, :3]
                
        # Set color and size 
        color_arr = np.zeros((len(contacts.names), 3), dtype=float)
        size_arr = np.zeros(len(contacts.names), dtype=int) 
        chan_to_size  = dict(zip(highlight_electrodes, highlight_sizes))
        chan_to_color = dict(zip(highlight_electrodes, highlight_colors))                
        for i, name in enumerate(contacts.names):
            size_arr[i]  = chan_to_size.get(name, 0)                # Default size 0 (invisible) if not highlighted
            color_arr[i] = chan_to_color.get(name, [0, 0, 0])
    
        # Attach plotting specifications
        contacts.add_color(color_arr)
        contacts.add_size(size_arr)
        # contacts.interpolate_electrodes()         # If you want straight shafts, contacts will appear in slightly wrong position
        contacts_list.append(contacts)
        
    # Plot all contacts in MNI
    if save_plot:
        scene = brainplots.plot(brain, contacts=contacts_list, show=False)
        save_path = Path(__file__).resolve().parents[1] / 'Fig2' 
        brainplots.take_screenshots(scene, outpath=save_path, topdown=True)
    else:
        scene = brainplots.plot(brain, contacts=contacts_list, show=True)
            
if __name__=='__main__':
    example()
