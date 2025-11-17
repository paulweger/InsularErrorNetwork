''' 
Plot zoom-in of insular electrodes (all, motor, error)
Script written by Paul Weger (paul.weger@maastrichtuniversity.nl)
11th of November 2025
''' 
# Import libaries
from pathlib import Path
import numpy as np
import brainplots
import csv
import pickle
# Local function import
from functions import *

def example():
    #################
    participants = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
    save_plot = True               # False = view plot, True = save plot   
    plot_all_elecs = False          # True = view all elecs in ROI, False = highlight electrodes specified below
    highlight_error = False         # True = highlight error electrodes, False = highlight motor electrodes
    #################

    # Define paths & initialize mni surface
    path_data = Path(__file__).resolve().parents[2] / 'Data'
    file_left_hemisphere  = path_data/'MNI_surface'/'lh_pial.mat'
    file_right_hemisphere = path_data/'MNI_surface'/'rh_pial.mat'
    path_cortical_regions = path_data/'MNI_surface'/'label'/'gyri'
    path_subcortical_regions = path_data/'MNI_surface'/'subcortical'
    path_error_tvals = Path(__file__).resolve().parents[1] / 'Plotting_values' / 'errorChannels.pkl'
    path_motor_tvals = Path(__file__).resolve().parents[1] / 'Plotting_values' / 'motorHFAchannels.pkl'
    if plot_all_elecs: brain = brainplots.Brain(id_='MNI', file_right=file_right_hemisphere, opacity=0.1)    
    else: brain = brainplots.Brain(id_='MNI', file_right=file_right_hemisphere, opacity=0.0)    
    
    # Load error electrodes to highlight
    if highlight_error: path_highlight = path_error_tvals
    else: path_highlight = path_motor_tvals
    with open(path_highlight, 'rb') as f:
        highlightChannels = pickle.load(f)
    highlightChannels, minimum_tval, maximum_tval = filter_to_roi(highlightChannels, rois=[1, 4])   # Filter to ROIs of interest
    print(f'Min abs t-value: {minimum_tval:.2f}, Max abs t-value: {maximum_tval:.2f}')
    minimum_tval, maximum_tval = 2, 16          # Hardcode to equalize motor & error scaling        
    
    # Define colors
    color_green = np.array((44, 160, 44))
    color_blue = np.array((31, 119, 180))
    color_turkoise = np.array([81, 224, 224])    
    grey_elec = np.array([0.65, 0.65, 0.65]) * 255    
    
    # Highlight insula
    brain.add_cortical_roi(['insula'], path_cortical_regions, 'rh', color=tuple(grey_elec/255), opacity=1) 
    
    # Loop over participants
    contacts_list = [] 
    for ppt_id in participants:
        
        # Load electrode coordinates and locations
        path_to_coordinates = path_data/'Electrode_coordinates'/f'coords_P{ppt_id:02d}.mat'
        contacts = brainplots.Contacts(path_to_coordinates)     
        contacts.xyz[:, 0] = np.abs(contacts.xyz[:, 0]) + 12            # onesided + lift so elecs are not covered by insula surface
        path_to_locations = path_data/'Electrode_locations'/f'electrode_locations_P{ppt_id:02d}.csv'
        with open(path_to_locations, newline='') as f:
            reader = csv.DictReader(f)
            electrode_names, locations, roi = zip(*[(r['electrode_name_1'], r['location'], r['ROI']) for r in reader]) 
            roi_map = {name: int(r) for name, r in zip(electrode_names, roi)}
            
        # Get channels to highlight and map tvalue to size
        highlight_electrodes = highlightChannels[ppt_id]['channels']
        highlight_tvals = highlightChannels[ppt_id]['tvalues']
        highlight_sizes = tvals_to_sizes(highlight_tvals, minimum_tval, maximum_tval)         
                
        # ---- Set color and size of electordes ----
        size_arr = np.zeros(len(contacts.names), float)
        color_arr = np.zeros((len(contacts.names), 3), float)
        highlight_map = dict(zip(highlight_electrodes, highlight_sizes)) if not plot_all_elecs else {}
        
        # Loop through electrodes
        for i, name in enumerate(contacts.names):
            r = roi_map.get(name, 0)
            # size logic
            if not plot_all_elecs: size_arr[i] = highlight_map.get(name, 0)   # Size depending on tvalue
            else: size_arr[i] = 3 if r in (1, 4) else 0                       # Standard size 3
            # color logic
            color_arr[i] = (
                color_blue/255     if r == 1 else
                color_turkoise/255 if r == 4 else
                [0, 0, 0]
            )

        # Attach plotting specifications
        contacts.add_color(color_arr)
        contacts.add_size(size_arr)
        # contacts.interpolate_electrodes()         # If you want straight shafts, contacts will appear in slightly wrong position
        contacts_list.append(contacts)
        
    # Plot all contacts in MNI
    if save_plot:
        scene = brainplots.plot(brain, contacts=contacts_list, show=False)
        save_path = Path(__file__).resolve().parents[1] / 'Fig4' 
        brainplots.take_screenshots(scene, outpath=save_path, angeled=True)
    else:
        scene = brainplots.plot(brain, contacts=contacts_list, show=True)
            
if __name__=='__main__':
    example()
