''' 
###### Plot multiple participants for a group-level electrode coverage ######
''' 
from pathlib import Path
import numpy as np
import brainplots
import csv
import re
from collections import defaultdict


def example():
    #################
    participants = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 
    save_plot = True           # False = view plot, True = save plot     
    onesided = True
    #################

    # Define paths & initialize mni surface
    path_data = Path(__file__).resolve().parents[2] / 'data'
    file_left_hemisphere  = path_data/'MNI_surface'/'lh_pial.mat'
    file_right_hemisphere = path_data/'MNI_surface'/'rh_pial.mat'
    path_cortical_regions = path_data/'MNI_surface'/'label'/'gyri'
    path_subcortical_regions = path_data/'MNI_surface'/'subcortical'
    if onesided: brain = brainplots.Brain(id_='MNI', file_right=file_right_hemisphere)
    else: brain = brainplots.Brain(id_= 'MNI', file_left=file_left_hemisphere, file_right=file_right_hemisphere)
    
    # Define colors
    color_green = np.array((44, 160, 44))
    color_blue = np.array((31, 119, 180))
    color_turkoise = np.array([81, 224, 224])        
    
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
                
        # Set color and size 
        color_arr = np.array([[0, 0, 0]] * len(contacts.names), dtype=float)        # Color electrodes black
        size_arr = np.ones(contacts.names.size, dtype=int) * 1.5                      # Choose electrode size

        # Attach plotting specifications
        contacts.add_color(color_arr)
        contacts.add_size(size_arr)
        # contacts.interpolate_electrodes()         # If you want straight shafts, contacts will appear in slightly wrong position
        contacts_list.append(contacts)
        
    # Plot all contacts in MNI
    if save_plot:
        scene = brainplots.plot(brain, contacts=contacts_list, show=False)
        save_path = Path(__file__).resolve().parents[1] / 'Fig1' 
        brainplots.take_screenshots(scene, outpath=save_path, topdown=True)
    else:
        scene = brainplots.plot(brain, contacts=contacts_list, show=True)
            
if __name__=='__main__':
    example()
