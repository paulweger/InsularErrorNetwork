''' 
###### Plot a single participant to observe electrode coverage ######
''' 
from pathlib import Path
import numpy as np
import brainplots


def example():
    #############
    ppt_id = 10                  # e.g. 1, 2, 3, ..., 10
    #############
    
    # Define paths & initialize mni surface
    path_data = Path(__file__).resolve().parents[2] / 'data'
    file_left_hemisphere  = path_data/'MNI_surface'/'lh_pial.mat'
    file_right_hemisphere = path_data/'MNI_surface'/'rh_pial.mat'
    path_cortical_regions = path_data/'MNI_surface'/'label'/'gyri'
    path_subcortical_regions = path_data/'MNI_surface'/'subcortical'
    brain = brainplots.Brain(id_={ppt_id}, 
                             file_left=  file_left_hemisphere,
                             file_right= file_right_hemisphere)

    # Load contacts
    path_to_coordinates = path_data/'Electrode_coordinates'/f'coords_P{ppt_id:02d}.mat'
    contacts = brainplots.Contacts(path_to_coordinates)       
    
    # Define channel colors and weights
    color_arr = np.zeros((contacts.names.size, 3), dtype=int)               # Define color
    size_arr = np.ones(contacts.names.size, dtype=int) * 2                  # Define size
    contacts.add_color(color_arr)
    contacts.add_size(size_arr)
    # contacts.interpolate_electrodes()         # If you want straight shafts, contacts will appear in slightly wrong position

    # Plot / save
    scene = brainplots.plot(brain, contacts, show=True)
    #brainplots.take_screenshots(scene, outpath=f'...', topdown = True)


if __name__=='__main__':
    example()