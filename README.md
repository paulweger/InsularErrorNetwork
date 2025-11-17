# Insular Error Network – Reproducible Code

This repository contains the code used to reproduce all analyses and figures from the paper:  
“Insular error network enables self-correcting intracranial brain–computer interface”


## Data
Download the dataset here:  
https://osf.io/wqsgr

Place all downloaded files into the `Data/` folder.




## Installation
Install all dependencies using the requirements file:
```bash
pip install -r requirements.txt
```
Depending on your system, Mayavi, PyQt, and VTK may require additional system packages.




## Repository Structure
```text
project/
├── Data/                         # place downloaded data here
├── Code/                         # main analysis scripts
│   ├── Calibration_analysis.py   # Analysis of calibration data / Movement results
│   ├── CL_analysis.py            # Analysis of closed-loop data / Error results
│   ├── Preprocessing/            # preprocessing scripts (already run; outputs in data)
│   └── Functions/                # helper functions for plotting, calculations, utils
├── Plotting/                     # output figures and brain-plotting code
└── requirements.txt              # dependency list
```




## Usage
Run the main analysis scripts:
python Code/Calibration_analysis.py
python Code/CL_analysis.py
All figures will be saved automatically in the Plotting/ folder.




## Notes
• Preprocessing outputs are included for faster reproduction

• Patient data are not included in the repository; download them from the link above

• Brain visualization requires Mayavi (included in requirements)

• For questions, please contact paul.weger@maastrichtuniversity.nl

• Written and uploaded on the 17th of November 2025
