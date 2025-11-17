''' Plots an interactive 3d brain. Functions include highlighting 
cortical and subcortical structures and adding electrode contacts
with certain weights. 

Written by:
    Maarten C Ottenhoff
    Dec 2022
    Maastricht University

'''
from __future__ import annotations

import warnings
from itertools import product
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from matplotlib import cm
from mayavi import mlab
from mayavi.core.scene import Scene
from mayavi.modules.surface import Surface
from mayavi.modules.vectors import Vectors
import numpy as np
import scipy.io as sio

MATLAB_INDEXING = 1

FIRST, LAST = 0, -1
X, Y, Z = 0, 1, 2

@dataclass
class Model:
    name: str
    ver: np.array
    tri: np.array
    color: tuple
    opacity: float

class Brain:

    def __init__(self, 
                 id_:str, 
                 file_left: Path=None, 
                 file_right: Path=None,
                 opacity: float=0.1) -> None:
        
        if not file_left and not file_right:
            raise ValueError('Please supply at least one hemisphere')
            
        self.id = id_  # TODO: necessary?
        self.lh = self._load(file_left, 'left', opacity=opacity)   if file_left else None
        self.rh = self._load(file_right, 'right', opacity=opacity) if file_right else None

        self.rois = []

        self.full = self._stack_meshes()

        self.mesh = None
        self.mesh_rois = []

    def __repr__(self):
        return f"Brain(id={self.id}, hemispheres=[{'left' if self.lh else ''}, {'right' if self.rh else ''}])"

    def _load(self,
             file: Path,
             side: str, 
             color=(.8, .8, .8),
             opacity=0.1) -> Model:                                                             # Change OPACITY here of brain, 0.1 for screenshot

        if not file.exists():
            raise OSError(f'{file} does not exists.')

        hs = sio.loadmat(file)
        ver = hs['cortex']['vert'][0][0]
        tri = hs['cortex']['tri'][0][0] - MATLAB_INDEXING

        return Model(side, ver, tri, color, opacity)

    def _stack_meshes(self) -> Model:

        if not self.lh:
            return Model('full', self.rh.ver, self.rh.tri, self.rh.color, self.rh.opacity)
        
        if not self.rh:
            return Model('full', self.lh.ver, self.lh.tri, self.lh.color, self.lh.opacity)
        
        ver = np.vstack([self.lh.ver, self.rh.ver])
        tri = np.vstack([self.lh.tri, self._reindex_mesh(self.rh.tri)])

        # Note that if both hemispheres are plotted, the color and opacity of the left
        # hemisphere is chosen (in this line)
        return Model('full', ver, tri, self.lh.color, self.lh.opacity)

    def _reindex_mesh(self, 
                     tri: np.ndarray) -> np.ndarray:

        return tri + self.lh.ver.shape[0]

    def add_cortical_roi(self, 
                regions: list,
                path_regions: Path,
                hemisphere: str,
                color: tuple=(.3, .6, .4),
                opacity: float=1.0) -> None:


        # Returns mesh model from a list of regions
        if hemisphere not in ['lh', 'rh']:
            print('Invalid hemisphere, please choose from [lh, rh].')
            return

        # Step 1: Get the Index/number of all vertices in an area
        #         i.e, find out which coordinates to use for triangles
        vert_nums = np.array([], dtype=np.int32)
        for region in regions:
            
            with open(path_regions/f'{hemisphere}.{region}.label', 'r') as f:
                lines = [l.split()[0] for l in f.readlines()]

            vert_nums = np.append(vert_nums, list(map(int, lines[2:])))

        # Step 2: Get the corresponding vertices from the brain model
        hs = getattr(self, hemisphere)

        verts = hs.ver[vert_nums, :]

        # Step 3: Find and select all triangles of which ALL vertices 
        #         are present in the corresponding hemisphere mesh.
        tri_row_idc =   set(np.where(np.in1d(hs.tri[:,0], vert_nums))[0]) \
                      & set(np.where(np.in1d(hs.tri[:,1], vert_nums))[0]) \
                      & set(np.where(np.in1d(hs.tri[:,2], vert_nums))[0])
        
        tri = hs.tri[list(tri_row_idc), :]
        
        # Step 4: Map each vertex in triangles to the list of vertices
        for i, vert in enumerate(vert_nums):
            tri[tri==vert] = i

        roi = Model(hemisphere, verts, tri, color, opacity)
        self.rois += [roi]

    def add_subcortical_roi(self, 
                            name: str, 
                            path: Path,
                            color: tuple=(0.3, 0, 0.6),
                            opacity: float=1.0) -> None:
        
        mesh = sio.loadmat(path)
        
        verts = mesh['cortex']['vert'][0][0]
        tri =   mesh['cortex']['tri'][0][0] - MATLAB_INDEXING

        self.rois += [Model(name, verts, tri, color=color, opacity=opacity)]


class Contacts:

    def __init__(self, 
                 path: Path) -> None:

        self.xyz = None
        self.names = None
        self.name_map = None
        self.electrodes = None
        self.weights = None
        self.weight_map = None

        self.colors = None
        self.cmap = 'coolwarm'
        self.color_type = None

        self._load(path)

        self.mlab_points = None
        

    def __repr__(self):
        return f"Contacts(names={self.names}, labels={True if self.name_map else False}, weights={True if self.weight_map else False})"

    def _load(self, 
             path: Path) -> None:

        if not path.exists():
            raise OSError(f'{path} does not exists.')

        info = sio.loadmat(path)

        self.xyz = info['elecmatrix']
        self.names = np.hstack(info['anatomy'][:, 0])
        self.name_map = dict(zip(self.names, np.hstack(info['anatomy'][:, 3])))
        self.electrodes = set([c.rstrip('1234567890') for c in self.names])
        
    

    def add_weights(self, 
                    weight_map: dict) -> None:
        
        if self.color_type:
            warnings.warn('colors are also supplied. Weights will now have priority')

        self.weight_map = weight_map
        self.weights = np.zeros(self.names.size)
    
        for k, v in weight_map.items():
            
            idx = np.where(self.names==k)[0]
            self.weights[idx] = v
        
        self.color_type = 'weights'

    def add_color(self,
                  color: np.array):
        ''' color: np.array [1 x 3] | [n_contacts x 3]
                  Expects RGB values between 0 and 255 '''

        if not isinstance(color, np.ndarray):
            raise ValueError('Expects a np.array')

        if self.color_type == 'weights':
            warnings.warn('Weights are already provided, Colors will now have priority')

        if color.size == 3:
            self.colors = color / 255
            self.color_type = 'uniform'
            return

        if color.shape[0] != self.names.size:
            raise ValueError('Color array should be of same length as self.names')
                
        self.colors = np.hstack([color, np.full((color.shape[0], 1), 255)])        
        self.color_type = 'individual'
        
        
    def add_size(self, size_arr: np.array):
        ''' size_arr: [n_contacts x 1] Expects size values '''

        if not isinstance(size_arr, np.ndarray):
            raise ValueError('Expects a np.array')

        if size_arr.shape[0] != self.names.size:
            raise ValueError('Size array should be of same length as self.names')
                
        self.size = size_arr


    def set_colormap(self, 
                     cmap: str):
        ''' Available cmaps are most of matplotlibs cmaps.
            Supplying a wrong cmap will result is an error message with
            all available options. The list provided as mayavi docs is incomplete.
        '''
        self.cmap = cmap

    def interpolate_electrodes(self):

        for electrode in self.electrodes:
        
            idc = [i for i, name in enumerate(self.names) if electrode in name]

            n_contacts = len(idc)
            start, end = self.xyz[idc[FIRST]], self.xyz[idc[LAST]]

            self.xyz[idc, :] = np.linspace(start, end, n_contacts)

def _get_kwargs(contacts):
    if contacts.color_type == 'uniform':
        return {'color': tuple(contacts.colors)}

    if contacts.color_type == 'individual':
        return {'scalars': np.arange(contacts.colors.shape[0])}

    if contacts.color_type == 'weights':
        return {'scalars': contacts.weights,
                'colormap': 'coolwarm' if contacts.cmap == None else contacts.cmap}
    
    return {}

def _render_brain_mesh(scene: Scene, 
                       brain: Brain) -> Surface:

    mesh = mlab.triangular_mesh(
                    brain.full.ver[:, 0],
                    brain.full.ver[:, 1],
                    brain.full.ver[:, 2],
                    brain.full.tri,
                    figure = scene,
                    opacity = brain.full.opacity,
                    color = brain.full.color,
                    representation='surface',
                    scale_factor=1)

    mesh.actor.property.ambient = 0.4225
    mesh.actor.property.specular = 0.333
    mesh.actor.property.specular_power = 66
    mesh.actor.property.diffuse = 0.6995
    mesh.actor.property.interpolation = 'phong'

    if brain.full.opacity < 1.0:
        mesh.scene.renderer.trait_set(use_depth_peeling=True)

    # Smooth angles of all polygons in a scene
    for child in mlab.get_engine().scenes[0].children:
        poly_data_normals = child.children[0]
        poly_data_normals.filter.feature_angle = 80.0

    brain.mesh = mesh
    return mesh

def _render_contacts(scene: Scene, 
                     contacts: Contacts,
                     contactsize: float=1) -> Vectors:                                               # Change electrode SIZE here

    # TODO: set min/max value of colorscale?

    dirs = np.zeros(contacts.xyz.shape)
    kwargs = _get_kwargs(contacts)
    #kwargs.pop('scalars', None)
    
    # Get individual sizes if available, else use default
    if hasattr(contacts, 'size') and contacts.size is not None:
        size_scalars = contacts.size
        scale_mode = 'scalar'
        scale_factor = 1.0  # Acts as a multiplier
    else:
        size_scalars = np.ones(contacts.xyz.shape[0]) * contactsize
        scale_mode = 'none'
        scale_factor = contactsize
        
        
    '''contacts.mlab_points = mlab.quiver3d(
        contacts.xyz[:, X],
        contacts.xyz[:, Y],
        contacts.xyz[:, Z],
        dirs[:, X], dirs[:, Y], dirs[:, Z],
        figure=scene,
        mode='sphere',
        scale_mode=scale_mode,
        scale_factor=scale_factor,
        **kwargs)'''
        
    contacts.mlab_points = []
    for i in range(len(contacts.xyz)):
        color = tuple(contacts.colors[i][:3]) if contacts.color_type == 'individual' else contacts.colors
        point = mlab.points3d(
            contacts.xyz[i, X],
            contacts.xyz[i, Y],
            contacts.xyz[i, Z],
            scale_factor=contacts.size[i],
            color=color,
            mode='sphere',                                              # sphere, 2dcircle, no mode maybe 
            figure=scene
        )
        #point.actor.property.interpolation = 'flat'                     # shading: flat, phong (= 2dcircle)
        #point.actor.property.lighting = False
        contacts.mlab_points.append(point)
        

    

    if isinstance(contacts.mlab_points, list):  # Check if it's a list (multiple participants)
        for point in contacts.mlab_points:
            if isinstance(point, Vectors) and contacts.color_type in ['individual', 'weights']:
                point.glyph.color_mode = 'color_by_scalar'
            if contacts.color_type == 'individual':
                # Set look-up table and redraw
                point.module_manager.scalar_lut_manager.lut.table = contacts.colors
                mlab.draw()

    else:  # Single contact (non-list)
        if isinstance(contacts.mlab_points, Vectors) and contacts.color_type in ['individual', 'weights']:
            contacts.mlab_points.glyph.color_mode = 'color_by_scalar'

        if contacts.color_type == 'individual':
            # Set look-up table and redraw
            contacts.mlab_points.module_manager.scalar_lut_manager.lut.table = contacts.colors
            mlab.draw()


    # Uncomment to reverse color scale
    # if contacts.color_type == 'weights'
        # contacts.mlab_points.module_manager.scalar_lut_manager.reverse_lut = True
        
    return contacts.mlab_points

def _render_roi_mesh(scene: Scene, 
                     roi: Model) -> Surface:

    mesh = mlab.triangular_mesh(
                    roi.ver[:, 0],
                    roi.ver[:, 1],
                    roi.ver[:, 2],
                    roi.tri,
                    figure = scene,
                    opacity = roi.opacity,
                    color = roi.color,
                    representation='surface',
                    scale_factor=1)

    mesh.actor.property.ambient = 0.4225
    mesh.actor.property.specular = 0.333
    mesh.actor.property.specular_power = 66
    mesh.actor.property.diffuse = 0.6995
    mesh.actor.property.interpolation = 'phong'

    if roi.opacity < 1.0:
        mesh.scene.renderer.trait_set(use_depth_peeling=True)

    # Smooth angles of all polygons in a scene
    for child in mlab.get_engine().scenes[0].children:
        poly_data_normals = child.children[0]
        poly_data_normals.filter.feature_angle = 80.0

    return mesh

def plot(brain: Brain, 
         contacts: Contacts=None, 
         show: bool=True,
         scene: Scene=None,
         contsize: float=1) -> tuple[Scene, list]:

    if not scene:
        scene = mlab.figure(fgcolor = (0, 0, 0),
                            bgcolor = (1, 1, 1),
                            engine = None,
                            size = (2*1200, 2*900))

    _render_brain_mesh(scene, brain)

    brain.mesh_rois = [_render_roi_mesh(scene, roi) for roi in brain.rois]

    contacts = [contacts] if type(contacts) != list else contacts
    
    for contact in contacts:
        _render_contacts(scene, contact, contactsize=contsize)

    if show:
        mlab.show()

    return scene

def take_screenshots(scene: Scene,
                     outpath: Path,
                     azimuths: list = [0],         # [0, 45, 90, 135, 180, 225, 270, 315],
                     elevations: list = [0],       # 30
                     distances: list = [400],
                     topdown: bool = False,
                     tiltedtop: bool = False,
                     right: bool = False,
                     innerright: bool = False,
                     angeled: bool = False,
                     left: bool = False,
                     front: bool = False) -> None:

    if topdown:
        azimuths, elevations, distances = [0], [0], [400]               
    elif tiltedtop:
        azimuths, elevations, distances = [90], [10], [400]    
    elif right:
        azimuths, elevations, distances = [0], [90], [400]              
    elif angeled:
        azimuths, elevations, distances = [10], [90], [400]             # 70 for higher elevation            
    elif innerright:
        azimuths, elevations, distances = [80], [65], [400]               # 180, 90
    elif left:
        azimuths, elevations, distances = [180], [90], [400]
    elif front:
        azimuths, elevations, distances = [90], [90], [400]        
    else: 
        azimuths, elevations, distances = [0], [0], [400]

    
    now = datetime.now().strftime('%Y%m%d_%H%M%S')
    outpath = Path(outpath)
    outpath.mkdir(exist_ok=True, parents=True)

    for a, e, d in product(azimuths, elevations, distances):
        mlab.view(azimuth=a, elevation=e, distance=d, figure=mlab.gcf())
        mlab.savefig(str(outpath/f'view_a{a}_e{e}_d{d}.png'), figure=scene, magnification=6)