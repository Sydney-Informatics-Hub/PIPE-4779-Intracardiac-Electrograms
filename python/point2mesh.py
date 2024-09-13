"""
Interpolate point data onto mesh geometry

Functionality:
- read in carto mesh file
- read in point data file (parquet) with X, Y, Z, and data columns (prediction, probability)
- process point data (average probabilities of model ensemble)
- interpolate point data onto mesh using Gaussian kernel
- write out vtk file with mapped point data and convert to carto format

need pip install carto-reader
"""

import os
import pyvista as pv
import pandas as pd
from carto_reader.carto_data import Carto
from cartoreader_lite.low_level.read_mesh import read_mesh_file
import vtk

class MeshDataMapper:
    """
    Interpolates point data onto a mesh geometry using a Gaussian kernel.

    Args:
        path_data_export: Path to the data export directory containing the mesh file.
        mesh_file: Path and filename to the the mesh file.
        point_data_file: Path and filename to the point data file (parquet) with X, Y, Z, and data columns (prediction, probability).
        fname_out: Path and filename to the output file to save the mesh with mapped data.
        vtk_meta_text: Text to add to the second line of the VTK file.
        target: The target array to interpolate onto the mesh. Default: 'predictions'
            choices: 'predictions', 'probabilities'
    
    """
    def __init__(self, path_data_export, mesh_file, point_data_file, fname_out, vtk_meta_text="", target='predictions'):
        self.mesh_file = mesh_file
        self.point_data_file = point_data_file
        self.path_data_export = path_data_export
        self.mesh = None
        self.mesh_itpl = None
        self.df = None
        self.predictions = None
        self.probabilities = None
        self.fname_out = fname_out
        self.vtk_meta_text = vtk_meta_text
        self.target = target

    def read_carto(self):
        """Reads a mesh file."""
        print("self.path_data_export",self.path_data_export)
        print("point_data_file",self.point_data_file)
        print("self.mesh_file",self.mesh_file)
       
        # ./deploy/data/Export_Analysis
        # carto = Carto(self.path_data_export)
      
        #mesh_file = ../deploy/data/Export_Analysis/9-1-ReLV RVp Penta.mesh
        
        # mesh_name = os.path.basename(self.mesh_file).split('.')[0]
        # print("self.mesh_file:",self.mesh_file)
        # print("mesh_name",mesh_name)
        # idx = [i for i, s in enumerate(carto) if mesh_name in s][0]
        # self.mesh = carto[idx].mesh
        # mesh_og = carto[idx].mesh
        # print(type(mesh_og))
        # print(type(mesh_og.pv_mesh()))

        mesh_ug, header = read_mesh_file(self.mesh_file)
        self.mesh = mesh_ug.extract_surface()
        print(header)
        
        #points_carto = carto[mesh_name].points
        #self.points_number = list(points_carto.keys())
        #print("Mesh loaded with", self.mesh.n_points, "points and", self.mesh.n_cells, "cells.")


    def process_point_data(self):
        """Reads a Parquet file containing X, Y, Z coordinates and 'prediction' data."""
        df0 = pd.read_parquet(self.point_data_file)
        df0.dropna(inplace=True)
        npoints_unique = df0['Point_Number'].nunique()
        print(f"Point data loaded with {npoints_unique} points.")
        # averaging probabilities linearly (as model ensemble), alternative is taking mean of log-probabilities
        self.df = df0.groupby('Point_Number').agg({'probability':'mean'}).reset_index()
        # merge average probability with original data
        self.df = pd.merge(self.df, df0[['Point_Number', 'X', 'Y', 'Z']], on='Point_Number', how='left')
        self.df['prediction'] = self.df['probability'].apply(lambda x: 1 if x > 0.5 else -1)
        self.predictions = self.df['prediction'].values
        self.probabilities = self.df['probability'].values

    def map_point_data_onto_mesh(self, null_strategy='closest_point'):
        """
        Interpolates point data onto the mesh.
        This uses a Gaussian interpolation kernel (using default kernel: sharpness =2)

        Args:
            null_strategy: Specify a strategy to use when encountering a “null” point during the interpolation process.
                Default: 'closest_point'
                Options: 'mask_points', 'null_value', 'closest_point' 
        """
         # convert to pyvista mesh
        if self.mesh is None:
            raise ValueError("No mesh available to write.")
        mesh_pv = self.mesh #.pv_mesh()
        print(type(mesh_pv))
         # Map the data from the point cloud to the mesh
        points = pv.PolyData(self.df[['X', 'Y', 'Z']].values)
        if self.target == 'predictions':
            points['values'] = self.predictions
        elif self.target == 'probabilities':
            points['values'] = self.probabilities
        else:
            raise ValueError("target must be 'predictions' or 'probabilities'")
        # write points to vtk
        # replace .vtk in self.fname_out with _points.vtk
        fname_points = self.fname_out.replace('.vtk', '_points.vtk')
        points.save(fname_points, binary=False)
        # interpolate point data onto mesh
        self.mesh_ipl = mesh_pv.interpolate(points, strategy=null_strategy)
        # compute normals
        #self.mesh_ipl.compute_normals(point_normals=True, cell_normals=False, inplace=True)
        #possible fix on colours - no orient() method 
        self.mesh_ipl.compute_normals(point_normals=True, cell_normals=True, inplace=True, flip_normals=False)
        self.mesh_ipl = self.mesh_ipl.extract_surface().clean().triangulate()
        

    def write_mesh_with_data(self):
        """Writes the mesh with the mapped point data to a file."""
        if self.mesh_ipl is None:
            raise ValueError("No interpolated mesh available to write.")
        # save mesh with mapped data as vtk
        self.mesh_ipl.save(self.fname_out, binary=False, recompute_normals = True)


    def vtk_to_carto(self):
        """
        Converts the VTK file to a specific Carto file vtk format.
        """
        # Create a reader for the input file
        reader = vtk.vtkGenericDataObjectReader()
        reader.SetFileName(self.fname_out)
        reader.Update()  # Necessary to load the data

        # Output the read mesh to a new VTK file
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(self.fname_out)
        
        # Check if the input is PolyData
        if isinstance(reader.GetOutput(), vtk.vtkPolyData):
            writer.SetInputData(reader.GetOutput())
        else:
            raise TypeError("Input file is not a polydata type which is required for vtkPolyDataWriter")

        # Set the version to 4.2
        writer.SetFileVersion(42)
        writer.Write()
        
        # replace second line of vtk file with custom txt string :
        with open(self.fname_out, 'r') as file:
            lines = file.readlines()
        lines[1] = self.vtk_meta_text + "\n"
        with open(self.fname_out, 'w') as file:
            file.writelines(lines)
        print(f"Mesh with mapped data saved to {self.fname_out}")

    def plot_mesh_and_points(self,mesh, add_points=True):
        # plot mesh with mapped data
        p = pv.Plotter()
        p.add_mesh(mesh, scalars='values', cmap='viridis')
        if add_points:
            p.add_points(self.df[['X', 'Y', 'Z']].values, scalars=self.predictions, cmap='viridis')
        p.show()

    def run(self, null_strategy='closest_point'):
        self.read_carto()
        self.process_point_data()
        self.map_point_data_onto_mesh(null_strategy)
        self.write_mesh_with_data()
        self.vtk_to_carto()


def test_MeshDataMapper():
    #'9-LV SR Penta',
    #'9-1-ReLV RVp Penta',
    #'9-1-1-ReLV LVp Penta',
    path_data_export = '../../../data/deploy/data/Export_Analysis'
    mesh_file= '../../../data/deploy/data/Export_Analysis/9-LV SR Penta.mesh'
    #point_data_file = '../../../data/deploy/data/predictions.parquet'
    #point_data_file = '../../../data/deploy/data/S18_SR_NoScar_groundtruth.parquet'
    #point_data_file = "../../../data/deploy/data/predictions_SR_NoScar.parquet"
    point_data_file = '../../../data/deploy/data/predictions_NoScar.parquet'
    fname_out = '../../../data/deploy/data/predictions_mapped_NoScar.vtk'
    vtk_meta_text = 'PatientData S18 S18 4290_S18'

    mapper = MeshDataMapper(path_data_export, mesh_file, point_data_file, fname_out, vtk_meta_text)
    mapper.run()
    #mapper.read_carto()
    #mapper.process_point_data()
    #mapper.map_point_data_onto_mesh()
    #mapper.write_mesh_with_data()
    #mapper.vtk_to_carto()


    # map probabilities
    fname_out = '../../../data/deploy/data/probabilities_mapped_NoScar.vtk'
    mapper = MeshDataMapper(path_data_export, mesh_file, point_data_file, fname_out, vtk_meta_text, target='probabilities')
    mapper.run()
