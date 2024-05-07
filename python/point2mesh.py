"""
Interpolate point data onto mesh geometry

Functionality:
- read in carto mesh file
- read in point data file (parquet) with X, Y, Z, and data columns (prediction, probability)
- interpolate point data onto mesh using Gaussian kernel
- write out vtk file with mapped point data

need pip install carto-reader
"""

import os
import pyvista as pv
import pandas as pd
from carto_reader.carto_data import Carto

class MeshDataMapper:
    def __init__(self, path_data_export. mesh_file, point_data_file):
        self.mesh_file = mesh_file
        self.point_data_file = point_data_file
        self.path_data_export = path_data_export
        self.mesh = None
        self.mesh_itpl = None
        self.points = None
        self.points_number = None
        self.point_data = None
        self.df = None
        self.predictions = None
        self.probabilities = None

    def read_carto(self):
        """Reads a mesh file."""
        carto = Carto(self.path_data_export)
        # list(carto)
        # find index of mesh file basename (w/o .mesh) in list of carto
        mesh_name = os.path.basename(self.mesh_file).split('.')[0]
        idx = [i for i, s in enumerate(carto) if mesh_name in s][0]
        self.mesh = carto[idx].mesh
        points_carto = carto[mesh_name].points
        self.points_number = list(points_carto.keys())
        #print("Mesh loaded with", self.mesh.n_points, "points and", self.mesh.n_cells, "cells.")

    def read_point_data(self):
        """Reads a Parquet file containing X, Y, Z coordinates and 'prediction' data."""
        df_pred = pd.read_parquet(self.point_data_file)
        print("Point data loaded with", len(self.point_data), "points.")
        df_pred['Point_Number'] = df_pred['Point_Number'].astype(str)
        df_pred = df_pred[df_pred['Point_Number'].isin(self.points_number)]
        self.df = pd.DataFrame(self.points_number, columns=['Point_Number'])
        self.df = self.df.merge(df_pred, on='Point_Number', how='left')
        self.predictions = self.df['prediction'].values
        self.predictions = self.predictions.astype(int)
        self.probabilities = self.df['probability'].values

    def map_point_data_onto_mesh(self, vtk_meta_txt, null_strategy='closest_point'):
        """
        Interpolates point data onto the mesh.
        This uses a Gaussian interpolation kernel (using default kernel: sharpness =2)

        Args:
            vtk_meta_txt: string with metadata to write to the vtk file
            null_strategy: Specify a strategy to use when encountering a “null” point during the interpolation process.
                Default: 'closest_point'
                Options: 'mask_points', 'null_value', 'closest_point' 
        """
         # convert to pyvista mesh
        if self.mesh is None:
            raise ValueError("No mesh available to write.")
        mesh_pv = self.mesh.pv_mesh()
         # Map the data from the point cloud to the mesh
        points = pv.PolyData(self.df[['X', 'Y', 'Z']].values)
        points['values'] = self.predictions
        points['probabilities'] = self.probabilities
        # interpolate point data onto mesh
        self.mesh_ipl = mesh_pv.interpolate(points, strategy=null_strategy)

    def write_mesh_with_data(self, output_file, vtk_meta_txt):
        """Writes the mesh with the mapped point data to a file."""
        if self.mesh_ipl is None:
            raise ValueError("No interpolated mesh available to write.")
        # save mesh with mapped data as vtk
        self.mesh_ipl.save(self.path_out, binary=False)
        # replace second line of vtk file with txt string :
        with open(self.path_out, 'r') as file:
            lines = file.readlines()
        lines[1] = vtk_meta_txt + "\n"
        with open(self.path_out, 'w') as file:
            file.writelines(lines)
        print(f"Mesh with mapped data saved to {self.path_out}")

    def plot_mesh_and_points(mesh, add_points=True):
        # plot mesh with mapped data
        p = pv.Plotter()
        p.add_mesh(mesh, scalars='values', cmap='viridis')
        if add_points:
            p.add_points(self.df[['X', 'Y', 'Z']].values, scalars=self.predictions, cmap='viridis')
        p.show()


def test_MeshDataMapper():
    path_data_export = '../../../data/deploy/data/Export_Analysis'
    mesh_file= '../../../data/deploy/data/Export_Analysis/9-LV SR Penta.mesh'
    point_data_file = '../../../data/deploy/data/predictions.parquet'
    path_out = '../../../data/deploy/data/predictions_mapped.vtk'
    vtk_meta_text = "PatientData S18 S18 4290_S18”

    mapper = MeshDataMapper(path_data_export, mesh_file, point_data_file)
    mapper.read_carto()
    mapper.read_point_data()
    mapper.map_point_data_onto_mesh()
    mapper.write_mesh_with_data(path_out, vtk_meta_text)
