use crate::core::{mesh::TriangleMesh, octree::OctreeBuilder};
use crate::io::config::Config;
use crate::python::types::{IntoPy, PyGrid};
use std::path::Path;

use pyo3::prelude::*;
use pyo3::types::PyAny;
extern crate nalgebra as na;
use na::{MatrixViewXx3, MatrixXx3};
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};

impl<'source> FromPyObject<'source> for TriangleMesh {
    /// Extract TriangleMesh from Python object
    ///
    /// Converts a Python object (typically PyVista PolyData) to a TriangleMesh
    /// by extracting points and faces arrays and converting them to nalgebra matrices.
    ///
    /// # Arguments
    ///
    /// * `obj` - Python object containing 'points' and 'faces' attributes
    ///
    /// # Returns
    ///
    /// A TriangleMesh constructed from the Python data
    ///
    /// # Panics
    ///
    /// Panics if the Python object doesn't have the required attributes or
    /// if the array dimensions are incompatible
    fn extract_bound(obj: &Bound<'source, PyAny>) -> PyResult<Self> {
        // Step 1: get PyVista ndarray
        let points_ndarray = obj
            .getattr("points")
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyAttributeError, _>(format!(
                    "Failed to get 'points' attribute: {}",
                    e
                ))
            })?
            .call_method0("__array__")
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to call __array__ on points: {}",
                    e
                ))
            })?;

        let faces_ndarray = obj
            .getattr("regular_faces")
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyAttributeError, _>(format!(
                    "Failed to get 'regular_faces' attribute: {}",
                    e
                ))
            })?
            .call_method0("__array__")
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                    "Failed to call __array__ on faces: {}",
                    e
                ))
            })?;

        // Step 2: extract as PyReadonlyArray2
        let py_points: PyReadonlyArray2<f32> = points_ndarray
            .extract::<PyReadonlyArray2<f32>>()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "Failed to extract points as PyReadonlyArray2<f32>: {}",
                    e
                ))
            })?;

        let py_faces: PyReadonlyArray2<i64> = faces_ndarray
            .extract::<PyReadonlyArray2<i64>>()
            .map_err(|e| {
                PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                    "Failed to extract faces as PyReadonlyArray2<i64>: {}",
                    e
                ))
            })?;

        // Step 3: get slice
        let points_slice = py_points.as_slice().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to get points as slice: {}",
                e
            ))
        })?;

        let faces_slice = py_faces.as_slice().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Failed to get faces as slice: {}",
                e
            ))
        })?;

        // Step 4: get ownership
        let points: MatrixXx3<f64> = MatrixXx3::from_row_slice(points_slice).cast::<f64>();
        let faces: MatrixXx3<usize> = MatrixXx3::from_row_slice(faces_slice)
            .try_cast::<usize>()
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Failed to cast face ids to usize")
            })?;

        // Step 5 create TriangleMesh
        Ok(TriangleMesh::new(points, faces))
    }
}

/// Generate an octree grid from PyVista PolyData
///
/// This function creates an octree grid structure from a PyVista PolyData object
/// using the specified configuration file. The function loads the mesh data,
/// applies the octree refinement strategy, and returns the resulting grid.
///
/// # Arguments
///
/// * `py` - Python interpreter instance
/// * `polydata` - PyVista PolyData object containing mesh geometry
/// * `config_path` - Path to the YAML configuration file
///
/// # Returns
///
/// A PyGrid containing the complete octree structure
///
/// # Panics
///
/// Panics if the configuration file cannot be loaded or if the mesh extraction fails
#[pyfunction]
pub fn generate_grid_from_polydata(
    py: Python,
    polydata: &Bound<'_, PyAny>,
    config_path: &str,
) -> PyResult<PyGrid> {
    let config_path = Path::new(config_path);
    let mesh = TriangleMesh::extract_bound(polydata).unwrap();
    let config = Config::from_file(config_path).unwrap();

    // Build octree
    let builder = OctreeBuilder::new(config);
    let grid = builder.build_with_mesh(mesh);

    Ok(grid.into_py(py))
}
