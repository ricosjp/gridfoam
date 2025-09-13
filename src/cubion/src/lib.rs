mod core;
mod io;
mod python;

use crate::core::neighbor::DIRECTIONS;
use crate::core::types::{NodeType, RawIndexConversionMode};
use crate::python::bindings::generate_grid_from_polydata;
use crate::python::types::{PyBBox, PyCubeCode, PyGrid, PyOctreeLevel, PyOctreeNode};
use nalgebra::SMatrix;
use numpy::ToPyArray;
use pyo3::prelude::*;

#[pymodule]
fn cubion(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(generate_grid_from_polydata))?;

    let pydirections: SMatrix<i64, 27, 3> = DIRECTIONS.transpose();
    let pydirections = pydirections.to_pyarray(py);
    m.add("DIRECTIONS", pydirections)?;

    // Register Python classes
    m.add_class::<PyCubeCode>()?;
    m.add_class::<NodeType>()?;
    m.add_class::<RawIndexConversionMode>()?;
    m.add_class::<PyBBox>()?;
    m.add_class::<PyOctreeNode>()?;
    m.add_class::<PyOctreeLevel>()?;
    m.add_class::<PyGrid>()?;
    Ok(())
}
