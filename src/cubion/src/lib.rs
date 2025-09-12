mod core;
mod io;
mod python;

use crate::core::octree::NodeType;
use crate::python::bindings::generate_grid_from_polydata;
use crate::python::types::{PyBBox, PyGrid, PyOctreeLevel, PyOctreeNode};

use pyo3::prelude::*;

#[pymodule]
fn cubion(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(generate_grid_from_polydata))?;

    // Register Python classes
    m.add_class::<NodeType>()?;
    m.add_class::<PyBBox>()?;
    m.add_class::<PyOctreeNode>()?;
    m.add_class::<PyOctreeLevel>()?;
    m.add_class::<PyGrid>()?;
    Ok(())
}
