use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyInt, PyList};

use crate::core::morton::ToCubeCode;
use crate::core::neighbor::NeighborIndices;
use crate::core::types::{CubeCode, GlobalIndexType, IndexBounds, RawIndexConversionMode};
use crate::python::types::{IntoPy, PyCubeCode};

/// Helper function to convert Py<PyArray1<GlobalIndexType>> to IndexBounds
///
/// # Arguments
///
/// * `bounds` - Python array containing 3 GlobalIndexType values
///
/// # Returns
///
/// IndexBounds (Vector3<GlobalIndexType>) or an error if the array doesn't have exactly 3 elements
fn py_array_to_index_bounds(
    bounds: Py<PyArray1<GlobalIndexType>>,
    py: Python<'_>,
) -> PyResult<IndexBounds> {
    let bounds_array = bounds.bind_borrowed(py);
    let bounds_data: PyReadonlyArray1<GlobalIndexType> = bounds_array.readonly();
    let bounds_slice = bounds_data.as_slice()?;

    if bounds_slice.len() != 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Bounds array must have exactly 3 elements",
        ));
    }

    Ok(IndexBounds::new(
        bounds_slice[0],
        bounds_slice[1],
        bounds_slice[2],
    ))
}

#[pymethods]
impl PyCubeCode {
    fn __repr__(&self, py: Python<'_>) -> Result<String, PyErr> {
        let value = self.0.bind(py).repr()?;
        Ok(format!("CubeCode({})", value))
    }
    /// Convert the cube code to a global index at the specified depth
    ///
    /// # Arguments
    ///
    /// * `depth` - The depth level for the conversion
    ///
    /// # Returns
    ///
    /// A NumPy array containing the global index coordinates
    fn to_global_index(
        &self,
        depth: usize,
        py: Python<'_>,
    ) -> PyResult<Py<PyArray1<GlobalIndexType>>> {
        let cubecode = CubeCode(self.0.bind_borrowed(py).extract()?);
        let global_index = cubecode.to_global_index(depth);
        let array = PyArray1::from_slice(py, global_index.as_slice()).unbind();
        Ok(array)
    }

    /// Get the parent cube code and local offset within the parent
    ///
    /// # Arguments
    ///
    /// * `depth` - The current depth level
    ///
    /// # Returns
    ///
    /// A tuple containing the parent cube code and local offset as NumPy array
    fn parent_and_offset_py(
        &self,
        depth: usize,
        py: Python<'_>,
    ) -> PyResult<(PyCubeCode, Py<PyArray1<u32>>)> {
        let cubecode = CubeCode(self.0.bind_borrowed(py).extract()?);
        let (parent_cubecode, local_offset) = cubecode
            .parent_and_offset(depth)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("{}", e)))?;
        let parent_cubecode = parent_cubecode.into_py(py);
        let local_offset = PyArray1::from_slice(py, local_offset.as_slice()).unbind();
        Ok((parent_cubecode, local_offset))
    }

    /// Get the children cube codes at the next depth level
    ///
    /// # Arguments
    ///
    /// * `depth` - The current depth level
    ///
    /// # Returns
    ///
    /// A NumPy array containing the 8 child cube codes
    fn children(&self, depth: usize, py: Python<'_>) -> PyResult<Py<PyList>> {
        let cubecode = CubeCode(self.0.bind_borrowed(py).extract()?);
        let codes = cubecode
            .children(depth)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
        Ok(PyList::new(
            py,
            codes
                .iter()
                .map(|&x| PyCubeCode(PyInt::new(py, x).unbind())),
        )
        .unwrap()
        .into())
    }

    /// Get the neighbor cube codes as a list (efficient version)
    ///
    /// # Arguments
    ///
    /// * `depth` - The current depth level
    /// * `bounds` - The grid bounds as a NumPy array
    /// * `mode` - The boundary conversion mode
    /// * `include_self` - Whether to include the center cell itself
    ///
    /// # Returns
    ///
    /// A Python list of neighbor cube codes (or None for out-of-bounds)
    fn neighbor_codes(
        &self,
        depth: usize,
        bounds: Py<PyArray1<GlobalIndexType>>,
        mode: Py<RawIndexConversionMode>,
        include_self: bool,
        py: Python<'_>,
    ) -> PyResult<Py<PyList>> {
        let cubecode = CubeCode(self.0.bind_borrowed(py).extract()?);
        let mode = *mode.bind(py).borrow();
        let global_index = cubecode.to_global_index(depth);
        let bounds = py_array_to_index_bounds(bounds, py)?;

        let neighbors = global_index.neighbor_indices(&bounds, mode, include_self);

        let pylist = PyList::empty(py);
        for opt_index in neighbors {
            match opt_index {
                Some(index) => {
                    let cubecode = index.to_cubecode(depth);
                    let py_cubecode = Py::new(py, PyCubeCode(PyInt::new(py, cubecode.0).unbind()))?;
                    pylist.append(py_cubecode)?;
                }
                None => {
                    pylist.append(py.None())?;
                }
            }
        }
        Ok(pylist.into())
    }

    /// Get the raw cube code value
    ///
    /// # Returns
    ///
    /// The raw 128-bit cube code value
    fn value(&self, py: Python<'_>) -> Py<PyInt> {
        self.0.clone_ref(py)
    }
}
