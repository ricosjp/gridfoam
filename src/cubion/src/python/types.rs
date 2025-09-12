use crate::core::mesh::BBox;
use crate::core::octree::{Grid, NodeType, OctreeLevel, OctreeNode};
use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyList};

/// Trait for converting Rust types to Python objects
///
/// This trait provides a generic interface for converting Rust data structures
/// to their corresponding Python representations without taking ownership.
pub trait IntoPy {
    /// The target Python type
    type Target;

    /// Convert the Rust type to a Python object
    ///
    /// # Arguments
    ///
    /// * `py` - Python interpreter instance
    ///
    /// # Returns
    ///
    /// The converted Python object
    fn into_py(&self, py: Python) -> Self::Target;
}

#[pyclass]
/// Python representation of a bounding box
///
/// Contains the lower and upper bounds as NumPy arrays for efficient
/// data transfer between Rust and Python.
pub struct PyBBox {
    #[pyo3(get)]
    /// Lower bounds of the bounding box
    lower: Py<PyArray1<f64>>,
    #[pyo3(get)]
    /// Upper bounds of the bounding box
    upper: Py<PyArray1<f64>>,
}

#[pymethods]
impl PyBBox {
    fn __repr__(&self) -> Result<String, PyErr> {
        Python::attach(|py| {
            let lower = self.lower.bind(py).repr()?;
            let upper = self.upper.bind(py).repr()?;
            Ok(format!("BBox(lower: {}, upper: {})", lower, upper))
        })
    }
}

impl IntoPy for BBox {
    type Target = PyBBox;

    /// Convert BBox to Python representation
    ///
    /// # Arguments
    ///
    /// * `py` - Python interpreter instance
    ///
    /// # Returns
    ///
    /// A PyBBox containing the lower and upper bounds as NumPy arrays
    fn into_py(&self, py: Python) -> Self::Target {
        PyBBox {
            lower: self.lower().to_pyarray(py).unbind(),
            upper: self.upper().to_pyarray(py).unbind(),
        }
    }
}

#[pyclass]
/// Python representation of an octree node
///
/// Contains the node's spatial identifier, associated face IDs, and node type
/// for efficient data transfer between Rust and Python.
pub struct PyOctreeNode {
    #[pyo3(get)]
    /// Unique identifier for this node in the octree hierarchy
    cubecode: Py<PyInt>,
    #[pyo3(get)]
    /// Face IDs that intersect with this node's bounding box
    face_ids: Py<PyArray1<usize>>,
    #[pyo3(get)]
    /// Type of this node (leaf, ghost, etc.)
    node_type: Py<NodeType>,
}

#[pymethods]
impl PyOctreeNode {
    fn __repr__(&self) -> Result<String, PyErr> {
        Python::attach(|py| {
            let cubecode = self.cubecode.bind(py).repr()?;
            let face_ids = self.face_ids.bind(py).repr()?;
            Ok(format!(
                "OctreeNode(cubecode: {}, face_ids: {}, node_type: {})",
                cubecode, face_ids, self.node_type
            ))
        })
    }
}

impl IntoPy for OctreeNode {
    type Target = PyOctreeNode;

    /// Convert OctreeNode to Python representation
    ///
    /// # Arguments
    ///
    /// * `py` - Python interpreter instance
    ///
    /// # Returns
    ///
    /// A PyOctreeNode containing the node's data as Python objects
    fn into_py(&self, py: Python) -> Self::Target {
        PyOctreeNode {
            cubecode: PyInt::new(py, self.cubecode.0).into(),
            face_ids: self.face_ids.to_pyarray(py).unbind(),
            node_type: Py::new(py, self.node_type.clone()).unwrap(),
        }
    }
}

/// Python representation of an octree level
///
/// Contains all nodes at a specific depth level as a Python dictionary
/// for efficient data transfer between Rust and Python.
#[pyclass]
pub struct PyOctreeLevel {
    #[pyo3(get)]
    /// Map of cube codes to nodes at this level
    nodes: Py<PyDict>,
    #[pyo3(get)]
    /// Grid bounds for this level
    bounds: Py<PyArray1<u64>>,
    #[pyo3(get)]
    /// Depth level (0 = root)
    depth: Py<PyInt>,
}

#[pymethods]
impl PyOctreeLevel {
    fn __repr__(&self) -> Result<String, PyErr> {
        Python::attach(|py| {
            let nodes_dict = self.nodes.bind(py);
            let bounds = self.bounds.bind(py).repr()?;
            let depth = self.depth.bind(py).repr()?;

            // take the first 5 nodes
            let mut node_preview = String::new();
            for (i, (key, value)) in nodes_dict.iter().enumerate() {
                if i >= 5 {
                    node_preview.push_str("\n\t\t...");
                    break;
                }
                if i > 0 {
                    node_preview.push_str(", ");
                }
                node_preview.push_str(format!("\n\t\t{}: {}", key.repr()?, value.repr()?).as_str());
            }

            Ok(format!(
                "OctreeLevel(\n\tnodes: {{{}}}, \n\tbounds: {}, \n\tdepth: {})",
                node_preview, bounds, depth
            ))
        })
    }
}

impl IntoPy for OctreeLevel {
    type Target = PyOctreeLevel;

    /// Convert OctreeLevel to Python representation
    ///
    /// # Arguments
    ///
    /// * `py` - Python interpreter instance
    ///
    /// # Returns
    ///
    /// A PyOctreeLevel containing the level's data as Python objects
    fn into_py(&self, py: Python) -> Self::Target {
        let nodes_dict = PyDict::new(py);
        for (cubecode, node) in self.nodes.iter() {
            let py_cubecode = PyInt::new(py, cubecode.0);
            let py_node = node.into_py(py);
            let _ = nodes_dict.set_item(py_cubecode, py_node);
        }
        PyOctreeLevel {
            nodes: nodes_dict.into(),
            bounds: PyArray1::from_slice(py, self.bounds.as_slice()).unbind(),
            depth: PyInt::new(py, self.depth).into(),
        }
    }
}

#[pyclass]
/// Python representation of a complete octree grid
///
/// Contains the spatial domain, block size, maximum depth, and all octree levels
/// for efficient data transfer between Rust and Python.
pub struct PyGrid {
    #[pyo3(get)]
    /// Spatial domain of the octree
    domain: Py<PyBBox>,
    #[pyo3(get)]
    /// Block size at the root level
    blocksize: Py<PyArray1<u64>>,
    #[pyo3(get)]
    /// Maximum depth reached during construction
    max_depth: Py<PyInt>,
    #[pyo3(get)]
    /// All octree levels in compact representation
    octree_levels: Py<PyList>,
}

#[pymethods]
impl PyGrid {
    fn __repr__(&self) -> Result<String, PyErr> {
        Python::attach(|py| {
            let domain = self.domain.bind(py).repr()?;
            let blocksize = self.blocksize.bind(py).repr()?;
            let max_depth = self.max_depth.bind(py).repr()?;
            let octree_levels_list = self.octree_levels.bind(py);
            let levels_count = octree_levels_list.len();
            let octree_levels_preview = if levels_count > 0 {
                format!("[OctreeLevel(...), ...] ({} levels)", levels_count)
            } else {
                "[]".to_string()
            };
            Ok(format!(
                "Grid(domain: {}, blocksize: {}, max_depth: {}, octree_levels: {})",
                domain, blocksize, max_depth, octree_levels_preview
            ))
        })
    }
}

impl IntoPy for Grid {
    type Target = PyGrid;

    /// Convert Grid to Python representation
    ///
    /// # Arguments
    ///
    /// * `py` - Python interpreter instance
    ///
    /// # Returns
    ///
    /// A PyGrid containing the complete grid structure as Python objects
    fn into_py(&self, py: Python) -> Self::Target {
        let octree_levels = self
            .octree_levels
            .iter()
            .map(|level| level.into_py(py))
            .collect::<Vec<_>>();
        PyGrid {
            domain: Py::new(py, self.domain.into_py(py)).unwrap(),
            blocksize: PyArray1::from_slice(py, self.blocksize.as_slice()).unbind(),
            max_depth: PyInt::new(py, self.max_depth).into(),
            octree_levels: PyList::new(py, octree_levels).unwrap().into(),
        }
    }
}
