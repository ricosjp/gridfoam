use crate::core::mesh::BBoxOps;
use crate::core::octree::Grid;
use crate::core::types::{NodeType, OctreeLevel};
extern crate hdf5_metno as hdf5;
use hdf5::types::FixedAscii;

/// Trait for generating node data from octree levels
///
/// This trait provides methods to generate node data from octree levels
/// for efficient spatial partitioning and visualization.
trait NodeData {
    /// Generate amrboxes from an octree level
    ///
    /// # Arguments
    ///
    /// * `depth` - The depth level to generate AMR boxes for
    /// * `only_leaves` - Whether to only include leaves
    ///
    /// # Returns
    ///
    /// (nrows, 6, data)
    /// A tuple of the number of rows, the number of columns, and the AMR boxes
    ///
    /// The array is of shape (nrows, 6), where nrows is the number of nodes in the octree level
    /// and the 6 values are the x, y, z coordinates of the lower and upper corners of the AMR box
    fn amrboxes(&self, only_leaves: bool) -> (usize, usize, Vec<u64>);

    /// Generate node types from an octree level
    ///
    /// # Arguments
    ///
    /// * `only_leaves` - Whether to only include leaves
    ///
    /// # Returns
    ///
    /// (nrows, data)
    /// A tuple of the number of rows and the node types
    ///
    /// The array is of shape (nrows, ), where nrows is the number of nodes in the octree level
    /// and the values are the node types
    fn node_types(&self, only_leaves: bool) -> (usize, Vec<u8>);

    /// Generate depths from an octree level
    ///
    /// # Arguments
    ///
    /// * `only_leaves` - Whether to only include leaves
    ///
    /// # Returns
    ///
    /// (nrows, data)
    /// A tuple of the number of rows and the depths
    ///
    /// The array is of shape (nrows, ), where nrows is the number of nodes in the octree level
    /// and the values are the depths
    fn depths(&self, only_leaves: bool) -> (usize, Vec<u8>);
}

impl NodeData for OctreeLevel {
    fn amrboxes(&self, only_leaves: bool) -> (usize, usize, Vec<u64>) {
        let data = self
            .nodes
            .iter()
            .filter(|(_, node)| !only_leaves || matches!(node.node_type, NodeType::Leaf))
            .map(|(cubecode, _)| {
                cubecode
                    .to_global_index(self.depth)
                    .into_iter()
                    .flat_map(|&x| [x, x])
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        (data.len(), 6, data.concat())
    }

    fn node_types(&self, only_leaves: bool) -> (usize, Vec<u8>) {
        let data = self
            .nodes
            .iter()
            .filter(|(_, node)| !only_leaves || matches!(node.node_type, NodeType::Leaf))
            .map(|(_, node)| node.node_type.clone() as u8)
            .collect::<Vec<_>>();
        (data.len(), data)
    }

    fn depths(&self, only_leaves: bool) -> (usize, Vec<u8>) {
        let data = self
            .nodes
            .iter()
            .filter(|(_, node)| !only_leaves || matches!(node.node_type, NodeType::Leaf))
            .map(|(_, _)| self.depth as u8)
            .collect::<Vec<_>>();
        (data.len(), data)
    }
}

/// Trait for HDF5 I/O operations on grid structures
///
/// This trait provides methods for saving grid data to HDF5 format files,
/// enabling efficient storage and visualization of octree structures.
pub trait Hdf5io {
    /// Save grid nodes to an HDF5 file
    ///
    /// # Arguments
    ///
    /// * `filename` - Path to the output HDF5 file
    /// * `only_leaves` - Whether to only save leaf nodes
    ///
    /// # Returns
    ///
    /// A result indicating success or failure of the save operation
    fn save_nodes(&self, filename: &str, only_leaves: bool) -> std::io::Result<()>;
}

impl Hdf5io for Grid {
    fn save_nodes(&self, filename: &str, only_leaves: bool) -> std::io::Result<()> {
        let filename = std::path::PathBuf::from(filename);
        let f = hdf5::File::create(filename)?;

        // Write grid description
        let grp = f.create_group("VTKHDF")?;
        let s = FixedAscii::<3>::from_ascii("XYZ").unwrap();
        grp.new_attr::<FixedAscii<3>>()
            .create("GridDescription")?
            .write_scalar(&s)?;
        grp.new_attr::<f64>()
            .shape(3)
            .create("Origin")?
            .write(&self.domain.lower())?;
        let s = FixedAscii::<14>::from_ascii("OverlappingAMR").unwrap();
        grp.new_attr::<FixedAscii<14>>()
            .create("Type")?
            .write_scalar(&s)?;
        grp.new_attr::<i64>()
            .shape(2)
            .create("Version")?
            .write(&[2, 3])?;

        // Write grid levels
        for (depth, octree_level) in self.octree_levels.iter().enumerate() {
            let octree_size = 1 << depth;
            let bounds = self.blocksize * octree_size;
            let level_grp = grp.create_group(&format!("Level{}", depth))?;
            level_grp
                .new_attr::<f64>()
                .shape(3)
                .create("Spacing")?
                .write(&self.domain.dx(&bounds))?;

            let (nrow, ncol, amrbox_data) = octree_level.amrboxes(only_leaves);
            level_grp
                .new_dataset::<u64>()
                .shape((nrow, ncol))
                .create("AMRBox")?
                .write_raw(&amrbox_data)?;
            let cell_data_grp = level_grp.create_group("CellData")?;
            let (nrow, node_types_data) = octree_level.node_types(only_leaves);
            cell_data_grp
                .new_dataset::<u8>()
                .shape((nrow,))
                .create("cube_type")?
                .write(&node_types_data)?;
            let (nrow, depths_data) = octree_level.depths(only_leaves);
            cell_data_grp
                .new_dataset::<u8>()
                .shape((nrow,))
                .create("depth")?
                .write(&depths_data)?;
            let _ = level_grp.create_group("PointData")?;
            let _ = level_grp.create_group("FieldData")?;
        }
        f.close()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::mesh::TriangleMesh;
    use crate::core::octree::OctreeBuilder;
    use crate::io::config::Config;

    use rstest::{fixture, rstest};
    use std::path::Path;

    #[fixture]
    fn simple_mesh() -> TriangleMesh {
        TriangleMesh::from_stl_file("../../tests/data/stl/bunny.stl")
    }

    #[fixture]
    fn config() -> Config {
        let filepath = Path::new("../../tests/data/yaml/bunny.yaml");
        Config::from_file(filepath).unwrap()
    }

    #[fixture]
    fn grid(simple_mesh: TriangleMesh, config: Config) -> Grid {
        let builder = OctreeBuilder::new(config);
        builder.build_with_mesh(simple_mesh)
    }

    #[rstest]
    fn test_save_nodes(grid: Grid) {
        let temp_file = tempfile::NamedTempFile::new().unwrap();
        let temp_path = temp_file.path().to_str().unwrap();

        grid.save_nodes(temp_path, false).unwrap();

        assert!(std::path::Path::new(temp_path).exists());
    }
}
