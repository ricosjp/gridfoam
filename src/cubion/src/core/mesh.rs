extern crate nalgebra as na;
use crate::core::types::{BBox, CubeCode, IndexBounds, Point};
use na::{Matrix2x3, MatrixXx3, Vector3};
use rstar::{Envelope, RTree, RTreeObject};
use std::fs::File;

/// Extension trait for AABB (Axis-Aligned Bounding Box) operations
///
/// This trait provides additional methods for AABB calculations including
/// diagonal computation, bounds extraction, and spatial splitting.
pub trait BBoxOps {
    /// Calculate the diagonal vector of the AABB
    ///
    /// # Returns
    ///
    /// A 3D vector representing the diagonal from lower to upper corner
    fn diag(&self) -> Vector3<f64>;

    /// Extract bounds as a 2x3 matrix
    ///
    /// # Returns
    ///
    /// A 2x3 matrix where the first row contains lower bounds and
    /// the second row contains upper bounds
    fn bounds(&self) -> Matrix2x3<f64>;

    /// Split the AABB into 8 sub-AABBs
    ///
    /// # Returns
    ///
    /// A vector of 8 AABBs representing the octants of the original AABB
    fn split(&self) -> Vec<BBox>;

    /// Calculate the size of the AABB in each dimension
    ///
    /// # Arguments
    ///
    /// * `bounds` - The bounds of the AABB
    ///
    /// # Returns
    ///
    /// A vector of 3 elements representing the size of the AABB in each dimension
    fn dx(&self, bounds: &Vector3<u64>) -> Vector3<f64>;

    /// Calculate an AABB from a CubeCode
    ///
    /// # Arguments
    ///
    /// * `cubecode` - The CubeCode to create the AABB from
    /// * `bounds` - The bounds of the AABB
    /// * `depth` - The depth of the AABB
    ///
    /// # Returns
    ///
    /// An AABB created from the CubeCode
    fn calculate_bbox_from(&self, cubecode: &CubeCode, bounds: &IndexBounds, depth: usize) -> BBox;
}

impl BBoxOps for BBox {
    fn diag(&self) -> Vector3<f64> {
        let upper: Vector3<f64> = self.upper().into();
        let lower: Vector3<f64> = self.lower().into();
        upper - lower
    }

    fn bounds(&self) -> Matrix2x3<f64> {
        let lower: Vector3<f64> = self.lower().into();
        let upper: Vector3<f64> = self.upper().into();
        Matrix2x3::new(lower[0], lower[1], lower[2], upper[0], upper[1], upper[2])
    }

    fn split(&self) -> Vec<BBox> {
        let center = self.center();
        let xs = [self.lower()[0], center[0], self.upper()[0]];
        let ys = [self.lower()[1], center[1], self.upper()[1]];
        let zs = [self.lower()[2], center[2], self.upper()[2]];

        let mut result = Vec::with_capacity(8);

        for idx in 0..8 {
            let i = (idx & 1) as usize;
            let j = ((idx >> 1) & 1) as usize;
            let k = ((idx >> 2) & 1) as usize;

            let min = [xs[i], ys[j], zs[k]];
            let max = [xs[i + 1], ys[j + 1], zs[k + 1]];
            result.push(BBox::from_corners(min, max));
        }

        result
    }

    fn dx(&self, bounds: &Vector3<u64>) -> Vector3<f64> {
        self.diag().component_div(&bounds.cast::<f64>())
    }

    fn calculate_bbox_from(
        &self,
        cubecode: &CubeCode,
        bounds: &Vector3<u64>,
        depth: usize,
    ) -> BBox {
        let dx = self.dx(bounds);
        let domain_lower: Vector3<f64> = self.lower().into();
        let global_index = cubecode.to_global_index(depth);
        let bbox_lower: Vector3<f64> = global_index.cast::<f64>().component_mul(&dx) + domain_lower;
        let bbox_upper: Vector3<f64> = (global_index.add_scalar(1))
            .cast::<f64>()
            .component_mul(&dx)
            + domain_lower;
        BBox::from_corners(bbox_lower.into(), bbox_upper.into())
    }
}

/// A triangle in 3D space with three vertices
///
/// Represents a triangular face with an ID and three vertex coordinates.
/// Used for spatial indexing and intersection calculations.
struct Triangle {
    /// Unique identifier for this triangle
    id: usize,
    /// Three vertices defining the triangle
    vertices: [Point; 3],
}

impl Triangle {
    pub fn new(id: usize, vertices: [Point; 3]) -> Self {
        Self { id, vertices }
    }

    /// Calculate the axis-aligned bounding box of this triangle
    ///
    /// # Returns
    ///
    /// An AABB that completely contains the triangle
    fn bbox(&self) -> BBox {
        let v0 = &self.vertices[0];
        let v1 = &self.vertices[1];
        let v2 = &self.vertices[2];

        let mut min = [0.0; 3];
        let mut max = [0.0; 3];
        for i in 0..3 {
            min[i] = v0[i].min(v1[i].min(v2[i]));
            max[i] = v0[i].max(v1[i].max(v2[i]));
        }
        BBox::from_corners(min, max)
    }
}

impl RTreeObject for Triangle {
    type Envelope = BBox;

    /// Get the envelope (bounding box) for spatial indexing
    ///
    /// # Returns
    ///
    /// The AABB of this triangle for use in R-tree spatial indexing
    fn envelope(&self) -> Self::Envelope {
        self.bbox()
    }
}

/// A triangle mesh with spatial indexing capabilities
///
/// Contains vertex coordinates, face connectivity, and an R-tree for
/// efficient spatial queries and intersection calculations.
pub struct TriangleMesh {
    /// Vertex coordinates as an Nx3 matrix
    points: MatrixXx3<f64>,
    /// Face connectivity as an Mx3 matrix of vertex indices
    faces: MatrixXx3<usize>,
    /// R-tree for spatial indexing of triangles
    rtree: RTree<Triangle>,
}

impl TriangleMesh {
    pub fn new(points: MatrixXx3<f64>, faces: MatrixXx3<usize>) -> Self {
        let triangles: Vec<Triangle> = faces
            .row_iter()
            .enumerate()
            .map(|(fid, face)| {
                let vertices = [
                    [
                        points.row(face[0])[0],
                        points.row(face[0])[1],
                        points.row(face[0])[2],
                    ],
                    [
                        points.row(face[1])[0],
                        points.row(face[1])[1],
                        points.row(face[1])[2],
                    ],
                    [
                        points.row(face[2])[0],
                        points.row(face[2])[1],
                        points.row(face[2])[2],
                    ],
                ];
                Triangle::new(fid, vertices)
            })
            .collect();
        Self {
            points,
            faces,
            rtree: RTree::bulk_load(triangles),
        }
    }

    /// Create a triangle mesh from an STL file
    ///
    /// # Arguments
    ///
    /// * `filename` - Path to the STL file
    ///
    /// # Returns
    ///
    /// A new TriangleMesh with loaded geometry and spatial indexing
    ///
    /// # Panics
    ///
    /// Panics if the file cannot be opened or read, or if the STL format is invalid
    pub fn from_stl_file(filename: &str) -> TriangleMesh {
        // Error handling using map_err for file open and STL read
        let mut file = File::open(filename)
            .map_err(|e| panic!("Failed to open file {}: {}", filename, e))
            .unwrap();
        let stl = stl_io::read_stl(&mut file)
            .map_err(|e| panic!("Failed to read STL file {}: {}", filename, e))
            .unwrap();

        let n_points = stl.vertices.len();
        let n_faces = stl.faces.len();

        let points = MatrixXx3::from_fn(n_points, |i, j| stl.vertices[i].0[j] as f64);
        let faces = MatrixXx3::from_fn(n_faces, |i, j| stl.faces[i].vertices[j]);

        Self::new(points, faces)
    }

    /// Find all face IDs that intersect with the given query box
    ///
    /// # Arguments
    ///
    /// * `query_box` - The AABB to test for intersections
    ///
    /// # Returns
    ///
    /// A vector of face IDs that intersect with the query box
    pub fn find_intersecting_face_ids(&self, query_box: BBox) -> Vec<usize> {
        self.rtree
            .locate_in_envelope_intersecting(&query_box)
            .map(|tri| tri.id)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::{fixture, rstest};

    /// Create a simple BBox for testing
    #[fixture]
    fn simple_bbox() -> BBox {
        BBox::from_corners([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
    }

    /// Test BBox diagonal calculation
    #[rstest]
    #[case(Vector3::new(1.0, 1.0, 1.0))]
    fn test_bbox_diag(simple_bbox: BBox, #[case] expected: Vector3<f64>) {
        assert_eq!(simple_bbox.diag(), expected);
    }

    /// Test BBox bounds extraction
    #[rstest]
    #[case(Matrix2x3::new(0.0, 0.0, 0.0, 1.0, 1.0, 1.0))]
    fn test_bbox_bounds(simple_bbox: BBox, #[case] expected: Matrix2x3<f64>) {
        assert_eq!(simple_bbox.bounds(), expected);
    }

    /// Test BBox splitting into 8 octants
    #[rstest]
    #[case(vec![BBox::from_corners([0.0, 0.0, 0.0], [0.5, 0.5, 0.5]),
        BBox::from_corners([0.5, 0.0, 0.0], [1.0, 0.5, 0.5]),
        BBox::from_corners([0.0, 0.5, 0.0], [0.5, 1.0, 0.5]),
        BBox::from_corners([0.5, 0.5, 0.0], [1.0, 1.0, 0.5]),
        BBox::from_corners([0.0, 0.0, 0.5], [0.5, 0.5, 1.0]),
        BBox::from_corners([0.5, 0.0, 0.5], [1.0, 0.5, 1.0]),
        BBox::from_corners([0.0, 0.5, 0.5], [0.5, 1.0, 1.0]),
        BBox::from_corners([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]),
    ])]
    fn test_bbox_split(simple_bbox: BBox, #[case] expected_sub_aabbs: Vec<BBox>) {
        for (i, sub_bbox) in simple_bbox.split().iter().enumerate() {
            assert_eq!(sub_bbox, &expected_sub_aabbs[i]);
        }
    }

    /// Test AABB dx calculation
    #[rstest]
    #[case(Vector3::new(2, 3, 4), Vector3::new(1.0/2.0, 1.0/3.0, 1.0/4.0))]
    fn test_bbox_dx(
        simple_bbox: BBox,
        #[case] bounds: Vector3<u64>,
        #[case] expected: Vector3<f64>,
    ) {
        assert_eq!(simple_bbox.dx(&bounds), expected);
    }

    /// Test AABB calculation from CubeCode
    #[rstest]
    #[case(CubeCode(0x0), Vector3::new(2, 3, 4), 0, BBox::from_corners([0.0, 0.0, 0.0], [1.0/2.0, 1.0/3.0, 1.0/4.0]))]
    fn test_bbox_calculate_bbox_from(
        simple_bbox: BBox,
        #[case] cubecode: CubeCode,
        #[case] bounds: Vector3<u64>,
        #[case] depth: usize,
        #[case] expected: BBox,
    ) {
        assert_eq!(
            simple_bbox.calculate_bbox_from(&cubecode, &bounds, depth),
            expected
        );
    }

    /// Test triangle querying with R-tree
    #[test]
    fn test_query_triangles() {
        let mesh = vec![
            Triangle {
                id: 0,
                vertices: [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            },
            Triangle {
                id: 1,
                vertices: [[2.0, 2.0, 0.0], [3.0, 2.0, 0.0], [2.0, 3.0, 0.0]],
            },
        ];

        let tree = RTree::bulk_load(mesh);

        let query_box = BBox::from_corners([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]);

        let hits = tree
            .locate_in_envelope_intersecting(&query_box)
            .map(|tri| tri.id)
            .collect::<Vec<usize>>();
        assert_eq!(hits, vec![0]);
    }

    /// Test loading mesh from STL file
    #[rstest]
    #[case("../../tests/data/stl/bunny.stl")]
    fn test_from_stl(#[case] filename: &str) {
        let mesh = TriangleMesh::from_stl_file(filename);
        assert_eq!(mesh.points.nrows(), 14290);
        assert_eq!(mesh.faces.nrows(), 28576);
    }
}
