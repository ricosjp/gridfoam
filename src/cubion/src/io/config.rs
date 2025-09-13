use crate::core::constants::{MAX_BLOCK_AXIS, MAX_OCTREE_DEPTH};
use crate::core::types::{BBox, IndexBounds};
use garde::{Unvalidated, Validate};
use serde::Deserialize;
use std::{fs::File, io::BufReader, path::Path};

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("Failed to open config file '{path}': {source}")]
    FileOpen {
        path: String,
        source: std::io::Error,
    },

    #[error("Failed to parse YAML config: {source}")]
    YamlParse { source: serde_yaml::Error },

    #[error("Config validation failed: {source}")]
    Validation { source: garde::Report },
}

#[derive(Debug, Clone, Deserialize, Validate)]
pub struct Config {
    #[garde(dive)]
    pub grid: GridConfig,
    #[garde(dive)]
    pub octree: OctreeConfig,
    #[garde(dive)]
    pub io: IoConfig,
}

impl Config {
    /// Load configuration from a YAML file with proper error handling
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the YAML configuration file
    ///
    /// # Returns
    ///
    /// * `Ok(Config)` - Successfully loaded and validated configuration
    /// * `Err(ConfigError)` - Error during file loading, parsing, or validation
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::path::Path;
    ///
    /// let config = Config::from_file(Path::new("config.yaml"))?;
    /// ```
    pub fn from_file(path: &Path) -> Result<Config, ConfigError> {
        // Open file with proper error handling
        let file = File::open(path).map_err(|source| ConfigError::FileOpen {
            path: path.to_string_lossy().to_string(),
            source,
        })?;

        let reader = BufReader::new(file);

        // Parse YAML with proper error handling
        let raw_config: Config =
            serde_yaml::from_reader(reader).map_err(|source| ConfigError::YamlParse { source })?;

        // Validate configuration with proper error handling
        let valid_config = Unvalidated::new(raw_config)
            .validate()
            .map_err(|source| ConfigError::Validation { source })?;

        Ok(valid_config.into_inner())
    }
}

/// Configuration for grid parameters
///
/// This structure contains parameters that define the spatial domain
/// and block size for the octree grid.
#[derive(Debug, Clone, Deserialize, Validate)]
pub struct GridConfig {
    /// Spatial domain bounding box
    #[garde(custom(validate_domain))]
    pub domain: BBox,
    /// Block size at the root level
    #[garde(custom(validate_blocksize))]
    pub blocksize: IndexBounds,
}

/// Configuration for octree parameters
///
/// This structure contains parameters that control octree construction
/// and refinement behavior.
#[derive(Debug, Clone, Deserialize, Validate)]
pub struct OctreeConfig {
    /// Refinement configuration parameters
    #[garde(dive)]
    pub refinement: RefinementConfig,
}

/// Types of split strategies for octree refinement
///
/// This enum defines the available strategies for determining which nodes
/// should be split during octree construction.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SplitStrategyType {
    /// Default strategy based on depth and face intersections
    Default,
}

/// Configuration for octree refinement parameters
///
/// This structure contains parameters that control how the octree is refined,
/// including the split strategy, depth limits, and refinement criteria.
#[derive(Debug, Clone, Deserialize, Validate)]
pub struct RefinementConfig {
    /// Strategy for determining which nodes to split
    #[garde(skip)]
    pub strategy: SplitStrategyType,
    /// Maximum depth allowed in the octree
    #[garde(range(min = 0, max = MAX_OCTREE_DEPTH))]
    pub depth_limit: usize,
    /// Refinement parameter for split criteria
    #[garde(range(min = 0.3, max = 0.5))]
    pub alpha: f64,
}

/// Configuration for I/O operations
///
/// This structure contains parameters that control input/output operations,
/// including output directory and data filtering options.
#[derive(Debug, Clone, Deserialize, Validate)]
pub struct IoConfig {
    /// Directory path for output files
    #[garde(ascii)]
    pub output_dir: String,
    /// Whether to only output leaf nodes
    #[garde(skip)]
    pub only_leaves: bool,
}

fn validate_domain<T>(domain: &BBox, _: &T) -> garde::Result {
    let lower = domain.lower();
    let upper = domain.upper();
    if lower[0] >= upper[0] || lower[1] >= upper[1] || lower[2] >= upper[2] {
        return Err(garde::Error::new(format!(
            "lower must be less than upper(got lower {lower:?}, upper {upper:?})"
        )));
    }
    Ok(())
}

fn validate_blocksize<T>(blocksize: &IndexBounds, _: &T) -> garde::Result {
    if blocksize.iter().all(|&x| (1..MAX_BLOCK_AXIS).contains(&x)) {
        Ok(())
    } else {
        Err(garde::Error::new(format!(
            "each element of blocksize must be between 1 and {MAX_BLOCK_AXIS} (got {blocksize:?})"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_file() {
        let filepath = Path::new("../../tests/data/yaml/bunny.yaml");
        Config::from_file(filepath).unwrap();
    }

    #[test]
    fn test_config_from_nonexistent_file() {
        let filepath = Path::new("nonexistent.yaml");
        let result = Config::from_file(filepath);

        if let Err(ConfigError::FileOpen { path, .. }) = result {
            assert!(path.contains("nonexistent.yaml"));
        }
    }

    #[test]
    fn test_config_from_invalid_yaml() {
        // Create a temporary invalid YAML file
        let filepath = Path::new("../../tests/data/yaml/parse_error.yaml");
        let result = Config::from_file(filepath);
        assert!(matches!(result, Err(ConfigError::YamlParse { .. })));
    }

    #[test]
    fn test_config_validation_error() {
        let filepath = Path::new("../../tests/data/yaml/validation_error.yaml");
        let result = Config::from_file(filepath);
        assert!(matches!(result, Err(ConfigError::Validation { .. })));
    }
}
