use serde::{Deserialize, Serialize};

/// Blendshape weights for avatar mouth movements.
/// Clamped strictly in the range [0.0, 1.0].
/// Matches standard VRM vowel presets: A, I, U, E, O.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VisemeWeights {
    pub aa: f32,
    pub ih: f32,
    pub ou: f32,
    pub ee: f32,
    pub oh: f32,
}

impl VisemeWeights {
    /// Creates a new `VisemeWeights` struct with all values clamped to [0.0, 1.0].
    /// Parameter order: (aa, ih, ou, ee, oh).
    pub fn new(aa: f32, ih: f32, ou: f32, ee: f32, oh: f32) -> Self {
        Self {
            aa: Self::clamp_weight(aa),
            ih: Self::clamp_weight(ih),
            ou: Self::clamp_weight(ou),
            ee: Self::clamp_weight(ee),
            oh: Self::clamp_weight(oh),
        }
    }

    /// Returns a `VisemeWeights` instance with all weights set to 0.0.
    pub fn zero() -> Self {
        Self {
            aa: 0.0,
            ih: 0.0,
            ou: 0.0,
            ee: 0.0,
            oh: 0.0,
        }
    }

    #[inline]
    fn clamp_weight(w: f32) -> f32 {
        if w.is_nan() || w < 0.0 {
            0.0
        } else if w > 1.0 {
            1.0
        } else {
            w
        }
    }
}

impl Default for VisemeWeights {
    fn default() -> Self {
        Self::zero()
    }
}
