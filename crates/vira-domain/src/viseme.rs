use serde::{Deserialize, Serialize};

/// Blendshape weights for avatar mouth movements.
/// Clamped strictly in the range [0.0, 1.0].
/// Matches standard VRM vowel presets: A, I, U, E, O.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct VisemeWeights {
    #[serde(deserialize_with = "deserialize_clamped")]
    pub aa: f32,
    #[serde(deserialize_with = "deserialize_clamped")]
    pub ih: f32,
    #[serde(deserialize_with = "deserialize_clamped")]
    pub ou: f32,
    #[serde(deserialize_with = "deserialize_clamped")]
    pub ee: f32,
    #[serde(deserialize_with = "deserialize_clamped")]
    pub oh: f32,
}

fn deserialize_clamped<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let val = f32::deserialize(deserializer)?;
    Ok(VisemeWeights::clamp_weight(val))
}

impl VisemeWeights {
    /// Creates a new `VisemeWeights` struct with all values clamped to [0.0, 1.0].
    /// Parameter order: (aa, ih, ou, ee, oh) matching standard Japanese Gojūon (A, I, U, E, O).
    pub fn new(aa: f32, ih: f32, ou: f32, ee: f32, oh: f32) -> Self {
        Self {
            aa: Self::clamp_weight(aa),
            ih: Self::clamp_weight(ih),
            ou: Self::clamp_weight(ou),
            ee: Self::clamp_weight(ee),
            oh: Self::clamp_weight(oh),
        }
    }

    /// Alias constructor using standard Japanese Gojūon vowel order: (A, I, U, E, O).
    pub fn from_vowels(a: f32, i: f32, u: f32, e: f32, o: f32) -> Self {
        Self::new(a, i, u, e, o)
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

    /// Builder modifier for `aa` weight.
    pub fn with_aa(mut self, aa: f32) -> Self {
        self.aa = Self::clamp_weight(aa);
        self
    }

    /// Builder modifier for `ih` weight.
    pub fn with_ih(mut self, ih: f32) -> Self {
        self.ih = Self::clamp_weight(ih);
        self
    }

    /// Builder modifier for `ou` weight.
    pub fn with_ou(mut self, ou: f32) -> Self {
        self.ou = Self::clamp_weight(ou);
        self
    }

    /// Builder modifier for `ee` weight.
    pub fn with_ee(mut self, ee: f32) -> Self {
        self.ee = Self::clamp_weight(ee);
        self
    }

    /// Builder modifier for `oh` weight.
    pub fn with_oh(mut self, oh: f32) -> Self {
        self.oh = Self::clamp_weight(oh);
        self
    }

    /// Returns a copy where all fields are strictly clamped to [0.0, 1.0].
    pub fn clamped(&self) -> Self {
        Self {
            aa: Self::clamp_weight(self.aa),
            ih: Self::clamp_weight(self.ih),
            ou: Self::clamp_weight(self.ou),
            ee: Self::clamp_weight(self.ee),
            oh: Self::clamp_weight(self.oh),
        }
    }

    /// Linear interpolation between current weights and target weights by factor t [0.0, 1.0].
    /// Used for smooth 60Hz lip-sync transitions.
    pub fn lerp(&self, target: &Self, factor: f32) -> Self {
        let t = Self::clamp_weight(factor);
        Self {
            aa: Self::clamp_weight(self.aa + (target.aa - self.aa) * t),
            ih: Self::clamp_weight(self.ih + (target.ih - self.ih) * t),
            ou: Self::clamp_weight(self.ou + (target.ou - self.ou) * t),
            ee: Self::clamp_weight(self.ee + (target.ee - self.ee) * t),
            oh: Self::clamp_weight(self.oh + (target.oh - self.oh) * t),
        }
    }

    /// Checks if two weights are approximately equal within a small epsilon.
    pub fn approx_eq(&self, other: &Self, epsilon: f32) -> bool {
        (self.aa - other.aa).abs() <= epsilon
            && (self.ih - other.ih).abs() <= epsilon
            && (self.ou - other.ou).abs() <= epsilon
            && (self.ee - other.ee).abs() <= epsilon
            && (self.oh - other.oh).abs() <= epsilon
    }

    /// Returns true if all viseme weights are effectively zero (mouth closed/silent).
    pub fn is_silent(&self) -> bool {
        const EPSILON: f32 = 1e-4;
        self.aa <= EPSILON
            && self.ih <= EPSILON
            && self.ou <= EPSILON
            && self.ee <= EPSILON
            && self.oh <= EPSILON
    }

    /// Converts weights to an array [aa, ih, ou, ee, oh].
    pub fn as_array(&self) -> [f32; 5] {
        [self.aa, self.ih, self.ou, self.ee, self.oh]
    }

    /// Constructs weights from an array [aa, ih, ou, ee, oh].
    pub fn from_array(weights: [f32; 5]) -> Self {
        Self::new(weights[0], weights[1], weights[2], weights[3], weights[4])
    }

    /// Returns array of name-weight pairs matching standard VRM 1.0 expression presets.
    pub fn to_named_weights(&self) -> [(&'static str, f32); 5] {
        [
            ("aa", self.aa),
            ("ih", self.ih),
            ("ou", self.ou),
            ("ee", self.ee),
            ("oh", self.oh),
        ]
    }

    #[inline]
    pub fn clamp_weight(w: f32) -> f32 {
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
