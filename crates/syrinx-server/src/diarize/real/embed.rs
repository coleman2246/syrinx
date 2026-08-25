//! Speaker-embedding models from the sherpa-onnx zoo, behind one interface.
//!
//! The three candidate model families disagree about feature layout and
//! normalisation, and nothing in the ONNX file says which is which -- the
//! layout is readable from the input shape, but the normalisation is a
//! property of the training recipe. Ported from `spike/diarize/src/embed.rs`,
//! with one change required by that spike's review: normalisation used to be
//! guessed from the model's filename there; here it is an explicit
//! constructor argument, and the filename guess survives only as a
//! documented convenience in [`Embedder::from_path`].

use anyhow::{Result, bail};
use ort::{session::Session, value::Tensor, value::ValueType};

use super::fbank::{Fbank, NUM_BINS, Norm};
use super::session;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Layout {
    /// `[batch, frames, 80]` — WeSpeaker, 3D-Speaker.
    TimeMajor,
    /// `[batch, 80, frames]` — NeMo.
    FeatMajor,
}

pub struct Embedder {
    session: Session,
    input: String,
    /// NeMo's encoder takes an explicit valid-length vector alongside the
    /// features.
    length_input: Option<String>,
    output: String,
    layout: Layout,
    norm: Norm,
    fbank: Fbank,
    pub dim: usize,
}

impl Embedder {
    /// Build an embedder with an explicit normalisation mode.
    ///
    /// `norm` is the one property the ONNX file cannot tell us: it is a fact
    /// about the training recipe (NeMo's preprocessor normalises per
    /// feature; WeSpeaker and 3D-Speaker apply plain CMN), not the model
    /// graph, so the caller states it rather than this constructor guessing
    /// it. The feature *layout* (`TimeMajor` vs `FeatMajor`), by contrast,
    /// is read from the input tensor's own declared shape below -- that one
    /// is safe to infer because it is structural, not a recipe detail.
    pub fn new(path: &str, threads: usize, norm: Norm) -> Result<Self> {
        let session = session(path, threads)?;

        let feats = &session.inputs()[0];
        let (input, layout) = match feats.dtype() {
            ValueType::Tensor { shape, .. } if shape.len() == 3 => {
                let layout = if shape[2] == NUM_BINS as i64 {
                    Layout::TimeMajor
                } else if shape[1] == NUM_BINS as i64 {
                    Layout::FeatMajor
                } else {
                    bail!("{path}: no 80-dim axis in feature input shape {shape:?}");
                };
                (feats.name().to_string(), layout)
            }
            other => bail!("{path}: unexpected feature input {other:?}"),
        };

        let length_input = session
            .inputs()
            .iter()
            .find(|i| i.name() == "length")
            .map(|i| i.name().to_string());

        // TitaNet exposes its 16k-way training classifier alongside the
        // embedding; the embedding is always the narrower output.
        let (output, dim) = session
            .outputs()
            .iter()
            .filter_map(|o| match o.dtype() {
                ValueType::Tensor { shape, .. } if shape.len() == 2 => {
                    Some((o.name().to_string(), shape[1] as usize))
                }
                _ => None,
            })
            .min_by_key(|(_, dim)| *dim)
            .ok_or_else(|| anyhow::anyhow!("{path}: no 2-D output to read an embedding from"))?;

        Ok(Self {
            session,
            input,
            length_input,
            output,
            layout,
            norm,
            fbank: Fbank::new(),
            dim,
        })
    }

    /// [`Embedder::new`], with `norm` guessed from the filename.
    ///
    /// A convenience, not the contract: it exists so a model directory laid
    /// out the way `spike/diarize` expected it (`*nemo*`/`*titanet*` for
    /// NeMo-style models, anything else for WeSpeaker/3D-Speaker) can be
    /// loaded with one call. Anything that matters -- a differently-named
    /// checkpoint, a fourth model family -- should call
    /// [`Embedder::new`] directly with the normalisation it actually needs
    /// rather than rely on this guess.
    pub fn from_path(path: &str, threads: usize) -> Result<Self> {
        Self::new(path, threads, guess_norm(path))
    }

    /// Embed a batch of equal-length windows, L2-normalised, one row each.
    pub fn embed_batch(&mut self, windows: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        if windows.is_empty() {
            return Ok(Vec::new());
        }
        let batch = windows.len();
        let frames = Fbank::num_frames(windows[0].len());

        let mut packed = vec![0.0f32; batch * frames * NUM_BINS];
        for (b, w) in windows.iter().enumerate() {
            let mut feats = self.fbank.compute(w);
            debug_assert_eq!(feats.len(), frames * NUM_BINS);
            Fbank::normalize(&mut feats, self.norm);

            let base = b * frames * NUM_BINS;
            match self.layout {
                Layout::TimeMajor => {
                    packed[base..base + frames * NUM_BINS].copy_from_slice(&feats);
                }
                Layout::FeatMajor => {
                    for f in 0..frames {
                        for d in 0..NUM_BINS {
                            packed[base + d * frames + f] = feats[f * NUM_BINS + d];
                        }
                    }
                }
            }
        }

        let shape = match self.layout {
            Layout::TimeMajor => [batch, frames, NUM_BINS],
            Layout::FeatMajor => [batch, NUM_BINS, frames],
        };
        let mut inputs = ort::inputs![self.input.clone() => Tensor::from_array((shape, packed))?];
        if let Some(name) = &self.length_input {
            let lengths = vec![frames as i64; batch];
            inputs.push((
                name.clone().into(),
                Tensor::from_array(([batch], lengths))?.into(),
            ));
        }

        let out = self.session.run(inputs)?;
        let (shape, data) = out[self.output.as_str()].try_extract_tensor::<f32>()?;
        let dim = shape[1] as usize;
        // A single non-finite value would spread through an EMA update and
        // poison that centroid for the rest of the run, silently. Better to
        // stop than to publish numbers computed against a dead centroid.
        if let Some(bad) = data.iter().position(|x| !x.is_finite()) {
            bail!(
                "embedding {} is not finite at dimension {}",
                bad / dim,
                bad % dim
            );
        }

        Ok((0..batch)
            .map(|b| l2_normalize(&data[b * dim..(b + 1) * dim]))
            .collect())
    }
}

/// The filename convention `from_path` guesses normalisation from: NeMo's
/// preprocessor normalises per feature, WeSpeaker and 3D-Speaker apply plain
/// CMN, and neither fact is visible in the ONNX graph itself.
fn guess_norm(path: &str) -> Norm {
    let name = path.rsplit('/').next().unwrap_or(path);
    if name.contains("nemo") || name.contains("titanet") {
        Norm::MeanVar
    } else {
        Norm::Mean
    }
}

/// A unit-length copy. `diarize::cluster` carries its own private copy of
/// this same function (it predates this module and is pure arithmetic with
/// no `ort` in sight, so there was no reason to disturb its already-reviewed
/// code to share one); this one exists so `embed_batch` above does not reach
/// across a feature gate to borrow it.
fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter().map(|x| x / norm).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_path_picks_meanvar_for_nemo_and_titanet_names() {
        for name in [
            "nemo_titanet.onnx",
            "titanet-large.onnx",
            "some-nemo-v2.onnx",
        ] {
            assert_eq!(guess_norm(name), Norm::MeanVar, "{name}");
        }
    }

    #[test]
    fn from_path_picks_mean_for_everything_else() {
        for name in ["wespeaker.onnx", "3dspeaker_eres2net.onnx", "model.onnx"] {
            assert_eq!(guess_norm(name), Norm::Mean, "{name}");
        }
    }

    #[test]
    fn l2_normalize_produces_a_unit_vector() {
        let v = l2_normalize(&[3.0, 4.0]);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_guards_against_a_zero_vector() {
        // The 1e-9 floor keeps this finite rather than NaN.
        let v = l2_normalize(&[0.0, 0.0]);
        assert!(v.iter().all(|x| x.is_finite()));
    }
}
