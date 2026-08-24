//! Speaker-embedding models from the sherpa-onnx zoo, behind one interface.
//!
//! The three candidates disagree about feature layout and normalisation, and
//! nothing in the ONNX file says which is which — the layout is readable from
//! the input shape, but the normalisation is a property of the training
//! recipe and has to be tabulated by name.

use anyhow::{Result, bail};
use ort::{session::Session, value::Tensor, value::ValueType};

use crate::fbank::{Fbank, NUM_BINS, Norm};

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
    pub fn new(path: &str, threads: usize) -> Result<Self> {
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

        let name = path.rsplit('/').next().unwrap_or(path);
        let norm = if name.contains("nemo") || name.contains("titanet") {
            Norm::MeanVar // NeMo's preprocessor normalises per feature
        } else {
            Norm::Mean // WeSpeaker and 3D-Speaker apply plain CMN
        };

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

        Ok((0..batch)
            .map(|b| l2_normalize(&data[b * dim..(b + 1) * dim]))
            .collect())
    }
}

/// `SessionBuilder`'s error type carries the builder itself, which is neither
/// `Send` nor `Sync`, so it cannot cross into `anyhow` without this hop
/// through `ort`'s own erased error.
pub fn session(path: &str, threads: usize) -> Result<Session> {
    let mut builder = Session::builder()?
        .with_intra_threads(threads)
        .map_err(<ort::Error>::from)?;
    Ok(builder.commit_from_file(path)?)
}

pub fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    v.iter().map(|x| x / norm).collect()
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
