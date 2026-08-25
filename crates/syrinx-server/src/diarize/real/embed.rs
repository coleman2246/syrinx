//! Speaker-embedding models from the sherpa-onnx zoo, behind one interface.
//!
//! The three candidate model families disagree about feature layout and
//! normalisation, and nothing in the ONNX file says which is which -- the
//! layout is readable from the input shape, but the normalisation is a
//! property of the training recipe. Ported from the go/no-go spike -- deleted,
//! `git log -- 'spike/diarize'` -- with one change its review required:
//! normalisation used to be guessed from the model's filename there; here it
//! is an explicit constructor argument. The filename convention survives, but
//! in exactly one place -- `diarizer::norm_for`, which resolves a model
//! directory once at startup and refuses names it does not recognise. There is deliberately no
//! guessing constructor on this type: a second copy of that mapping is a
//! second answer to "which recipe is this?", and the wrong answer produces
//! embeddings that separate nobody without failing anywhere.

use anyhow::{Result, anyhow, bail};
use ort::{session::Session, value::Tensor, value::ValueType};

use super::session;
use crate::diarize::cluster::l2_normalize;
use crate::diarize::fbank::{Fbank, NUM_BINS, Norm};

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
    dim: usize,
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
        //
        // `shape[1] > 0` excludes a dynamic axis: ONNX reports one as -1,
        // and casting that to `usize` wraps to `usize::MAX`, which would
        // otherwise either "win" a `min_by_key` tie against every other
        // dynamic output or -- worse -- get picked as the embedding with a
        // nonsensical multi-exabyte `dim`.
        let (output, dim) = session
            .outputs()
            .iter()
            .filter_map(|o| match o.dtype() {
                ValueType::Tensor { shape, .. } if shape.len() == 2 && shape[1] > 0 => {
                    Some((o.name().to_string(), shape[1] as usize))
                }
                _ => None,
            })
            .min_by_key(|(_, dim)| *dim)
            .ok_or_else(|| anyhow!("{path}: no 2-D output to read an embedding from"))?;

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

    /// The embedding's dimensionality (192 for the production ERes2Net
    /// model), read once from the ONNX output shape at construction.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Embed a single window. The common case for [`super::RealDiarizer`],
    /// where one `push` completes at most one voiced window; a thin wrapper
    /// over [`Embedder::embed_batch`] with a batch of one.
    pub fn embed(&mut self, window: &[f32]) -> Result<Vec<f32>> {
        let mut rows = self.embed_batch(&[window])?;
        Ok(rows.pop().expect("embed_batch(1 window) returns 1 row"))
    }

    /// Embed a batch of equal-length windows, L2-normalised, one row each.
    ///
    /// Not on the session's path, which has one window at a time and no next
    /// one to batch it with: the caller is `examples/diarize_probe`, which
    /// embeds a whole meeting before scoring it and is the reason this is
    /// worth having at all.
    ///
    /// Every window must be the same length -- the fbank frame count is
    /// derived once, from the first window, and reused for the whole batch;
    /// mismatched lengths are rejected rather than silently truncated or
    /// left to panic on a slice-length mismatch downstream. A window
    /// shorter than one fbank frame ([`crate::diarize::fbank::FRAME_LEN`],
    /// 400 samples / 25 ms at 16 kHz) yields zero frames and an empty
    /// feature tensor, which the embedding model will reject -- callers
    /// should not offer the embedder anything shorter than that.
    pub fn embed_batch(&mut self, windows: &[&[f32]]) -> Result<Vec<Vec<f32>>> {
        if windows.is_empty() {
            return Ok(Vec::new());
        }
        let batch = windows.len();
        let window_len = windows[0].len();
        if let Some(bad) = windows.iter().position(|w| w.len() != window_len) {
            bail!(
                "embed_batch: window {bad} has {} samples, window 0 has {window_len}; \
                 every window in a batch must be the same length",
                windows[bad].len()
            );
        }
        let frames = Fbank::num_frames(window_len);

        let mut packed = vec![0.0f32; batch * frames * NUM_BINS];
        for (b, &w) in windows.iter().enumerate() {
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
        // `SessionOutputs`' `Index<&str>` panics on a missing name; see the
        // matching comment in `vad.rs::prob` for why a bad or
        // differently-exported model must fail this call with an `Err`
        // instead.
        let (shape, data) = out
            .get(self.output.as_str())
            .ok_or_else(|| anyhow!("embedding model has no output named \"{}\"", self.output))?
            .try_extract_tensor::<f32>()?;
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

#[cfg(test)]
mod tests {
    use super::*;

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
