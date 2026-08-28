#[cfg(not(feature = "cuda"))]
use anyhow::bail;
use anyhow::{Context, Result};
use syrinx_server::app::build_router;
use syrinx_server::asr::AsrBackend;
use syrinx_server::asr::lifecycle::{ModelHandle, NvidiaSmiProbe, VramGuard, VramProbe};
use syrinx_server::asr::mock::MockBackend;
use syrinx_server::config::{Config, Provider};
use syrinx_server::diarize::DiarizerFactory;
#[cfg(feature = "diarize")]
use syrinx_server::diarize::real::RealDiarizerFactory;
use std::sync::Arc;
use std::time::Duration;
use tracing::info;

/// How often the reaper checks whether the model has gone idle.
const REAPER_TICK: Duration = Duration::from_secs(30);

fn build_loader(
    config: &Config,
) -> Arc<dyn Fn() -> Result<Arc<dyn AsrBackend>> + Send + Sync> {
    #[cfg_attr(not(feature = "cuda"), allow(unused_variables))]
    let dir = config.model_dir.clone();
    let provider = config.provider;

    Arc::new(move || -> Result<Arc<dyn AsrBackend>> {
        match provider {
            Provider::Mock => Ok(Arc::new(MockBackend::new(&["mock"])) as Arc<dyn AsrBackend>),

            #[cfg(feature = "cuda")]
            Provider::Cpu => {
                use syrinx_server::asr::parakeet::ParakeetBackend;
                Ok(Arc::new(ParakeetBackend::load_cpu(std::path::Path::new(&dir))?))
            }

            #[cfg(feature = "cuda")]
            Provider::Cuda => {
                use syrinx_server::asr::parakeet::ParakeetBackend;
                let b = ParakeetBackend::load_cuda(std::path::Path::new(&dir))?;
                // Refuse to serve at CPU speed while claiming to use the GPU.
                // ORT registers the CUDA provider without error_on_failure, so
                // this is the only thing standing between a broken provider and
                // a server that is quietly 4x slower.
                b.verify_gpu()?;
                Ok(Arc::new(b))
            }

            #[cfg(not(feature = "cuda"))]
            Provider::Cpu | Provider::Cuda => {
                bail!(
                    "provider {provider:?} requires the `cuda` cargo feature, and this \
                     binary was built without it. Cargo writes both variants to the same \
                     path, so a plain `cargo build` or `cargo test` replaces a GPU-capable \
                     binary with one that is not. Rebuild with:\n  \
                     ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so cargo build -p syrinx-server --features cuda\n  \
                     (or just `make serve`, which does this for you). \
                     To run without a GPU instead, set provider = \"cpu\" in the config."
                )
            }
        }
    })
}

/// Load the diarization models, or explain in the log why speaker labels will
/// not be available.
///
/// Never fatal, by design: a dictation server that will not boot over an
/// optional feature's model file is the wrong trade (spec, "Error handling").
/// Every path that returns `None` says so at `error!`, because the confusing
/// state is a feature that was configured and then quietly did nothing --
/// `session.ready` will answer `diarize: false` and the client will render the
/// transcript unlabelled, with nothing anywhere to say why.
#[cfg(feature = "diarize")]
fn build_diarizer(config: &Config) -> Option<Arc<dyn DiarizerFactory>> {
    let dir = config.diarize_model_dir.as_ref()?;
    let tuning = syrinx_server::diarize::DiarizeTuning {
        min_pool: config.diarize_min_pool,
        margin: config.diarize_margin,
        mint_ceiling: config.diarize_mint_ceiling,
        change_threshold: config.diarize_change_threshold,
    };
    match RealDiarizerFactory::load(std::path::Path::new(dir), tuning) {
        Ok(factory) => Some(Arc::new(factory) as Arc<dyn DiarizerFactory>),
        Err(e) => {
            tracing::error!("speaker labelling disabled: {e:#}");
            None
        }
    }
}

/// The same decision in a binary built without the feature: there is nothing
/// to load, and the only thing worth doing is saying so when the configuration
/// expected otherwise. Cargo writes both builds to the same path, so a plain
/// `cargo build` replacing a diarize-capable binary is a real way to arrive
/// here.
#[cfg(not(feature = "diarize"))]
fn build_diarizer(config: &Config) -> Option<Arc<dyn DiarizerFactory>> {
    if config.diarize_model_dir.is_some() {
        tracing::error!(
            "diarize_model_dir is set, but this binary was built without the `diarize` \
             cargo feature, so speaker labelling is not compiled in and every session \
             will be answered diarize: false. Rebuild with:\n  \
             ORT_DYLIB_PATH=/usr/lib/libonnxruntime.so cargo build -p syrinx-server --features diarize"
        );
    }
    None
}

/// Ask the running service whether it is listening.
///
/// Hand-rolled rather than pulling in an HTTP client: one GET against
/// loopback, and a dependency added for a health probe is a dependency in the
/// image forever.
async fn health_probe(bind: &str) -> Result<()> {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    // A service bound to 0.0.0.0 is reached at loopback from inside its own
    // container; connecting to 0.0.0.0 is not portable.
    let port = bind
        .rsplit_once(':')
        .map(|(_, p)| p)
        .unwrap_or("8770")
        .to_string();
    let addr = format!("127.0.0.1:{port}");

    let mut stream = tokio::time::timeout(
        Duration::from_secs(3),
        tokio::net::TcpStream::connect(&addr),
    )
    .await
    .with_context(|| format!("timed out connecting to {addr}"))?
    .with_context(|| format!("connecting to {addr}"))?;

    stream
        .write_all(b"GET /health HTTP/1.0\r\nHost: localhost\r\n\r\n")
        .await
        .context("sending the health request")?;

    let mut buf = Vec::new();
    tokio::time::timeout(Duration::from_secs(3), stream.read_to_end(&mut buf))
        .await
        .context("timed out reading the health response")?
        .context("reading the health response")?;

    let head = String::from_utf8_lossy(&buf);
    if head.starts_with("HTTP/1.1 200") || head.starts_with("HTTP/1.0 200") {
        Ok(())
    } else {
        anyhow::bail!("unexpected response: {}", head.lines().next().unwrap_or(""))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "syrinx_server=info,ort::ep=info".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let healthcheck = args.iter().any(|a| a == "--healthcheck");
    let path = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "config.toml".into());
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading config from {path}"))?;
    let config = Arc::new(Config::from_toml_and_env(&text)?);

    // The container carries no curl or wget, so the health probe is the binary
    // itself. Shipping a fetch tool purely to ask a question the service can
    // answer about itself would be a strange thing to add to an image.
    if healthcheck {
        return match health_probe(&config.bind).await {
            Ok(()) => Ok(()),
            Err(e) => {
                eprintln!("unhealthy: {e:#}");
                std::process::exit(1);
            }
        };
    }

    // Only consult the GPU when we intend to use it. On CPU there is no shared
    // VRAM to protect, and probing would just invite a spurious refusal.
    let probe: Arc<dyn VramProbe> = match config.provider {
        Provider::Cuda => Arc::new(NvidiaSmiProbe),
        _ => Arc::new(syrinx_server::asr::lifecycle::FixedVramProbe(None)),
    };

    let model = Arc::new(ModelHandle::new(
        build_loader(&config),
        VramGuard::new(config.vram_floor_mib),
        probe,
        config.model_mib,
        Duration::from_secs(config.idle_unload_secs),
    ));

    // Nothing is loaded here on purpose: the server holds zero VRAM until the
    // first session arrives.
    info!(
        provider = ?config.provider,
        idle_unload_secs = config.idle_unload_secs,
        max_sessions = config.max_sessions,
        "starting; model will load on first session"
    );

    tokio::spawn(model.clone().run_idle_reaper(REAPER_TICK));

    // Diarization models load here rather than on first session: they are
    // small, they are checked at load, and a startup that has already proved
    // the feature works is what lets `session.ready` promise it honestly.
    let app = build_router(model, config.clone(), build_diarizer(&config));
    let listener = tokio::net::TcpListener::bind(&config.bind)
        .await
        .with_context(|| format!("binding {}", config.bind))?;
    info!("listening on {}", config.bind);
    axum::serve(listener, app).await?;
    Ok(())
}
