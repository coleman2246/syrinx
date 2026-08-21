#[cfg(not(feature = "cuda"))]
use anyhow::bail;
use anyhow::{Context, Result};
use parakeet_server::app::build_router;
use parakeet_server::asr::AsrBackend;
use parakeet_server::asr::lifecycle::{ModelHandle, NvidiaSmiProbe, VramGuard, VramProbe};
use parakeet_server::asr::mock::MockBackend;
use parakeet_server::config::{Config, Provider};
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
                use parakeet_server::asr::parakeet::ParakeetBackend;
                Ok(Arc::new(ParakeetBackend::load_cpu(std::path::Path::new(&dir))?))
            }

            #[cfg(feature = "cuda")]
            Provider::Cuda => {
                use parakeet_server::asr::parakeet::ParakeetBackend;
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
                    "provider {provider:?} requires the `cuda` cargo feature; \
                     this binary was built without it"
                )
            }
        }
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "parakeet_server=info,ort::ep=info".into()),
        )
        .init();

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "config.toml".into());
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading config from {path}"))?;
    let config = Arc::new(Config::from_toml(&text)?);

    // Only consult the GPU when we intend to use it. On CPU there is no shared
    // VRAM to protect, and probing would just invite a spurious refusal.
    let probe: Arc<dyn VramProbe> = match config.provider {
        Provider::Cuda => Arc::new(NvidiaSmiProbe),
        _ => Arc::new(parakeet_server::asr::lifecycle::FixedVramProbe(None)),
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

    let app = build_router(model, config.clone());
    let listener = tokio::net::TcpListener::bind(&config.bind)
        .await
        .with_context(|| format!("binding {}", config.bind))?;
    info!("listening on {}", config.bind);
    axum::serve(listener, app).await?;
    Ok(())
}
