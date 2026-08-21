use anyhow::{Context, Result};
use parakeet_server::app::build_router;
use parakeet_server::asr::AsrBackend;
use parakeet_server::asr::mock::MockBackend;
use parakeet_server::config::Config;
use std::sync::Arc;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "parakeet_server=info".into()),
        )
        .init();

    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "config.toml".into());
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading config from {path}"))?;
    let config = Arc::new(Config::from_toml(&text)?);

    // TODO(task 10): select the real parakeet-rs backend. The mock keeps the
    // binary runnable end-to-end before the GPU backend lands.
    let backend: Arc<dyn AsrBackend> = Arc::new(MockBackend::new(&["placeholder"]));

    let app = build_router(backend, config.clone());
    let listener = tokio::net::TcpListener::bind(&config.bind)
        .await
        .with_context(|| format!("binding {}", config.bind))?;
    info!("listening on {}", config.bind);
    axum::serve(listener, app).await?;
    Ok(())
}
