//! Igris — qlaster integration shim for the Agave validator.
//!
//! Wraps a `qlaster::sender::QlasterSender` behind a stable Agave-side API.
//! Construction is two-phase:
//!
//!   1. `Igris::new()`         — cheap, no runtime, used unconditionally so
//!                               that downstream wiring can hold an `Arc<Igris>`
//!                               without caring whether qlaster is enabled.
//!   2. `igris.init_qlaster()` — spawns a dedicated 2-worker tokio runtime,
//!                               builds the broadcast channels, and runs the
//!                               qlaster sender + metrics tasks. Called only
//!                               when the operator passes `--qlaster-enable`.
//!
//! Hot-path notify methods (`notify_account_update`, `notify_transaction_update`)
//! are no-ops when qlaster is not initialised, and never block: they push to a
//! `tokio::broadcast` channel and ignore `SendError` (no subscribers / shutdown).

use std::{
    net::IpAddr,
    path::PathBuf,
    sync::{Arc, RwLock},
    time::Duration,
};

use bytes::Bytes;
use serde::{Deserialize, Serialize};
use solana_account::{AccountSharedData, ReadableAccount};
use solana_clock::Slot;
use solana_metrics::datapoint_info;
use solana_pubkey::Pubkey;
use solana_signature::Signature;
use solana_transaction::versioned::VersionedTransaction;
use solana_transaction_status_client_types::TransactionStatusMeta;
use tokio::{
    runtime::Runtime,
    sync::broadcast,
    task::JoinHandle,
};

use qlaster::{
    metrics::QlasterSenderMetrics,
    sender::{
        SenderConfig, ShmTransportConfig, setup_sender_with_transactions,
    },
    types::{AccountUpdate, TransactionUpdate},
};

/// Operator-supplied configuration for the qlaster sender.
#[derive(Clone, Debug)]
pub struct QlasterConfig {
    pub bind_ip: IpAddr,
    pub uds_path: PathBuf,
    pub account_channel_capacity: usize,
    pub enable_transactions: bool,
    pub txn_channel_capacity: usize,
}

/// An account write observed by the validator and forwarded to qlaster.
///
/// The struct holds the strongly-typed Agave-side values; conversion to
/// qlaster wire types (`AccountUpdate`) happens inside `notify_account_update`,
/// because qlaster pulls a different major version of `solana-pubkey` from
/// crates.io and the two `Pubkey` types are nominally distinct.
#[derive(Clone, Debug)]
pub struct IgrisAccount {
    pub pubkey: Pubkey,
    pub account: AccountSharedData,
    pub slot: Slot,
    pub write_version: u64,
}

/// A committed transaction observed by the validator and forwarded to qlaster.
///
/// Payload is opaque to qlaster; consumers `bincode::deserialize` it back to
/// `(VersionedTransaction, TransactionStatusMeta)`.
#[derive(Clone, Debug)]
pub struct IgrisTransaction {
    pub slot: Slot,
    pub index: u32,
    pub signature: Signature,
    pub is_vote: bool,
    pub payload: Bytes,
}

/// Wire-friendly subset of `TransactionStatusMeta`. Full `TransactionStatusMeta`
/// is not `Serialize`, and its UI counterpart is JSON-shaped; this struct ships
/// the high-value execution-result fields that consumers care about most.
/// v2 can promote to `UiTransactionStatusMeta` if richer detail is needed.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IgrisTxMeta {
    pub success: bool,
    pub fee: u64,
    pub compute_units_consumed: Option<u64>,
    pub log_messages: Option<Vec<String>>,
    pub pre_balances: Vec<u64>,
    pub post_balances: Vec<u64>,
}

impl IgrisTxMeta {
    pub fn from_status_meta(m: &TransactionStatusMeta) -> Self {
        Self {
            success: m.status.is_ok(),
            fee: m.fee,
            compute_units_consumed: m.compute_units_consumed,
            log_messages: m.log_messages.clone(),
            pre_balances: m.pre_balances.clone(),
            post_balances: m.post_balances.clone(),
        }
    }
}

/// Helper to build the bincode payload at the call site, so the hot path in
/// `TransactionStatusService` does not have to know the wire format.
///
/// Wire format: `bincode::serialize(&(VersionedTransaction, IgrisTxMeta))`.
pub fn encode_transaction_payload(
    tx: &VersionedTransaction,
    meta: &TransactionStatusMeta,
) -> Bytes {
    let payload = (tx, IgrisTxMeta::from_status_meta(meta));
    let bytes = bincode::serialize(&payload)
        .expect("VersionedTransaction + IgrisTxMeta must serialize");
    Bytes::from(bytes)
}

struct IgrisInner {
    // Owns the tokio runtime; dropping it aborts the spawned tasks below.
    #[allow(dead_code)]
    runtime: Runtime,
    account_tx: broadcast::Sender<AccountUpdate>,
    txn_tx: Option<broadcast::Sender<TransactionUpdate>>,
    // Kept alive for the sender task; not read directly today, exposed via
    // `metrics()` in a future revision if needed.
    #[allow(dead_code)]
    metrics: Arc<QlasterSenderMetrics>,
    _sender_task: JoinHandle<()>,
    _metrics_task: JoinHandle<()>,
}

/// The public Agave-side handle.
///
/// Hold this as `Arc<Igris>` so the same instance can be shared between
/// `AccountsDb` (account writes) and `TransactionStatusService`
/// (committed transactions).
pub struct Igris {
    inner: RwLock<Option<IgrisInner>>,
}

impl std::fmt::Debug for Igris {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Igris")
            .field("enabled", &self.enabled())
            .finish()
    }
}

impl Igris {
    /// Cheap constructor. Does not spawn a runtime or open any sockets.
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            inner: RwLock::new(None),
        })
    }

    /// Bring up the qlaster sender. Idempotent: re-initialising returns Ok.
    pub fn init_qlaster(&self, cfg: QlasterConfig) -> anyhow::Result<()> {
        let mut guard = self.inner.write().unwrap();
        if guard.is_some() {
            log::warn!("igris: init_qlaster called when already initialised; ignoring");
            return Ok(());
        }

        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("solIgrisRt")
            .enable_all()
            .build()?;

        let metrics = Arc::new(QlasterSenderMetrics::default());

        let (account_tx, _account_rx) =
            broadcast::channel::<AccountUpdate>(cfg.account_channel_capacity);
        let txn_tx = cfg.enable_transactions.then(|| {
            let (tx, _rx) = broadcast::channel::<TransactionUpdate>(cfg.txn_channel_capacity);
            tx
        });

        let sender_cfg = SenderConfig {
            shm: ShmTransportConfig::defaults(cfg.uds_path.clone()),
        };

        let sender_metrics = metrics.clone();
        let account_tx_for_sender = account_tx.clone();
        let txn_tx_for_sender = txn_tx.clone();

        let sender_task = runtime.spawn(async move {
            match setup_sender_with_transactions(
                sender_cfg,
                account_tx_for_sender,
                txn_tx_for_sender,
                None, // bloom_updates_tx: not used in v1, qlaster filters per-connection
                sender_metrics,
            )
            .await
            {
                Ok(sender) => {
                    if let Err(err) = sender.run().await {
                        log::error!("igris: qlaster sender exited: {err}");
                    }
                }
                Err(err) => {
                    log::error!("igris: qlaster sender setup failed: {err}");
                }
            }
        });

        let metrics_for_task = metrics.clone();
        let metrics_task = runtime.spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));
            loop {
                interval.tick().await;
                let dispatch = metrics_for_task.dispatch.flush();
                let shm_send = metrics_for_task.shm_send.flush();
                let ring_full = metrics_for_task.shm_ring_full.flush();
                datapoint_info!(
                    "igris-sender",
                    ("dispatch_count", dispatch.count as i64, i64),
                    ("dispatch_avg_us", dispatch.avg_us() as i64, i64),
                    ("dispatch_max_us", dispatch.max_us as i64, i64),
                    ("shm_send_count", shm_send.count as i64, i64),
                    ("shm_send_avg_us", shm_send.avg_us() as i64, i64),
                    ("shm_send_max_us", shm_send.max_us as i64, i64),
                    ("shm_ring_full", ring_full as i64, i64),
                );
            }
        });

        let _ = cfg.bind_ip; // reserved for future use (network transports)

        *guard = Some(IgrisInner {
            runtime,
            account_tx,
            txn_tx,
            metrics,
            _sender_task: sender_task,
            _metrics_task: metrics_task,
        });

        log::info!(
            "igris: qlaster sender initialised (uds={:?}, transactions={})",
            cfg.uds_path,
            cfg.enable_transactions,
        );
        Ok(())
    }

    pub fn enabled(&self) -> bool {
        self.inner.read().unwrap().is_some()
    }

    /// Cheap pre-gate. v1 always returns `enabled()`; qlaster filters per-connection
    /// on the receive side. v2 will maintain a union bloom and return false when
    /// no subscriber would match `pubkey` or `owner`.
    pub fn need_notify(&self, _pubkey: &Pubkey, _owner: &Pubkey) -> bool {
        self.enabled()
    }

    /// Forward an account write to qlaster. Never blocks; returns immediately
    /// on shutdown / no-subscribers.
    pub fn notify_account_update(&self, a: IgrisAccount) {
        let guard = self.inner.read().unwrap();
        let Some(inner) = guard.as_ref() else {
            return;
        };

        let payload = Bytes::copy_from_slice(a.account.data());
        let pubkey = pubkey_to_qlaster(&a.pubkey);
        let owner = pubkey_to_qlaster(a.account.owner());

        let update = match AccountUpdate::new(
            pubkey,
            owner,
            a.account.lamports(),
            a.account.executable(),
            a.account.rent_epoch(),
            payload,
        ) {
            Ok(u) => u.with_slot_write_version(a.slot, a.write_version),
            Err(err) => {
                log::warn!("igris: AccountUpdate::new rejected: {err}");
                return;
            }
        };

        // Drop on SendError (no receivers — initialisation race or shutdown).
        let _ = inner.account_tx.send(update);
    }

    /// Forward a committed transaction to qlaster. No-op if transactions are
    /// not enabled in the config.
    pub fn notify_transaction_update(&self, t: IgrisTransaction) {
        let guard = self.inner.read().unwrap();
        let Some(inner) = guard.as_ref() else {
            return;
        };
        let Some(txn_tx) = inner.txn_tx.as_ref() else {
            return;
        };

        let sig_bytes: [u8; 64] = match t.signature.as_ref().try_into() {
            Ok(arr) => arr,
            Err(_) => {
                log::warn!("igris: signature was not 64 bytes; dropping");
                return;
            }
        };

        let update = match TransactionUpdate::new(
            t.slot,
            t.index,
            sig_bytes,
            t.is_vote,
            t.payload,
        ) {
            Ok(u) => u,
            Err(err) => {
                log::warn!("igris: TransactionUpdate::new rejected: {err}");
                return;
            }
        };

        let _ = txn_tx.send(update);
    }

    /// Shut down the runtime and abort all qlaster tasks. Safe to call multiple
    /// times.
    pub fn shutdown(&self) {
        let inner = {
            let mut guard = self.inner.write().unwrap();
            guard.take()
        };
        if let Some(inner) = inner {
            log::info!("igris: shutting down qlaster sender");
            // Dropping the runtime aborts spawned tasks and joins worker threads.
            drop(inner);
        }
    }
}

/// Convert workspace `solana-pubkey::Pubkey` (v4) to qlaster's
/// `solana_pubkey::Pubkey` (v3) via the on-wire 32-byte representation.
///
/// The two crate versions are nominally distinct types from cargo's view even
/// though the on-wire representation is identical. We bridge at the qlaster
/// API boundary so the rest of Agave continues to use the workspace pubkey.
fn pubkey_to_qlaster(pk: &Pubkey) -> qlaster_pubkey::Pubkey {
    qlaster_pubkey::Pubkey::new_from_array(pk.to_bytes())
}

// Re-export the qlaster types we ship through our public API, so callers do
// not have to depend on the qlaster crate directly.
pub use qlaster::error::QlasterError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn igris_disabled_is_noop() {
        let igris = Igris::new();
        assert!(!igris.enabled());
        assert!(!igris.need_notify(&Pubkey::default(), &Pubkey::default()));
        // notify_* on a non-initialised Igris must not panic.
        igris.notify_account_update(IgrisAccount {
            pubkey: Pubkey::default(),
            account: AccountSharedData::default(),
            slot: 0,
            write_version: 0,
        });
        igris.notify_transaction_update(IgrisTransaction {
            slot: 0,
            index: 0,
            signature: Signature::default(),
            is_vote: false,
            payload: Bytes::new(),
        });
        igris.shutdown();
    }
}
