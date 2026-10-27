use super::*;
use crate::cleanup::testing::{on_next_trim, DropEvent};
use std::sync::Mutex;

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

// Observe cleanup after the actual core guard drops, including its ordering
// relative to the conversion's local buffers and the admission permit.
fn observe_trim(
    permits: Arc<tokio::sync::Semaphore>,
    events: Arc<Mutex<Vec<&'static str>>>,
    held: Arc<AtomicBool>,
) {
    on_next_trim(move || {
        held.store(permits.available_permits() == 0, Ordering::SeqCst);
        events.lock().unwrap().push("trim");
    });
}

#[test]
fn mcp_cleanup_on_success_preserves_response() {
    runtime().block_on(async {
        let permits = Arc::new(tokio::sync::Semaphore::new(1));
        let events = Arc::new(Mutex::new(Vec::new()));
        let held = Arc::new(AtomicBool::new(false));
        let permit = permits.clone().acquire_owned().await.unwrap();
        let (p, e, h) = (permits.clone(), events.clone(), held.clone());
        let captured = DropEvent(events.clone(), "capture");
        let response = spawn_conversion(permit, move || {
            let _captured = captured;
            observe_trim(p, e.clone(), h);
            let _buffer = DropEvent(e.clone(), "buffer");
            (
                String::from("response still live"),
                DropEvent(e, "response"),
            )
        })
        .await
        .unwrap();
        assert_eq!(response.0, "response still live");
        assert_eq!(*events.lock().unwrap(), ["buffer", "capture", "trim"]);
        assert!(held.load(Ordering::SeqCst));
        assert_eq!(permits.available_permits(), 1);
        drop(response);
        assert_eq!(
            *events.lock().unwrap(),
            ["buffer", "capture", "trim", "response"]
        );
    });
}

#[test]
fn mcp_cleanup_on_error_and_unwind() {
    runtime().block_on(async {
        for unwind in [false, true] {
            let permits = Arc::new(tokio::sync::Semaphore::new(1));
            let events = Arc::new(Mutex::new(Vec::new()));
            let held = Arc::new(AtomicBool::new(false));
            let permit = permits.clone().acquire_owned().await.unwrap();
            let (p, e, h) = (permits.clone(), events.clone(), held.clone());
            let result = spawn_conversion(permit, move || -> Result<(), &'static str> {
                observe_trim(p, e.clone(), h);
                let _buffer = DropEvent(e, "buffer");
                if unwind {
                    panic!("cleanup test unwind");
                }
                Err::<(), _>("cleanup test error")?;
                Ok(())
            })
            .await;
            if unwind {
                assert!(result.unwrap_err().is_panic());
            } else {
                assert_eq!(result.unwrap(), Err("cleanup test error"));
            }
            assert_eq!(*events.lock().unwrap(), ["buffer", "trim"]);
            assert!(held.load(Ordering::SeqCst));
            assert_eq!(permits.available_permits(), 1);
        }
    });
}

#[test]
fn mcp_cancelled_caller_keeps_permit_until_blocking_cleanup() {
    runtime().block_on(async {
        let permits = Arc::new(tokio::sync::Semaphore::new(1));
        let events = Arc::new(Mutex::new(Vec::new()));
        let held = Arc::new(AtomicBool::new(false));
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (finish_tx, finish_rx) = std::sync::mpsc::channel();
        let (p, e, h) = (permits.clone(), events.clone(), held.clone());
        let caller = tokio::spawn(async move {
            let permit = p.clone().acquire_owned().await.unwrap();
            spawn_conversion(permit, move || {
                observe_trim(p, e.clone(), h);
                let _buffer = DropEvent(e, "buffer");
                started_tx.send(()).unwrap();
                // A finite fail-safe avoids a hanging test on a regression;
                // the test releases the worker through the channel, not time.
                finish_rx.recv_timeout(Duration::from_secs(30)).unwrap();
            })
            .await
        });
        started_rx.await.unwrap();
        caller.abort();
        assert!(caller.await.unwrap_err().is_cancelled());
        assert_eq!(permits.available_permits(), 0);
        assert!(events.lock().unwrap().is_empty());

        // A different caller cancelled while queued must not enter work or
        // affect the first caller's still-running blocking job.
        let entered = Arc::new(AtomicBool::new(false));
        let (p, entered_work) = (permits.clone(), entered.clone());
        let (queued_tx, queued_rx) = tokio::sync::oneshot::channel();
        let waiter = tokio::spawn(async move {
            queued_tx.send(()).unwrap();
            let permit = p.acquire_owned().await.unwrap();
            spawn_conversion(permit, move || entered_work.store(true, Ordering::SeqCst)).await
        });
        queued_rx.await.unwrap();
        waiter.abort();
        assert!(waiter.await.unwrap_err().is_cancelled());
        assert!(!entered.load(Ordering::SeqCst));
        assert_eq!(permits.available_permits(), 0);

        finish_tx.send(()).unwrap();
        let acquire = permits.clone().acquire_owned();
        let released = tokio::time::timeout(Duration::from_secs(30), acquire)
            .await
            .expect("blocking work must release its permit")
            .unwrap();
        assert_eq!(*events.lock().unwrap(), ["buffer", "trim"]);
        assert!(held.load(Ordering::SeqCst));
        assert!(!entered.load(Ordering::SeqCst));
        drop(released);
        assert_eq!(permits.available_permits(), 1);
    });
}
