use super::*;
use crate::cleanup::testing::{on_next_trim, DropEvent};
use std::sync::{Arc, Mutex};

#[derive(Clone, Copy)]
enum Finish {
    Success,
    Error,
    Unwind,
}

struct Backend {
    finish: Finish,
    events: Arc<Mutex<Vec<&'static str>>>,
}

impl docling::backend::DeclarativeBackend for Backend {
    fn convert(&self, _: &SourceDocument) -> Result<DoclingDocument, docling::ConversionError> {
        let _buffer = DropEvent(self.events.clone(), "buffer");
        match self.finish {
            Finish::Success => {
                let mut document = DoclingDocument::new("cleanup");
                document.add_paragraph("cleanup");
                Ok(document)
            }
            Finish::Error => Err(docling::ConversionError::Parse("cleanup test error".into())),
            Finish::Unwind => panic!("cleanup test unwind"),
        }
    }
}

struct OutputDir(PathBuf);

impl OutputDir {
    fn new() -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        loop {
            let path = std::env::temp_dir().join(format!(
                "docling-batch-cleanup-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match std::fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => panic!("creating test output: {e}"),
            }
        }
    }
}

impl Drop for OutputDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn run(finish: Finish, images_only: bool) {
    let output = OutputDir::new();
    let cfg = BatchCfg {
        to: vec![if images_only { "images" } else { "md" }.into()],
        image_mode: ImageMode::Placeholder,
        export: ExportLayers::default(),
        pandoc_image_mode: ImageMode::Placeholder,
        opts: docling::ConvertOptions::default(),
        pages: None,
        progress: false,
        scale: 1.0,
        chunk: ChunkOptions::default(),
        vlm: None,
    };
    // A control-flow payload, not a PDF, model result or quality fixture.
    let source = SourceDocument::from_bytes("cleanup", InputFormat::Text, b"cleanup".to_vec())
        .with_media_type(Some("application/x-cleanup-test".into()));
    let item = BatchItem {
        file: "cleanup.txt".into(),
        base: PathBuf::new(),
        entry: None,
        preloaded: Some(("cleanup test".into(), source)),
        merged: false,
    };
    let events = Arc::new(Mutex::new(Vec::new()));
    let converter = DocumentConverter::new().register_backend(
        "cleanup test",
        docling::BackendMatcher::new().media_types(["application/x-cleanup-test"]),
        Arc::new(Backend {
            finish,
            events: events.clone(),
        }),
        0,
    );
    let observed = events.clone();
    on_next_trim(move || observed.lock().unwrap().push("trim"));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        batch_convert_one(
            &item,
            &output.0,
            &cfg,
            &converter,
            &Mutex::new(None),
            &mut ArchiveCache::default(),
            1,
        )
    }));
    if images_only {
        let Err(error) = result.unwrap() else {
            panic!("non-PDF image export must fail");
        };
        assert!(error.contains("rasterizes PDF inputs only"), "{error}");
        assert_eq!(*events.lock().unwrap(), ["trim"]);
        return;
    }
    match finish {
        Finish::Success => {
            let outcome = result.unwrap().unwrap();
            assert_eq!(outcome.written, [output.0.join("cleanup.md")]);
            assert_eq!(
                std::fs::read_to_string(&outcome.written[0]).unwrap(),
                "cleanup\n"
            );
        }
        Finish::Error => {
            let Err(error) = result.unwrap() else {
                panic!("backend error must propagate");
            };
            assert!(error.contains("cleanup test error"), "{error}");
        }
        Finish::Unwind => assert!(result.is_err()),
    }
    assert_eq!(*events.lock().unwrap(), ["buffer", "trim"]);
}

#[test]
fn batch_cleanup_on_success() {
    run(Finish::Success, false);
}

#[test]
fn batch_cleanup_on_conversion_error() {
    run(Finish::Error, false);
}

#[test]
fn batch_cleanup_on_unwind() {
    run(Finish::Unwind, false);
}

#[test]
fn batch_cleanup_on_early_error() {
    run(Finish::Success, true);
}
