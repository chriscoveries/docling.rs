//! Collision-free ORT profiling prefixes, without loading a model or runtime.
//!
//! ORT appends a timestamp but can truncate an existing trace when two sessions
//! share a prefix. Reserve a run directory atomically, then number its sessions.

use std::collections::HashMap;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Default)]
struct Runs {
    directories: HashMap<PathBuf, Run>,
}

struct Run {
    directory: PathBuf,
    sequence: u64,
}

/// An unset/blank environment value reaches here as `None`: no initialization,
/// lock or filesystem work. An unwritable profiling destination must not turn
/// optional diagnostics into a conversion failure, or reuse an unsafe prefix.
pub(super) fn prefix(dir: Option<&Path>, model_path: &Path) -> Option<PathBuf> {
    static RUNS: OnceLock<Mutex<Runs>> = OnceLock::new();
    match prefix_with(&RUNS, dir, model_path) {
        Ok(prefix) => prefix,
        Err(error) => {
            write_warning(io::stderr().lock(), dir, &error);
            None
        }
    }
}

fn write_warning(mut output: impl Write, dir: Option<&Path>, error: &io::Error) {
    // Even a closed diagnostic stream must not turn optional profiling into a
    // conversion failure (`eprintln!` would panic on a failed stderr write).
    let _ = writeln!(
        output,
        "docling-onnx: DOCLING_RS_ORT_PROFILE: could not reserve a unique profile \
         in {} ({error}); profiling disabled for this session",
        dir.unwrap_or_else(|| Path::new("")).display()
    );
}

fn prefix_with(
    runs: &OnceLock<Mutex<Runs>>,
    dir: Option<&Path>,
    model_path: &Path,
) -> io::Result<Option<PathBuf>> {
    let Some(dir) = dir else {
        return Ok(None);
    };
    let mut runs = runs
        .get_or_init(|| Mutex::new(Runs::default()))
        .lock()
        .map_err(|_| io::Error::other("profiling namespace lock poisoned"))?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    runs.prefix(dir, model_path, std::process::id(), nonce)
        .map(Some)
}

impl Runs {
    fn prefix(
        &mut self,
        dir: &Path,
        model_path: &Path,
        pid: u32,
        nonce: u128,
    ) -> io::Result<PathBuf> {
        // Canonicalize so relative paths and aliases cannot accidentally reuse a
        // cached namespace in a different destination after a CWD change.
        std::fs::create_dir_all(dir)?;
        let dir = std::fs::canonicalize(dir)?;
        let run = match self.directories.entry(dir.clone()) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => entry.insert(Run {
                directory: reserve_directory(&dir, &format!("{pid}-{nonce}"))?,
                sequence: 0,
            }),
        };
        run.sequence = run
            .sequence
            .checked_add(1)
            .ok_or_else(|| io::Error::other("profiling session sequence exhausted"))?;
        let stem = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("session");
        Ok(run.directory.join(format!("{stem}-s{}", run.sequence)))
    }
}

// Never remove a reservation: ORT may write on session drop, and retaining
// the directory keeps future processes from overwriting completed traces.
fn reserve_directory(parent: &Path, identity: &str) -> io::Result<PathBuf> {
    let mut attempt = 0u64;
    loop {
        let directory = parent.join(format!("run-{identity}-{attempt}"));
        match std::fs::create_dir(&directory) {
            Ok(()) => return Ok(directory),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                // PID/time are hints, not the uniqueness guarantee: processes
                // in separate containers can share both, as can restarted runs.
                attempt = attempt
                    .checked_add(1)
                    .ok_or_else(|| io::Error::other("profiling run sequence exhausted"))?;
            }
            Err(error) => return Err(error),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use std::sync::Barrier;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            Self(
                reserve_directory(
                    &std::env::temp_dir(),
                    &format!("docling-profile-test-{}", std::process::id()),
                )
                .unwrap(),
            )
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    fn parallel_same_model_sessions_have_distinct_prefixes() {
        let dir = TestDir::new();
        let runs = OnceLock::new();
        let barrier = Barrier::new(32);
        let prefixes = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..32)
                .map(|_| {
                    scope.spawn(|| {
                        barrier.wait();
                        prefix_with(&runs, Some(&dir.0), Path::new("recognizer.onnx"))
                            .unwrap()
                            .unwrap()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(prefixes.iter().collect::<HashSet<_>>().len(), 32);
        assert_eq!(
            prefixes
                .iter()
                .map(|p| p.parent())
                .collect::<HashSet<_>>()
                .len(),
            1
        );
        for prefix in prefixes {
            assert!(prefix
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("recognizer-s"));
        }
    }

    #[test]
    fn identical_stems_in_different_model_paths_do_not_collide() {
        let dir = TestDir::new();
        let mut runs = Runs::default();
        let a = runs.prefix(&dir.0, Path::new("a/model.onnx"), 1, 2).unwrap();
        let b = runs.prefix(&dir.0, Path::new("b/model.onnx"), 1, 2).unwrap();
        assert_eq!(a.parent(), b.parent());
        assert_eq!(a.file_name().unwrap(), "model-s1");
        assert_eq!(b.file_name().unwrap(), "model-s2");
        assert_ne!(a, b);
    }

    #[test]
    fn independent_runs_with_identical_pid_and_nonce_reserve_distinct_directories() {
        let dir = TestDir::new();
        let barrier = Barrier::new(16);
        let prefixes = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..16)
                .map(|_| {
                    scope.spawn(|| {
                        // Independent registries simulate separate processes;
                        // only the filesystem can arbitrate this forced race.
                        let mut runs = Runs::default();
                        barrier.wait();
                        runs.prefix(&dir.0, Path::new("model.onnx"), 1, 2).unwrap()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert_eq!(prefixes.iter().collect::<HashSet<_>>().len(), 16);
        assert_eq!(
            prefixes
                .iter()
                .map(|p| p.parent())
                .collect::<HashSet<_>>()
                .len(),
            16
        );
    }

    #[test]
    fn restarted_run_does_not_reuse_a_retained_directory_or_file() {
        let dir = TestDir::new();
        let occupied = dir.0.join("run-1-2-0");
        std::fs::write(&occupied, "not a directory").unwrap();
        let first = Runs::default()
            .prefix(&dir.0, Path::new("model.onnx"), 1, 2)
            .unwrap();
        std::fs::write(&first, "retained profile marker").unwrap();
        let second = Runs::default()
            .prefix(&dir.0, Path::new("model.onnx"), 1, 2)
            .unwrap();
        assert_ne!(first.parent(), second.parent());
        assert_eq!(
            std::fs::read_to_string(first).unwrap(),
            "retained profile marker"
        );
        assert_eq!(std::fs::read_to_string(occupied).unwrap(), "not a directory");
    }

    #[test]
    fn separate_destinations_are_not_replaced_by_the_first_run() {
        let dir = TestDir::new();
        let a = dir.0.join("a");
        let b = dir.0.join("b");
        let mut runs = Runs::default();
        let first = runs.prefix(&a, Path::new("model.onnx"), 1, 2).unwrap();
        let second = runs.prefix(&b, Path::new("model.onnx"), 1, 2).unwrap();
        assert!(first.starts_with(std::fs::canonicalize(a).unwrap()));
        assert!(second.starts_with(std::fs::canonicalize(b).unwrap()));
        assert_ne!(first, second);
    }

    #[test]
    fn profiling_off_does_not_initialize_or_touch_the_filesystem() {
        let dir = TestDir::new();
        let runs = OnceLock::new();
        assert!(prefix_with(&runs, None, &dir.0.join("missing-model.onnx"))
            .unwrap()
            .is_none());
        assert!(runs.get().is_none());
        assert_eq!(std::fs::read_dir(&dir.0).unwrap().count(), 0);
    }

    #[test]
    fn unusable_destination_disables_optional_profiling() {
        let dir = TestDir::new();
        let file = dir.0.join("not-a-directory");
        std::fs::write(&file, "keep me").unwrap();
        assert!(prefix(Some(&file), Path::new("model.onnx")).is_none());
        assert_eq!(std::fs::read_to_string(file).unwrap(), "keep me");
    }

    #[test]
    fn diagnostic_write_failure_is_nonfatal() {
        struct FailingDiagnostic;

        impl Write for FailingDiagnostic {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                Err(io::ErrorKind::BrokenPipe.into())
            }

            fn flush(&mut self) -> io::Result<()> {
                Err(io::ErrorKind::BrokenPipe.into())
            }
        }

        write_warning(
            FailingDiagnostic,
            Some(Path::new("unwritable")),
            &io::ErrorKind::PermissionDenied.into(),
        );
    }

    #[test]
    fn session_sequence_cannot_wrap_and_reuse_a_prefix() {
        let dir = TestDir::new();
        let mut runs = Runs::default();
        runs.prefix(&dir.0, Path::new("model.onnx"), 1, 2).unwrap();
        runs.directories.values_mut().next().unwrap().sequence = u64::MAX;
        assert!(runs.prefix(&dir.0, Path::new("model.onnx"), 1, 2).is_err());
    }
}
