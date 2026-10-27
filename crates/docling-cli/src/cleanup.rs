//! Reuse core's allocator guard; only unit tests add a drop observer.

#[cfg(not(test))]
pub(crate) fn trim_on_drop() -> docling::memory::TrimOnDrop {
    docling::memory::TrimOnDrop
}

#[cfg(test)]
pub(crate) fn trim_on_drop() -> testing::ObservedTrim {
    testing::ObservedTrim(Some(docling::memory::TrimOnDrop))
}

#[cfg(test)]
pub(crate) mod testing {
    use std::cell::RefCell;
    use std::sync::{Arc, Mutex};

    thread_local! {
        // One-shot and thread-local: concurrent tests cannot observe each
        // other's cleanup. No allocator/RSS behavior is mocked or asserted.
        static AFTER_TRIM: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
    }

    pub(crate) struct ObservedTrim(pub(super) Option<docling::memory::TrimOnDrop>);

    impl Drop for ObservedTrim {
        fn drop(&mut self) {
            drop(self.0.take());
            let observe = AFTER_TRIM.with(|slot| slot.borrow_mut().take());
            if let Some(observe) = observe {
                observe();
            }
        }
    }

    pub(crate) fn on_next_trim(observe: impl FnOnce() + 'static) {
        AFTER_TRIM.with(|slot| {
            assert!(slot.borrow().is_none(), "previous trim was not observed");
            *slot.borrow_mut() = Some(Box::new(observe));
        });
    }

    pub(crate) struct DropEvent(
        pub(crate) Arc<Mutex<Vec<&'static str>>>,
        pub(crate) &'static str,
    );

    impl Drop for DropEvent {
        fn drop(&mut self) {
            self.0.lock().unwrap().push(self.1);
        }
    }
}
