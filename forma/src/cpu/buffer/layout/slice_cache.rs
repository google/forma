// Copyright 2022 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// IMPORTANT: Upon any code-related modification to this file, please ensure that all commented-out
//            tests that start with `fails_` actually fail to compile *independently* from one
//            another.

use std::{
    fmt, hint,
    marker::PhantomData,
    mem,
    ops::{Bound, Deref, DerefMut, RangeBounds},
    ptr::NonNull,
    slice,
};

use crossbeam_utils::atomic::AtomicCell;

// `SliceCache` is virtually identical to a `Vec<Range<usize>>` whose range are statically
// guaranteed not to overlap or overflow the initial `slice` that the object has been made with.
//
// This is achieved by forcing the user to produce `Span`s in a closure provided to the constructor
// from a root `Span` that cannot escape the closure.
//
// In `SliceCache::access`, we make sure that the slice doesn't overflow the `len` passed in
// `SliceCache::new` and then save the pointer to the global `ROOT` that is then guarded until the
// `Ref` is dropped.

#[repr(transparent)]
#[derive(Clone, Copy, Eq, PartialEq)]
struct SendNonNull<T> {
    ptr: NonNull<T>,
}

unsafe impl<T> Send for SendNonNull<T> {}

impl<T> From<NonNull<T>> for SendNonNull<T> {
    fn from(ptr: NonNull<T>) -> Self {
        Self { ptr }
    }
}

static ROOT: AtomicCell<Option<SendNonNull<()>>> = AtomicCell::new(None);

/// A [`prim@slice`] wrapper produced by [`SliceCache::access`].
#[repr(C)]
pub struct Slice<'a, T> {
    offset: isize,
    len: usize,
    _phantom: PhantomData<&'a mut [T]>,
}

// Since this type is equivalent to `&mut [T]`, it also implements `Send`.
unsafe impl<'a, T: Send> Send for Slice<'a, T> {}

// Since this type is equivalent to `&mut [T]`, it also implements `Sync`.
unsafe impl<'a, T: Sync> Sync for Slice<'a, T> {}

impl<'a, T> Deref for Slice<'a, T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &'a Self::Target {
        let root: NonNull<T> = ROOT.load().unwrap().ptr.cast();

        // `Slice`s should only be dereferences when tainted with the `'s` lifetime from the
        // `SliceCache::access` method. This ensures that the slice that results from derefrencing
        // here will also be constrained by the same lifetime.
        //
        // This also expects the `ROOT` pointer to be correctly set up in `SliceCache::access`.
        unsafe { slice::from_raw_parts(root.as_ptr().offset(self.offset), self.len) }
    }
}

impl<'a, T> DerefMut for Slice<'a, T> {
    #[inline]
    fn deref_mut(&mut self) -> &'a mut Self::Target {
        let root: NonNull<T> = ROOT.load().unwrap().ptr.cast();

        // `Slice`s should only be dereferences when tainted with the `'s` lifetime from the
        // `SliceCache::access` method. This ensures that the slice that results from derefrencing
        // here will also be constrained by the same lifetime.
        //
        // This also expects the `ROOT` pointer to be correctly set up in `SliceCache::access`.
        unsafe { slice::from_raw_parts_mut(root.as_ptr().offset(self.offset), self.len) }
    }
}

impl<T: fmt::Debug> fmt::Debug for Slice<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

/// A marker produced by [`SliceCache`] that ensures that all resulting `Span`s will be mutually
/// non-overlapping.
///
/// # Examples
///
/// ```
/// # use forma_render::cpu::buffer::layout::SliceCache;
/// let _cache = SliceCache::new(4, |span| {
///     Box::new([span])
/// });
/// ```
#[repr(transparent)]
pub struct Span<'a>(Slice<'a, ()>);

impl<'a> Span<'a> {
    fn from_slice(slice: &Slice<'a, ()>) -> Self {
        Self(Slice {
            offset: slice.offset,
            len: slice.len,
            _phantom: PhantomData,
        })
    }

    /// cache span at `mid`. Analogous to [`slice::split_at`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::cpu::buffer::layout::SliceCache;
    /// let _cache = SliceCache::new(4, |span| {
    ///     let (left, right) = span.split_at(2);
    ///     Box::new([left, right])
    /// });
    /// ```
    #[inline]
    pub fn slice<R: RangeBounds<usize>>(&self, range: R) -> Option<Self> {
        let start = match range.start_bound() {
            Bound::Included(&i) => i,
            Bound::Excluded(&i) => i + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(&i) => i + 1,
            Bound::Excluded(&i) => i,
            Bound::Unbounded => self.0.len,
        };

        (start <= end && end <= self.0.len).then_some(Span(Slice {
            offset: self.0.offset + start as isize,
            len: end,
            ..self.0
        }))
    }

    /// cache span at `mid`. Analogous to [`slice::split_at`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::cpu::buffer::layout::SliceCache;
    /// let _cache = SliceCache::new(4, |span| {
    ///     let (left, right) = span.split_at(2);
    ///     Box::new([left, right])
    /// });
    /// ```
    #[inline]
    pub fn split_at(&self, mid: usize) -> (Self, Self) {
        assert!(mid <= self.0.len);

        (
            Span(Slice { len: mid, ..self.0 }),
            Span(Slice {
                offset: self.0.offset + mid as isize,
                len: self.0.len - mid,
                ..self.0
            }),
        )
    }

    /// Returns an [Iterator](Chunks) over `chunk_size` elements of thr slice. Analogous to [`slice::chunks`].
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::cpu::buffer::layout::SliceCache;
    /// let _cache = SliceCache::new(4, |span| {
    ///     span.chunks(2).collect()
    /// });
    /// ```
    #[inline]
    pub fn chunks(self, chunk_size: usize) -> Chunks<'a> {
        Chunks {
            slice: self.0,
            size: chunk_size,
        }
    }
}

impl fmt::Debug for Span<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.offset.fmt(f)?;
        write!(f, "..")?;
        self.0.len.fmt(f)?;

        Ok(())
    }
}

/// An [iterator](std::iter::Iterator) returned by [`Span::chunks`].
pub struct Chunks<'a> {
    slice: Slice<'a, ()>,
    size: usize,
}

impl<'a> Iterator for Chunks<'a> {
    type Item = Span<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        (self.slice.len > 0).then(|| {
            let span = Span(Slice {
                len: self.size.min(self.slice.len),
                ..self.slice
            });

            self.slice.offset += self.size as isize;
            self.slice.len = self.slice.len.saturating_sub(self.size);

            span
        })
    }
}

impl fmt::Debug for Chunks<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Chunks")
            .field("span", &Span::from_slice(&self.slice))
            .field("size", &self.size)
            .finish()
    }
}

/// A [reference] wrapper returned by [`SliceCache::access`].
#[repr(transparent)]
#[derive(Debug)]
pub struct Ref<'a, T: ?Sized>(&'a mut T);

impl<'a, T: ?Sized> Ref<'a, T> {
    pub fn get(&'a mut self) -> &'a mut T {
        self.0
    }
}

impl<T: ?Sized> Deref for Ref<'_, T> {
    type Target = T;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<T: ?Sized> DerefMut for Ref<'_, T> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
    }
}

impl<T: ?Sized> Drop for Ref<'_, T> {
    #[inline]
    fn drop(&mut self) {
        ROOT.store(None);
    }
}

/// A cache of non-overlapping mutable sub-slices of that enforces lifetimes dynamically.
///
/// This type is useful when you have to give up on the mutable reference to a slice but need
/// a way to cache mutable sub-slices deriving from it.
///
/// # Examples
///
/// ```
/// # use forma_render::cpu::buffer::layout::SliceCache;
/// let mut array = [1, 2, 3];
///
/// let mut cache = SliceCache::new(3, |span| {
///     let (left, right) = span.split_at(1);
///     Box::new([left, right])
/// });
///
/// for slice in cache.access(&mut array).unwrap().iter_mut() {
///     for val in slice.iter_mut() {
///         *val += 1;
///     }
/// }
///
/// assert_eq!(array, [2, 3, 4]);
/// ```
pub struct SliceCache {
    len: usize,
    slices: Box<[Slice<'static, ()>]>,
}

impl SliceCache {
    /// Creates a new slice cache by storing sub-spans created from a root passed to the closure
    /// `f`. `len` is the minimum slice length that can then be passed to [`access`](Self::access).
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::cpu::buffer::layout::SliceCache;
    /// let _cache = SliceCache::new(3, |span| {
    ///     let (left, right) = span.split_at(1);
    ///     // All returned sub-spans stem from the span passed above.
    ///     Box::new([left, right])
    /// });
    /// ```
    #[inline]
    pub fn new<F>(len: usize, f: F) -> Self
    where
        F: Fn(Span<'_>) -> Box<[Span<'_>]> + 'static,
    {
        let span = Span(Slice {
            offset: 0,
            len,
            _phantom: PhantomData,
        });

        // `Span<'_>` is transparent over `Slice<'_, ()>`. Since the `'_` above is used just to
        // trap the span inside the closure, transmuting to `Slice<'static, ()>` does not make any
        // difference.
        Self {
            len,
            slices: unsafe { mem::transmute(f(span)) },
        }
    }

    /// Accesses the `slice` by returning all the sub-slices equivalent to the previously created
    /// [spans](Span) in the closure passed to [`new`](Self::new).
    ///
    /// If the `slice` does not have a length at least as large as the one passed to
    /// [`new`](Self::new), this function returns `None`.
    ///
    /// Note: this method should not be called concurrently with any other `access` calls since it
    /// will wait for the previously returned [`Ref`] to be dropped.
    ///
    /// # Examples
    ///
    /// ```
    /// # use forma_render::cpu::buffer::layout::SliceCache;
    /// let mut array = [1, 2, 3];
    ///
    /// let mut cache = SliceCache::new(3, |span| {
    ///     let (left, right) = span.split_at(1);
    ///     Box::new([left, right])
    /// });
    ///
    /// let copy = array;
    /// let skipped_one = cache.access(&mut array).unwrap();
    ///
    /// assert_eq!(&*skipped_one[1], &copy[1..]);
    /// ```
    #[inline]
    pub fn access<'c, 's, T>(&'c mut self, slice: &'s mut [T]) -> Option<Ref<'c, [Slice<'s, T>]>> {
        if slice.len() >= self.len {
            while ROOT
                .compare_exchange(
                    None,
                    Some(NonNull::new(slice.as_mut_ptr()).unwrap().cast().into()),
                )
                .is_err()
            {
                // This spin lock here is mostly for being able to run tests in parallel. Being
                // able to render to `forma::Composition`s in parallel is currently not supported
                // and might poor performance due to priority inversion.
                hint::spin_loop();
            }

            // Generic `Slice<'static, ()>` are transmuted to `Slice<'s, T>`, enforcing the
            // original `slice`'s lifetime. Since slices are simply pairs of `(offset, len)`,
            // transmuting `()` to `T` relies on the `ROOT` being set up above with the correct pointer.
            return Some(unsafe { mem::transmute(&mut *self.slices) });
        }

        None
    }

    #[cfg(test)]
    fn try_access<'c, 's, T>(&'c mut self, slice: &'s mut [T]) -> Option<Ref<'c, [Slice<'s, T>]>> {
        if slice.len() >= self.len
            && ROOT
                .compare_exchange(
                    None,
                    Some(NonNull::new(slice.as_mut_ptr()).unwrap().cast().into()),
                )
                .is_ok()
        {
            // Generic `Slice<'static, ()>` are transmuted to `Slice<'s, T>`, enforcing the
            // original `slice`'s lifetime. Since slices are simply pairs of `(offset, len)`,
            // transmuting `()` to `T` relies on the `ROOT` being set up above with the correct pointer.
            return Some(unsafe { mem::transmute(&mut *self.slices) });
        }

        None
    }
}

impl fmt::Debug for SliceCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list()
            .entries(self.slices.iter().map(Span::from_slice))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_at() {
        let mut cache = SliceCache::new(5, |span| {
            let (left, right) = span.split_at(2);
            Box::new([left, right])
        });
        let mut array = [1, 2, 3, 4, 5];

        for slice in cache.access(&mut array).unwrap().iter_mut() {
            for val in slice.iter_mut() {
                *val += 1;
            }
        }

        assert_eq!(array, [2, 3, 4, 5, 6]);
    }

    #[test]
    fn chunks() {
        let mut cache = SliceCache::new(5, |span| span.chunks(2).collect());
        let mut array = [1, 2, 3, 4, 5];

        for slice in cache.access(&mut array).unwrap().iter_mut() {
            for val in slice.iter_mut() {
                *val += 1;
            }
        }

        assert_eq!(array, [2, 3, 4, 5, 6]);
    }

    #[test]
    fn ref_twice() {
        let mut cache = SliceCache::new(5, |span| {
            let (left, right) = span.split_at(2);
            Box::new([left, right])
        });
        let mut array = [1, 2, 3, 4, 5];

        for slice in cache.access(&mut array).unwrap().iter_mut() {
            for val in slice.iter_mut() {
                *val += 1;
            }
        }

        for slice in cache.access(&mut array).unwrap().iter_mut() {
            for val in slice.iter_mut() {
                *val += 1;
            }
        }

        assert_eq!(array, [3, 4, 5, 6, 7]);
    }

    #[test]
    fn access_twice() {
        let mut cache0 = SliceCache::new(5, |span| Box::new([span]));
        let mut cache1 = SliceCache::new(5, |span| Box::new([span]));

        let mut array0 = [1, 2, 3, 4, 5];
        let mut array1 = [1, 2, 3, 4, 5];

        let _slices = cache0.access(&mut array0).unwrap();

        assert!(matches!(cache1.try_access(&mut array1), None));
    }

    // #[test]
    // fn fails_due_to_too_short_lifetime() {
    //     let mut cache = SliceCache::new(16, |span| Box::new([span]));

    //     let slices = {
    //         let mut buffer = [0u8; 16];

    //         let slices = cache.access(&mut buffer).unwrap();
    //         let slice = &mut *slices[0];

    //         slice
    //     };

    //     &slices[0];
    // }

    // #[test]
    // fn fails_due_to_mixed_spans() {
    //     SliceCache::new(16, |span0| {
    //         let (left, right) = span0.split_at(2);

    //         SliceCache::new(4, |span1| {
    //             Box::new([left])
    //         });

    //         Box::new([right])
    //     });
    // }

    // #[test]
    // fn fails_due_to_t_not_being_send() {
    //     use std::rc::Rc;

    //     use rayon::prelude::*;

    //     let mut array = [Rc::new(1), Rc::new(2), Rc::new(3)];

    //     let mut cache = SliceCache::new(3, |span| {
    //         let (left, right) = span.split_at(1);
    //         Box::new([left, right])
    //     });

    //     cache.access(&mut array).unwrap().par_iter_mut().for_each(|slice| {
    //         for val in slice.iter_mut() {
    //             *val += 1;
    //         }
    //     });
    // }

    // #[test]
    // fn fails_to_export_span() {
    //     let mut leaked = None;

    //     let mut cache0 = SliceCache::new(1, |span| {
    //         leaked = Some(span);
    //         Box::new([])
    //     });

    //     let mut cache1 = SliceCache::new(1, |span| {
    //         Box::new([leaked.take().unwrap()])
    //     });
    // }

    // #[test]
    // fn fails_due_to_dropped_slice() {
    //     let mut array = [1, 2, 3];

    //     let mut cache = SliceCache::new(3, |span| {
    //         let (left, right) = span.split_at(1);
    //         Box::new([left, right])
    //     });

    //     let slices = cache.access(&mut array).unwrap();

    //     std::mem::drop(array);

    //     slices[0][0] = 0;
    // }
}
