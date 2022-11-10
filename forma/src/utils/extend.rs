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

macro_rules! extend_tuple {
    ( $name:ident, $( ( $fields:tt $types:ident ) ),+ ) => {
        pub struct $name<'a, $($types),+> {
            tuple: ($(&'a mut Vec<$types>),+),
        }

        impl<'a, $($types),+> $name<'a, $($types),+> {
            pub fn new(tuple: ($(&'a mut Vec<$types>),+)) -> Self {
                Self { tuple }
            }
        }

        impl<$($types),+> ::rayon::iter::ParallelExtend<($($types),+)> for $name<'_, $($types),+>
        where
            $(
                $types: Send,
            )+
        {
            fn par_extend<PI>(&mut self, par_iter: PI)
            where
                PI: ::rayon::iter::IntoParallelIterator<Item = ($($types),+)>,
            {
                use ::std::{
                    collections::LinkedList, ptr, slice, sync::atomic::{AtomicUsize, Ordering},
                };

                use ::rayon::{
                    iter::plumbing::{Consumer, Folder, Reducer, UnindexedConsumer},
                    prelude::*,
                };

                struct NoopReducer;

                impl Reducer<()> for NoopReducer {
                    fn reduce(self, _left: (), _right: ()) {}
                }

                struct CollectTupleConsumer<'c, $($types: Send),+> {
                    writes: &'c AtomicUsize,
                    targets: ($(&'c mut [$types]),+),
                }

                struct CollectTupleFolder<'c, $($types: Send),+> {
                    global_writes: &'c AtomicUsize,
                    local_writes: usize,
                    targets: ($(slice::IterMut<'c, $types>),+),
                }

                impl<'c, $($types: Send + 'c),+> Consumer<($($types),+)>
                for CollectTupleConsumer<'c, $($types),+>
                {
                    type Folder = CollectTupleFolder<'c, $($types),+>;
                    type Reducer = NoopReducer;
                    type Result = ();

                    fn split_at(self, index: usize) -> (Self, Self, NoopReducer) {
                        let CollectTupleConsumer { writes, targets } = self;

                        let splits = (
                            $(
                                targets.$fields.split_at_mut(index),
                            )+
                        );

                        (
                            CollectTupleConsumer {
                                writes,
                                targets: (
                                    $(
                                        splits.$fields.0,
                                    )+
                                ),
                            },
                            CollectTupleConsumer {
                                writes,
                                targets: (
                                    $(
                                        splits.$fields.1,
                                    )+
                                ),
                            },
                            NoopReducer,
                        )
                    }

                    fn into_folder(self) -> CollectTupleFolder<'c, $($types),+> {
                        CollectTupleFolder {
                            global_writes: self.writes,
                            local_writes: 0,
                            targets: (
                                $(
                                    self.targets.$fields.iter_mut(),
                                )+
                            ),
                        }
                    }

                    fn full(&self) -> bool {
                        false
                    }
                }

                impl<'c, $($types: Send + 'c),+> Folder<($($types),+)>
                for CollectTupleFolder<'c, $($types),+>
                {
                     type Result = ();

                    fn consume(
                        mut self,
                        item: ($($types),+),
                    ) -> CollectTupleFolder<'c, $($types),+> {
                        $(
                            let head = self
                                .targets
                                .$fields
                                .next()
                                .expect("too many values pushed to consumer");
                            unsafe {
                                ptr::write(head, item.$fields);
                            }
                        )+

                        self.local_writes += 1;
                        self
                    }

                    fn complete(self) {
                        self.global_writes.fetch_add(self.local_writes, Ordering::Relaxed);
                    }

                    fn full(&self) -> bool {
                        false
                    }
                }

                impl<'c, $($types: Send + 'c),+> UnindexedConsumer<($($types),+)>
                for CollectTupleConsumer<'c, $($types),+>
                {
                     fn split_off_left(&self) -> Self {
                        unreachable!("CollectTupleConsumer must be indexed!")
                    }
                    fn to_reducer(&self) -> Self::Reducer {
                        NoopReducer
                    }
                }

                struct CollectTuple<'c, $($types: Send),+> {
                    writes: AtomicUsize,
                    tuple: ($(&'c mut Vec<$types>),+),
                    len: usize,
                }

                impl<'c, $($types: Send),+> CollectTuple<'c, $($types),+> {
                    pub fn new(tuple: ($(&'c mut Vec<$types>),+), len: usize) -> Self {
                        Self {
                            writes: AtomicUsize::new(0),
                            tuple,
                            len,
                        }
                    }

                    pub fn as_consumer(&mut self) -> CollectTupleConsumer<'_, $($types),+> {
                        $(
                            self.tuple.$fields.reserve(self.len);
                        )+

                        CollectTupleConsumer {
                            writes: &self.writes,
                            targets: (
                                $(
                                    {
                                        let vec = &mut self.tuple.$fields;
                                        let start = vec.len();
                                        let slice = &mut vec[start..];
                                        unsafe {
                                            slice::from_raw_parts_mut(
                                                slice.as_mut_ptr(),
                                                self.len,
                                            )
                                        }
                                    }
                                ),+
                            ),
                        }
                    }

                    pub fn complete(mut self) {
                        unsafe {
                            let actual_writes = self.writes.load(Ordering::Relaxed);
                            assert!(
                                actual_writes == self.len,
                                "expected {} total writes, but got {}",
                                self.len,
                                actual_writes
                            );

                            $(
                                let vec = &mut self.tuple.$fields;
                                let new_len = vec.len() + self.len;
                                vec.set_len(new_len);
                            )+
                        }
                    }
                }

                let par_iter = par_iter.into_par_iter();
                match par_iter.opt_len() {
                    Some(len) => {
                        let mut collect = CollectTuple::new(($(self.tuple.$fields),+), len);
                        par_iter.drive_unindexed(collect.as_consumer());
                        collect.complete()
                    }
                    None => {
                        let list = par_iter
                            .into_par_iter()
                            .fold(|| ($(Vec::<$types>::new()),+), |mut vecs, elem| {
                                $(
                                    vecs.$fields.push(elem.$fields);
                                )+
                                vecs
                            })
                            .map(|item| {
                                let mut list = LinkedList::new();
                                list.push_back(item);
                                list
                            })
                            .reduce(LinkedList::new, |mut list1, mut list2| {
                                list1.append(&mut list2);
                                list1
                            });
                        let len = list.iter().map(|vecs| vecs.0.len()).sum();

                        $(
                            self.tuple.$fields.reserve(len);
                        )+
                        for mut vecs in list {
                            $(
                                self.tuple.$fields.append(&mut vecs.$fields);
                            )+
                        }
                    }
                }
            }
        }
    };
}

extend_tuple!(ExtendTuple3, (0 A), (1 B), (2 C));
extend_tuple!(ExtendTuple10, (0 A), (1 B), (2 C), (3 D), (4 E), (5 F), (6 G), (7 H), (8 I), (9 J));

pub struct ExtendVec<'a, T> {
    vec: &'a mut Vec<T>,
}

impl<'a, T> ExtendVec<'a, T> {
    pub fn new(vec: &'a mut Vec<T>) -> Self {
        Self { vec }
    }
}

impl<T: Send> rayon::iter::ParallelExtend<T> for ExtendVec<'_, T> {
    fn par_extend<I>(&mut self, par_iter: I)
    where
        I: rayon::iter::IntoParallelIterator<Item = T>,
    {
        self.vec.par_extend(par_iter);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use rayon::prelude::*;

    #[test]
    fn tuple10() {
        let mut vec0 = vec![];
        let mut vec1 = vec![];
        let mut vec2 = vec![];
        let mut vec3 = vec![];
        let mut vec4 = vec![];
        let mut vec5 = vec![];
        let mut vec6 = vec![];
        let mut vec7 = vec![];
        let mut vec8 = vec![];
        let mut vec9 = vec![];

        ExtendTuple10::new((
            &mut vec0, &mut vec1, &mut vec2, &mut vec3, &mut vec4, &mut vec5, &mut vec6, &mut vec7,
            &mut vec8, &mut vec9,
        ))
        .par_extend((0..3).into_par_iter().map(|i| {
            (
                i,
                i + 1,
                i + 2,
                i + 3,
                i + 4,
                i + 5,
                i + 6,
                i + 7,
                i + 8,
                i + 9,
            )
        }));

        assert_eq!(vec0, [0, 1, 2]);
        assert_eq!(vec1, [1, 2, 3]);
        assert_eq!(vec2, [2, 3, 4]);
        assert_eq!(vec3, [3, 4, 5]);
        assert_eq!(vec4, [4, 5, 6]);
        assert_eq!(vec5, [5, 6, 7]);
        assert_eq!(vec6, [6, 7, 8]);
        assert_eq!(vec7, [7, 8, 9]);
        assert_eq!(vec8, [8, 9, 10]);
        assert_eq!(vec9, [9, 10, 11]);
    }
}
