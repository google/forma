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

use rayon::{
    iter::plumbing::{bridge, Consumer, Producer, ProducerCallback, UnindexedConsumer},
    prelude::*,
};

/// Parallel-enabled prefix-scan that iterates over group- and local-indices.
#[derive(Clone, Debug, Default)]
pub struct PrefixScanIter<'s> {
    sums: &'s [u32],
    group_start: u32,
    group_end: u32,
    start: u32,
    end: u32,
}

impl Iterator for PrefixScanIter<'_> {
    type Item = (u32, u32);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.start >= self.end {
                return None;
            }

            let exclusive_sum = self
                .group_start
                .checked_sub(1)
                .map(|i| self.sums[i as usize])
                .unwrap_or_default();
            let inclusive_sum = self.sums[self.group_start as usize];

            if exclusive_sum == inclusive_sum {
                self.group_start += 1;
                continue;
            }

            let result = (self.group_start, self.start - exclusive_sum);

            self.start += 1;

            if self.start == inclusive_sum {
                self.group_start += 1;
            }

            return Some(result);
        }
    }
}

impl DoubleEndedIterator for PrefixScanIter<'_> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        loop {
            if self.start >= self.end {
                return None;
            }

            let exclusive_sum = self
                .group_end
                .checked_sub(1)
                .map(|i| self.sums[i as usize])
                .unwrap_or_default();
            let inclusive_sum = self.sums[self.group_end as usize];

            if exclusive_sum == inclusive_sum {
                self.group_end = self.group_end.saturating_sub(1);
                continue;
            }

            let result = (self.group_end, self.end - 1 - exclusive_sum);

            self.end -= 1;

            if self.end == exclusive_sum {
                self.group_end = self.group_end.saturating_sub(1);
            }

            return Some(result);
        }
    }
}

impl ExactSizeIterator for PrefixScanIter<'_> {
    fn len(&self) -> usize {
        (self.end - self.start) as usize
    }
}

struct PrefixScanIterProducer<'s> {
    inner: PrefixScanIter<'s>,
}

impl<'s> Producer for PrefixScanIterProducer<'s> {
    type Item = (u32, u32);

    type IntoIter = PrefixScanIter<'s>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.inner
    }

    #[inline]
    fn split_at(self, index: usize) -> (Self, Self) {
        let index = index as u32 + self.inner.start;

        let mid = match self.inner.sums.binary_search(&index) {
            Ok(mid) => mid + 1,
            Err(mid) => mid,
        } as u32;

        (
            Self {
                inner: PrefixScanIter {
                    sums: self.inner.sums,
                    group_start: self.inner.group_start,
                    group_end: mid,
                    start: self.inner.start,
                    end: index,
                },
            },
            Self {
                inner: PrefixScanIter {
                    sums: self.inner.sums,
                    group_start: mid,
                    group_end: self.inner.group_end,
                    start: index,
                    end: self.inner.end,
                },
            },
        )
    }
}

impl<'s> PrefixScanIter<'s> {
    #[inline]
    pub fn new(sums: &'s [u32]) -> Self {
        Self {
            sums,
            group_start: 0,
            group_end: sums.len().saturating_sub(1) as u32,
            start: 0,
            end: sums.last().copied().unwrap_or_default(),
        }
    }
}

impl<'s> IntoParallelIterator for PrefixScanIter<'s> {
    type Iter = PrefixScanParIter<'s>;

    type Item = (u32, u32);

    #[inline]
    fn into_par_iter(self) -> Self::Iter {
        PrefixScanParIter { iter: self }
    }
}

pub struct PrefixScanParIter<'s> {
    iter: PrefixScanIter<'s>,
}

impl ParallelIterator for PrefixScanParIter<'_> {
    type Item = (u32, u32);

    #[inline]
    fn drive_unindexed<C>(self, consumer: C) -> C::Result
    where
        C: UnindexedConsumer<Self::Item>,
    {
        bridge(self, consumer)
    }

    #[inline]
    fn opt_len(&self) -> Option<usize> {
        Some(self.iter.len())
    }
}

impl IndexedParallelIterator for PrefixScanParIter<'_> {
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }

    #[inline]
    fn drive<C: Consumer<Self::Item>>(self, consumer: C) -> C::Result {
        bridge(self, consumer)
    }

    #[inline]
    fn with_producer<CB: ProducerCallback<Self::Item>>(self, callback: CB) -> CB::Output {
        callback.callback(PrefixScanIterProducer { inner: self.iter })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_iter() {
        let sums = &[];

        assert_eq!(PrefixScanIter::new(sums).collect::<Vec<_>>(), []);
    }

    #[test]
    fn local_iter() {
        let sums = &[2, 5, 9, 15];
        let iter = PrefixScanIter::new(sums);

        assert_eq!(
            iter.collect::<Vec<_>>(),
            [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 0),
                (3, 1),
                (3, 2),
                (3, 3),
                (3, 4),
                (3, 5),
            ]
        );
    }

    #[test]
    fn local_iter_rev() {
        let sums = &[2, 5, 9, 15];
        let iter = PrefixScanIter::new(sums);

        assert_eq!(
            iter.rev().collect::<Vec<_>>(),
            [
                (3, 5),
                (3, 4),
                (3, 3),
                (3, 2),
                (3, 1),
                (3, 0),
                (2, 3),
                (2, 2),
                (2, 1),
                (2, 0),
                (1, 2),
                (1, 1),
                (1, 0),
                (0, 1),
                (0, 0),
            ]
        );
    }

    #[test]
    fn both_ends() {
        let sums = &[2, 5, 9, 15];
        let mut iter = PrefixScanIter::new(sums);

        assert_eq!(iter.len(), 15);

        assert_eq!(iter.next(), Some((0, 0)));
        assert_eq!(iter.next_back(), Some((3, 5)));
        assert_eq!(iter.next(), Some((0, 1)));
        assert_eq!(iter.next_back(), Some((3, 4)));
        assert_eq!(iter.next(), Some((1, 0)));
        assert_eq!(iter.next_back(), Some((3, 3)));
        assert_eq!(iter.next(), Some((1, 1)));

        assert_eq!(iter.len(), 8);

        assert_eq!(iter.next_back(), Some((3, 2)));
        assert_eq!(iter.next(), Some((1, 2)));
        assert_eq!(iter.next_back(), Some((3, 1)));
        assert_eq!(iter.next(), Some((2, 0)));
        assert_eq!(iter.next_back(), Some((3, 0)));
        assert_eq!(iter.next(), Some((2, 1)));
        assert_eq!(iter.next_back(), Some((2, 3)));
        assert_eq!(iter.next(), Some((2, 2)));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);

        assert_eq!(iter.len(), 0);
    }

    #[test]
    fn empty_groups() {
        let sums = &[2, 2, 5, 5];
        let iter = PrefixScanIter::new(sums);

        assert_eq!(
            iter.collect::<Vec<_>>(),
            [(0, 0), (0, 1), (2, 0), (2, 1), (2, 2),]
        );
    }

    #[test]
    fn empty_groups_rev() {
        let sums = &[2, 2, 5, 5];
        let iter = PrefixScanIter::new(sums);

        assert_eq!(
            iter.rev().collect::<Vec<_>>(),
            [(2, 2), (2, 1), (2, 0), (0, 1), (0, 0),]
        );
    }

    #[test]
    fn par_iter() {
        let sums = &[2, 5, 9, 15];
        let iter = PrefixScanIter::new(sums);

        assert_eq!(
            iter.into_par_iter().collect::<Vec<_>>(),
            [
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 0),
                (3, 1),
                (3, 2),
                (3, 3),
                (3, 4),
                (3, 5),
            ]
        );
    }

    #[test]
    fn par_iter2() {
        let sums = &[3, 6, 10, 11];
        let iter = PrefixScanIter::new(sums);

        assert_eq!(
            iter.into_par_iter().collect::<Vec<_>>(),
            [
                (0, 0),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 1),
                (1, 2),
                (2, 0),
                (2, 1),
                (2, 2),
                (2, 3),
                (3, 0),
            ]
        );
    }
}
