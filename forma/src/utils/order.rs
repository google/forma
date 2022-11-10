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

use std::{convert::TryFrom, error::Error, fmt, num::NonZeroU32};

use crate::consts;

#[derive(Debug, Eq, PartialEq)]
pub enum OrderError {
    ExceededLayerLimit,
}

impl fmt::Display for OrderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "exceeded layer limit ({})", consts::LAYER_LIMIT)
    }
}

impl Error for OrderError {}

#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Order(NonZeroU32);

impl Default for Order {
    fn default() -> Self {
        match Self::new(0) {
            Ok(order) => order,
            Err(_) => panic!("0 is smaller than Order::MAX"),
        }
    }
}

impl Order {
    pub const MAX: Self = Self(
        match NonZeroU32::new(consts::LAYER_LIMIT as u32 ^ u32::MAX) {
            Some(val) => val,
            None => panic!("LAYER_LIMIT is smaller than u32::MAX"),
        },
    );

    pub const fn as_u32(&self) -> u32 {
        self.0.get() ^ u32::MAX
    }

    pub const fn new(order: u32) -> Result<Self, OrderError> {
        if order > Self::MAX.as_u32() {
            Err(OrderError::ExceededLayerLimit)
        } else {
            Ok(Self(match NonZeroU32::new(order ^ u32::MAX) {
                Some(val) => val,
                None => panic!("Order::MAX is smaller than u32::MAX"),
            }))
        }
    }
}

impl fmt::Debug for Order {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Order").field(&self.as_u32()).finish()
    }
}

impl From<Order> for u32 {
    fn from(order: Order) -> Self {
        order.as_u32()
    }
}

impl TryFrom<u32> for Order {
    type Error = OrderError;

    fn try_from(order: u32) -> Result<Self, OrderError> {
        Self::new(order)
    }
}

impl TryFrom<usize> for Order {
    type Error = OrderError;

    fn try_from(order: usize) -> Result<Self, OrderError> {
        u32::try_from(order)
            .map_err(|_| OrderError::ExceededLayerLimit)
            .and_then(Self::try_from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrong_u32_order_value() {
        let order = Order::MAX.as_u32() + 1;

        assert_eq!(Order::try_from(order), Err(OrderError::ExceededLayerLimit));
    }

    #[test]
    fn wrong_usize_order_values() {
        let order = (Order::MAX.as_u32() + 1) as usize;

        assert_eq!(Order::try_from(order), Err(OrderError::ExceededLayerLimit));

        let order = u64::MAX as usize;

        assert_eq!(Order::try_from(order), Err(OrderError::ExceededLayerLimit));
    }

    #[test]
    fn correct_order_value() {
        let order_value = Order::MAX.as_u32();
        let order = Order::try_from(order_value);

        assert_eq!(
            order,
            Ok(Order(
                NonZeroU32::new(order_value as u32 ^ u32::MAX).unwrap()
            ))
        );
    }
}
