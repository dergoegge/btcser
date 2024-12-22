use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

pub trait WeightedReservoirSampler<T: Clone> {
    fn new(seed: u64) -> Self
    where
        Self: Sized;
    fn add(&mut self, item: T, weight: u64);
    fn get_sample(self) -> Option<T>;
}

// https://en.wikipedia.org/wiki/Reservoir_sampling#Algorithm_A-Chao
pub struct ChaoSampler<T> {
    rng: StdRng,
    current_sample: Option<T>,
    total_weight: u64,
}

impl<T: Clone> WeightedReservoirSampler<T> for ChaoSampler<T> {
    fn new(seed: u64) -> Self {
        Self {
            rng: StdRng::seed_from_u64(seed),
            current_sample: None,
            total_weight: 0,
        }
    }

    fn add(&mut self, item: T, weight: u64) {
        if weight == 0 {
            return;
        }

        self.total_weight += weight;

        if weight == self.total_weight || self.rng.gen_range(1..=self.total_weight) <= weight {
            self.current_sample = Some(item);
        }
    }

    fn get_sample(self) -> Option<T> {
        self.current_sample
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    pub struct TestSampler<T> {
        samples: Vec<T>,
    }

    impl<T: Clone> TestSampler<T> {
        pub fn new(_seed: u64) -> Self {
            Self {
                samples: Vec::new(),
            }
        }

        // Additional method specific to TestSampler to get all samples
        pub fn get_samples(&self) -> Vec<T> {
            self.samples.clone()
        }
    }

    impl<T: Clone> WeightedReservoirSampler<T> for TestSampler<T> {
        fn new(seed: u64) -> Self {
            Self::new(seed)
        }

        fn add(&mut self, item: T, weight: u64) {
            if weight > 0 {
                self.samples.push(item);
            }
        }

        fn get_sample(self) -> Option<T> {
            if self.samples.is_empty() {
                None
            } else {
                // For testing, always return the first item
                Some(self.samples[0].clone())
            }
        }
    }

    #[test]
    fn test_chao_sampler_basic() {
        let mut sampler = ChaoSampler::new(42);
        sampler.add(1, 1);
        sampler.add(2, 1);
        sampler.add(3, 1);
        assert!(sampler.get_sample().is_some());
    }

    #[test]
    fn test_chao_sampler_zero_weight() {
        let mut sampler = ChaoSampler::new(42);
        sampler.add(1, 0);
        assert!(sampler.get_sample().is_none());
    }

    #[test]
    fn test_chao_sampler_single_item() {
        let mut sampler = ChaoSampler::new(42);
        sampler.add(1, 1);
        assert_eq!(sampler.get_sample(), Some(1));
    }

    #[test]
    fn test_chao_sampler_empty() {
        let sampler = ChaoSampler::<i32>::new(42);
        assert_eq!(sampler.get_sample(), None);
    }
}
