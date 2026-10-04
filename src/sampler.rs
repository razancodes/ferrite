//! Token sampling: penalties, temperature, top-k, top-p, min-p.

use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct SamplerConfig {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for SamplerConfig {
    /// Qwen3.5 recommended non-thinking settings for text.
    fn default() -> Self {
        SamplerConfig {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.8,
            min_p: 0.0,
            presence_penalty: 1.5,
            frequency_penalty: 0.0,
            repetition_penalty: 1.0,
            seed: 0,
        }
    }
}

impl SamplerConfig {
    pub fn greedy() -> Self {
        SamplerConfig { temperature: 0.0, presence_penalty: 0.0, ..Default::default() }
    }
    /// Qwen3.5 recommended thinking-mode settings.
    pub fn thinking() -> Self {
        SamplerConfig { temperature: 1.0, top_p: 0.95, ..Default::default() }
    }
}

pub struct Sampler {
    pub cfg: SamplerConfig,
    rng: u64,
    counts: HashMap<u32, u32>,
}

impl Sampler {
    pub fn new(cfg: SamplerConfig) -> Self {
        let seed = if cfg.seed == 0 {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(42)
        } else {
            cfg.seed
        };
        Sampler { cfg, rng: seed ^ 0x9E37_79B9_7F4A_7C15, counts: HashMap::new() }
    }

    fn next_f32(&mut self) -> f32 {
        // splitmix64
        self.rng = self.rng.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Record a generated token (for presence/frequency/repetition penalties).
    pub fn accept(&mut self, tok: u32) {
        *self.counts.entry(tok).or_insert(0) += 1;
    }

    pub fn reset(&mut self) {
        self.counts.clear();
    }

    /// Apply penalties in place to the (few) tokens generated so far and return
    /// the original values so the caller can restore them afterwards: rows can
    /// be sampled without copying 248K logits.
    fn apply_penalties(&self, logits: &mut [f32]) -> Vec<(u32, f32)> {
        let c = &self.cfg;
        if c.presence_penalty == 0.0 && c.frequency_penalty == 0.0 && c.repetition_penalty == 1.0 {
            return Vec::new();
        }
        let mut saved = Vec::with_capacity(self.counts.len());
        for (&tok, &n) in &self.counts {
            let Some(l) = logits.get_mut(tok as usize) else { continue };
            saved.push((tok, *l));
            if c.repetition_penalty != 1.0 {
                *l = if *l > 0.0 { *l / c.repetition_penalty } else { *l * c.repetition_penalty };
            }
            *l -= c.presence_penalty + c.frequency_penalty * n as f32;
        }
        saved
    }

    fn restore(logits: &mut [f32], saved: &[(u32, f32)]) {
        for &(t, v) in saved {
            logits[t as usize] = v;
        }
    }

    /// The `k` largest logits, sorted descending. A running threshold means the
    /// scan over the vocabulary only touches the rare candidates.
    fn top_k(logits: &[f32], k: usize) -> Vec<(f32, u32)> {
        let cap = (4 * k).max(64);
        let mut buf: Vec<(f32, u32)> = Vec::with_capacity(cap + 1);
        let mut thr = f32::NEG_INFINITY;
        for (i, &l) in logits.iter().enumerate() {
            if l > thr {
                buf.push((l, i as u32));
                if buf.len() >= cap {
                    buf.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
                    buf.truncate(k);
                    thr = buf.iter().fold(f32::INFINITY, |m, x| m.min(x.0));
                }
            }
        }
        if buf.len() > k {
            buf.select_nth_unstable_by(k - 1, |a, b| b.0.total_cmp(&a.0));
            buf.truncate(k);
        }
        buf.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        buf
    }

    /// Final sampling distribution (penalties, temperature, top-k, min-p, top-p)
    /// for a logits row: candidate ids and normalized probabilities. `logits` is
    /// left unchanged.
    fn distribution(&mut self, logits: &mut [f32]) -> (Vec<u32>, Vec<f32>) {
        let saved = self.apply_penalties(logits);
        let c = self.cfg.clone();
        let cand = if c.top_k > 0 {
            Self::top_k(logits, c.top_k)
        } else {
            // no top-k: everything within e^-30 of the best (temperature-scaled)
            let best = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let cutoff = best - 30.0 * c.temperature;
            let mut v: Vec<(f32, u32)> =
                logits.iter().enumerate().filter(|(_, l)| **l >= cutoff).map(|(i, l)| (*l, i as u32)).collect();
            v.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            v
        };
        Self::restore(logits, &saved);
        let best = cand[0].0;
        let inv_t = 1.0 / c.temperature;
        let mut probs: Vec<f32> = cand.iter().map(|(l, _)| ((l - best) * inv_t).exp()).collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);
        let mut n = probs.len();
        if c.min_p > 0.0 {
            let th = c.min_p * probs[0];
            n = probs.iter().take_while(|&&p| p >= th).count().max(1);
        }
        if c.top_p < 1.0 {
            let mut cum = 0.0;
            for (i, p) in probs.iter().take(n).enumerate() {
                cum += p;
                if cum >= c.top_p {
                    n = i + 1;
                    break;
                }
            }
        }
        probs.truncate(n);
        let total: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= total);
        (cand[..n].iter().map(|c| c.1).collect(), probs)
    }

    fn greedy(&self, logits: &mut [f32]) -> u32 {
        let saved = self.apply_penalties(logits);
        let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
        for (i, &l) in logits.iter().enumerate() {
            if l > best {
                best = l;
                bi = i;
            }
        }
        Self::restore(logits, &saved);
        bi as u32
    }

    /// Sample a token from a logits row (the row is left unchanged).
    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        if self.cfg.temperature <= 0.0 || self.cfg.top_k == 1 {
            return self.greedy(logits);
        }
        let (cand, probs) = self.distribution(logits);
        let mut r = self.next_f32();
        for (t, p) in cand.iter().zip(&probs) {
            r -= p;
            if r <= 0.0 {
                return *t;
            }
        }
        *cand.last().unwrap()
    }

    /// Speculative sampling against a deterministic draft token `draft`.
    /// Returns (token, accepted). The draft is accepted with probability p(draft)
    /// under the fully processed target distribution; otherwise a token is drawn
    /// from p with the draft removed. The output distribution is exactly `p`.
    pub fn sample_spec(&mut self, logits: &mut [f32], draft: u32) -> (u32, bool) {
        if self.cfg.temperature <= 0.0 || self.cfg.top_k == 1 {
            let t = self.greedy(logits);
            return (t, t == draft);
        }
        let (cand, probs) = self.distribution(logits);
        let pd = cand.iter().position(|&t| t == draft).map(|i| probs[i]).unwrap_or(0.0);
        if self.next_f32() < pd {
            return (draft, true);
        }
        let total: f32 = cand.iter().zip(&probs).filter(|(t, _)| **t != draft).map(|(_, p)| p).sum();
        if total <= 0.0 {
            return (draft, true);
        }
        let mut r = self.next_f32() * total;
        for (t, p) in cand.iter().zip(&probs) {
            if *t == draft {
                continue;
            }
            r -= p;
            if r <= 0.0 {
                return (*t, false);
            }
        }
        (*cand.iter().rev().find(|t| **t != draft).unwrap(), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn logits(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 33) as f32 / (1u64 << 31) as f32) * 20.0 - 10.0
            })
            .collect()
    }

    #[test]
    fn top_k_matches_full_sort() {
        let l = logits(248_320, 9);
        for k in [1, 20, 64, 500] {
            let got = Sampler::top_k(&l, k);
            let mut all: Vec<(f32, u32)> = l.iter().enumerate().map(|(i, &v)| (v, i as u32)).collect();
            all.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            assert_eq!(got.len(), k);
            for i in 0..k {
                assert_eq!(got[i].0, all[i].0, "k={k} rank {i}");
            }
        }
    }

    #[test]
    fn sampling_leaves_logits_untouched_and_respects_top_k() {
        let mut l = logits(50_000, 3);
        let orig = l.clone();
        let mut s = Sampler::new(SamplerConfig { seed: 1, ..Default::default() });
        for t in [5u32, 77, 1234] {
            s.accept(t); // activate penalties
        }
        let top: Vec<u32> = Sampler::top_k(&l, 20).iter().map(|x| x.1).collect();
        for _ in 0..200 {
            let t = s.sample(&mut l);
            assert!(top.contains(&t) || [5, 77, 1234].contains(&t));
            assert_eq!(l, orig);
        }
        let (t, _) = s.sample_spec(&mut l, top[0]);
        assert!(top.contains(&t));
        assert_eq!(l, orig);
    }
}
