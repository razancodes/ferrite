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

    fn apply_penalties(&self, logits: &mut [f32]) {
        let c = &self.cfg;
        if c.presence_penalty == 0.0 && c.frequency_penalty == 0.0 && c.repetition_penalty == 1.0 {
            return;
        }
        for (&tok, &n) in &self.counts {
            let Some(l) = logits.get_mut(tok as usize) else { continue };
            if c.repetition_penalty != 1.0 {
                *l = if *l > 0.0 { *l / c.repetition_penalty } else { *l * c.repetition_penalty };
            }
            *l -= c.presence_penalty + c.frequency_penalty * n as f32;
        }
    }

    /// Speculative sampling against a deterministic draft token `draft`.
    /// Returns (token, accepted). The draft is accepted with probability p(draft)
    /// under the fully processed target distribution; otherwise a token is drawn
    /// from p with the draft removed. The output distribution is exactly `p`.
    pub fn sample_spec(&mut self, logits: &mut [f32], draft: u32) -> (u32, bool) {
        if self.cfg.temperature <= 0.0 || self.cfg.top_k == 1 {
            let t = self.sample(logits);
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

    /// The final sampling distribution after penalties, temperature, top-k,
    /// min-p and top-p: candidate ids with normalized probabilities.
    fn distribution(&mut self, logits: &mut [f32]) -> (Vec<u32>, Vec<f32>) {
        self.apply_penalties(logits);
        let c = self.cfg.clone();
        let best = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let cutoff = best - 30.0 * c.temperature;
        let mut cand: Vec<(f32, u32)> =
            logits.iter().enumerate().filter(|(_, l)| **l >= cutoff).map(|(i, l)| (*l, i as u32)).collect();
        if c.top_k > 0 && cand.len() > c.top_k {
            cand.select_nth_unstable_by(c.top_k - 1, |a, b| b.0.total_cmp(&a.0));
            cand.truncate(c.top_k);
        }
        cand.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
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

    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        self.apply_penalties(logits);
        let c = self.cfg.clone();
        let (mut best, mut bi) = (f32::NEG_INFINITY, 0usize);
        for (i, &l) in logits.iter().enumerate() {
            if l > best {
                best = l;
                bi = i;
            }
        }
        if c.temperature <= 0.0 || c.top_k == 1 {
            return bi as u32;
        }
        // candidates within e^-30 of the best (in temperature-scaled space)
        let cutoff = best - 30.0 * c.temperature;
        let mut cand: Vec<(f32, u32)> =
            logits.iter().enumerate().filter(|(_, l)| **l >= cutoff).map(|(i, l)| (*l, i as u32)).collect();
        if c.top_k > 0 && cand.len() > c.top_k {
            cand.select_nth_unstable_by(c.top_k - 1, |a, b| b.0.total_cmp(&a.0));
            cand.truncate(c.top_k);
        }
        cand.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
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
        let total: f32 = probs[..n].iter().sum();
        let mut r = self.next_f32() * total;
        for i in 0..n {
            r -= probs[i];
            if r <= 0.0 {
                return cand[i].1;
            }
        }
        cand[n - 1].1
    }
}
