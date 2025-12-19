package service

import (
	"math"
	"math/rand"
	"sort"
)

// ProbabilisticMatcher provides uncertainty-aware matching
type ProbabilisticMatcher struct {
	rng *rand.Rand
}

// NewProbabilisticMatcher creates a new matcher
func NewProbabilisticMatcher() *ProbabilisticMatcher {
	return &ProbabilisticMatcher{
		rng: rand.New(rand.NewSource(42)),
	}
}

// ConfidenceInterval represents a Bayesian confidence interval
type ConfidenceInterval struct {
	Lower      float64 `json:"lower"`
	Upper      float64 `json:"upper"`
	Mean       float64 `json:"mean"`
	Confidence float64 `json:"confidence"` // e.g., 0.95 for 95% CI
}

// BayesianConfidence calculates confidence intervals using Beta distribution
func (pm *ProbabilisticMatcher) BayesianConfidence(matches, total int) ConfidenceInterval {
	if total == 0 {
		return ConfidenceInterval{Lower: 0, Upper: 0, Mean: 0, Confidence: 0.95}
	}

	// Beta distribution parameters (with prior)
	alpha := float64(matches) + 1.0 // Add 1 for uniform prior
	beta := float64(total-matches) + 1.0

	// Mean of Beta distribution
	mean := alpha / (alpha + beta)

	// Calculate 95% confidence interval using quantiles
	lower := pm.betaQuantile(alpha, beta, 0.025)
	upper := pm.betaQuantile(alpha, beta, 0.975)

	return ConfidenceInterval{
		Lower:      lower,
		Upper:      upper,
		Mean:       mean,
		Confidence: 0.95,
	}
}

// betaQuantile approximates Beta distribution quantile
func (pm *ProbabilisticMatcher) betaQuantile(alpha, beta, p float64) float64 {
	// Simple approximation using normal approximation
	mean := alpha / (alpha + beta)
	variance := (alpha * beta) / ((alpha + beta) * (alpha + beta) * (alpha + beta + 1))
	stddev := math.Sqrt(variance)

	// Z-score for given probability
	z := pm.normalQuantile(p)

	quantile := mean + z*stddev

	// Clamp to [0, 1]
	if quantile < 0 {
		return 0
	}
	if quantile > 1 {
		return 1
	}

	return quantile
}

// normalQuantile approximates standard normal quantile
func (pm *ProbabilisticMatcher) normalQuantile(p float64) float64 {
	// Approximation for standard normal quantile
	if p <= 0 {
		return -10
	}
	if p >= 1 {
		return 10
	}

	// Beasley-Springer-Moro algorithm (simplified)
	if p < 0.5 {
		return -pm.normalQuantile(1 - p)
	}

	t := math.Sqrt(-2 * math.Log(1-p))
	return t - (2.515517+0.802853*t+0.010328*t*t)/
		(1+1.432788*t+0.189269*t*t+0.001308*t*t*t)
}

// EnsembleMatch combines multiple matchers with weighted voting
func (pm *ProbabilisticMatcher) EnsembleMatch(scores []float64, weights []float64) float64 {
	if len(scores) == 0 {
		return 0
	}

	// If no weights provided, use equal weights
	if len(weights) == 0 || len(weights) != len(scores) {
		weights = make([]float64, len(scores))
		for i := range weights {
			weights[i] = 1.0 / float64(len(scores))
		}
	}

	// Weighted average
	weightedSum := 0.0
	totalWeight := 0.0

	for i, score := range scores {
		weightedSum += score * weights[i]
		totalWeight += weights[i]
	}

	if totalWeight == 0 {
		return 0
	}

	return weightedSum / totalWeight
}

// MonteCarloUncertainty estimates uncertainty using bootstrap sampling
func (pm *ProbabilisticMatcher) MonteCarloUncertainty(
	matchFunc func() float64,
	numSamples int,
) ConfidenceInterval {
	samples := make([]float64, numSamples)

	for i := 0; i < numSamples; i++ {
		samples[i] = matchFunc()
	}

	// Calculate statistics
	mean := 0.0
	for _, s := range samples {
		mean += s
	}
	mean /= float64(numSamples)

	// Sort for quantiles
	sortedSamples := make([]float64, numSamples)
	copy(sortedSamples, samples)
	sort.Float64s(sortedSamples)

	// 95% confidence interval
	lowerIdx := int(float64(numSamples) * 0.025)
	upperIdx := int(float64(numSamples) * 0.975)

	return ConfidenceInterval{
		Lower:      sortedSamples[lowerIdx],
		Upper:      sortedSamples[upperIdx],
		Mean:       mean,
		Confidence: 0.95,
	}
}

// CalculateMatchProbability calculates probability of a true match
func (pm *ProbabilisticMatcher) CalculateMatchProbability(
	nameSim, dataSim, contextSim float64,
	priorProbability float64,
) float64 {
	// Bayesian update: P(match|evidence) ∝ P(evidence|match) * P(match)

	// Likelihood: P(evidence|match)
	// Assume independence (simplified)
	likelihood := nameSim * dataSim * contextSim

	// Prior: P(match)
	prior := priorProbability

	// P(evidence|no match) - assume low similarity if not a match
	likelihoodNoMatch := (1 - nameSim) * (1 - dataSim) * (1 - contextSim)

	// Bayes theorem
	numerator := likelihood * prior
	denominator := numerator + likelihoodNoMatch*(1-prior)

	if denominator == 0 {
		return prior
	}

	return numerator / denominator
}
