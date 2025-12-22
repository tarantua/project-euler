package service

import (
	"backend-go/internal/state"
	"math"
)

// TimeSeriesAnalyzer provides time-series aware correlation
type TimeSeriesAnalyzer struct{}

// NewTimeSeriesAnalyzer creates a new analyzer
func NewTimeSeriesAnalyzer() *TimeSeriesAnalyzer {
	return &TimeSeriesAnalyzer{}
}

// LagCorrelation calculates correlation at different time lags
func (tsa *TimeSeriesAnalyzer) LagCorrelation(df1, df2 *state.DataFrame, col1Idx, col2Idx int, maxLag int) map[int]float64 {
	vals1 := extractFloatValues(df1, col1Idx)
	vals2 := extractFloatValues(df2, col2Idx)

	if len(vals1) < 10 || len(vals2) < 10 {
		return nil
	}

	lagCorrelations := make(map[int]float64)

	// Calculate correlation at different lags
	for lag := -maxLag; lag <= maxLag; lag++ {
		corr := tsa.correlationAtLag(vals1, vals2, lag)
		lagCorrelations[lag] = corr
	}

	return lagCorrelations
}

// correlationAtLag calculates Pearson correlation with a time lag
func (tsa *TimeSeriesAnalyzer) correlationAtLag(x, y []float64, lag int) float64 {
	var x1, y1 []float64

	if lag >= 0 {
		// Positive lag: y leads x
		if lag >= len(y) {
			return 0
		}
		x1 = x[:len(x)-lag]
		y1 = y[lag:]
	} else {
		// Negative lag: x leads y
		lag = -lag
		if lag >= len(x) {
			return 0
		}
		x1 = x[lag:]
		y1 = y[:len(y)-lag]
	}

	// Ensure equal length
	minLen := len(x1)
	if len(y1) < minLen {
		minLen = len(y1)
	}
	x1 = x1[:minLen]
	y1 = y1[:minLen]

	if minLen < 3 {
		return 0
	}

	return pearsonCorrelation(x1, y1)
}

// SeasonalityDetection detects periodic patterns using FFT approximation
func (tsa *TimeSeriesAnalyzer) SeasonalityDetection(df *state.DataFrame, colIdx int) float64 {
	vals := extractFloatValues(df, colIdx)

	if len(vals) < 20 {
		return 0
	}

	// Simple seasonality detection: check for repeating patterns
	// Calculate autocorrelation at different lags
	maxLag := min(len(vals)/2, 30)
	autocorrs := make([]float64, maxLag)

	for lag := 1; lag < maxLag; lag++ {
		autocorrs[lag] = tsa.autocorrelation(vals, lag)
	}

	// Find peaks in autocorrelation (indicates seasonality)
	maxAutocorr := 0.0
	for _, ac := range autocorrs {
		if ac > maxAutocorr {
			maxAutocorr = ac
		}
	}

	return maxAutocorr
}

// autocorrelation calculates autocorrelation at a given lag
func (tsa *TimeSeriesAnalyzer) autocorrelation(vals []float64, lag int) float64 {
	if lag >= len(vals) {
		return 0
	}

	x1 := vals[:len(vals)-lag]
	x2 := vals[lag:]

	return pearsonCorrelation(x1, x2)
}

// TrendAnalysis detects linear trends in data
func (tsa *TimeSeriesAnalyzer) TrendAnalysis(df *state.DataFrame, colIdx int) (slope float64, rsquared float64) {
	vals := extractFloatValues(df, colIdx)

	if len(vals) < 3 {
		return 0, 0
	}

	// Simple linear regression: y = mx + b
	n := float64(len(vals))

	// Create time index
	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	sumX2 := 0.0
	sumY2 := 0.0

	for i, y := range vals {
		x := float64(i)
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
		sumY2 += y * y
	}

	// Calculate slope
	numerator := n*sumXY - sumX*sumY
	denominator := n*sumX2 - sumX*sumX

	if denominator == 0 {
		return 0, 0
	}

	slope = numerator / denominator

	// Calculate R-squared
	meanY := sumY / n
	ssTotal := 0.0
	ssResidual := 0.0

	for i, y := range vals {
		x := float64(i)
		predicted := slope*x + (meanY - slope*(sumX/n))
		ssTotal += (y - meanY) * (y - meanY)
		ssResidual += (y - predicted) * (y - predicted)
	}

	if ssTotal == 0 {
		return slope, 0
	}

	rsquared = 1 - (ssResidual / ssTotal)

	return slope, rsquared
}

// Helper functions

func pearsonCorrelation(x, y []float64) float64 {
	if len(x) != len(y) || len(x) == 0 {
		return 0
	}

	n := float64(len(x))

	sumX := 0.0
	sumY := 0.0
	sumXY := 0.0
	sumX2 := 0.0
	sumY2 := 0.0

	for i := 0; i < len(x); i++ {
		sumX += x[i]
		sumY += y[i]
		sumXY += x[i] * y[i]
		sumX2 += x[i] * x[i]
		sumY2 += y[i] * y[i]
	}

	numerator := n*sumXY - sumX*sumY
	denominator := math.Sqrt((n*sumX2 - sumX*sumX) * (n*sumY2 - sumY*sumY))

	if denominator == 0 {
		return 0
	}

	return numerator / denominator
}
