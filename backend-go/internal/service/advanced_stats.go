package service

import (
	"backend-go/internal/state"
	"math"
	"strconv"
)

// AdvancedStatsCalculator provides advanced statistical correlation methods
type AdvancedStatsCalculator struct{}

// NewAdvancedStatsCalculator creates a new calculator
func NewAdvancedStatsCalculator() *AdvancedStatsCalculator {
	return &AdvancedStatsCalculator{}
}

// MutualInformation calculates mutual information between two columns
// Detects both linear and non-linear relationships
func (asc *AdvancedStatsCalculator) MutualInformation(df1, df2 *state.DataFrame, col1Idx, col2Idx int) float64 {
	// Extract values as floats
	vals1 := extractFloatValues(df1, col1Idx)
	vals2 := extractFloatValues(df2, col2Idx)

	if len(vals1) == 0 || len(vals2) == 0 {
		return 0
	}

	// Discretize continuous values into bins
	bins1 := discretize(vals1, 10)
	bins2 := discretize(vals2, 10)

	// Calculate joint and marginal probabilities
	jointProb := make(map[[2]int]float64)
	prob1 := make(map[int]float64)
	prob2 := make(map[int]float64)

	n := float64(len(bins1))

	for i := 0; i < len(bins1); i++ {
		b1 := bins1[i]
		b2 := bins2[i]

		jointProb[[2]int{b1, b2}]++
		prob1[b1]++
		prob2[b2]++
	}

	// Normalize to probabilities
	for key := range jointProb {
		jointProb[key] /= n
	}
	for key := range prob1 {
		prob1[key] /= n
	}
	for key := range prob2 {
		prob2[key] /= n
	}

	// Calculate mutual information
	mi := 0.0
	for key, pxy := range jointProb {
		px := prob1[key[0]]
		py := prob2[key[1]]

		if pxy > 0 && px > 0 && py > 0 {
			mi += pxy * math.Log2(pxy/(px*py))
		}
	}

	// Normalize to [0, 1]
	// Max MI is min(H(X), H(Y))
	h1 := entropy(prob1)
	h2 := entropy(prob2)
	maxMI := math.Min(h1, h2)

	if maxMI == 0 {
		return 0
	}

	return mi / maxMI
}

// DistanceCorrelation calculates distance correlation
// Captures all types of dependencies (linear and non-linear)
func (asc *AdvancedStatsCalculator) DistanceCorrelation(df1, df2 *state.DataFrame, col1Idx, col2Idx int) float64 {
	vals1 := extractFloatValues(df1, col1Idx)
	vals2 := extractFloatValues(df2, col2Idx)

	if len(vals1) < 5 || len(vals2) < 5 {
		return 0
	}

	n := len(vals1)

	// Calculate distance matrices
	distX := make([][]float64, n)
	distY := make([][]float64, n)

	for i := 0; i < n; i++ {
		distX[i] = make([]float64, n)
		distY[i] = make([]float64, n)

		for j := 0; j < n; j++ {
			distX[i][j] = math.Abs(vals1[i] - vals1[j])
			distY[i][j] = math.Abs(vals2[i] - vals2[j])
		}
	}

	// Double center the distance matrices
	distX = doubleCenterMatrix(distX)
	distY = doubleCenterMatrix(distY)

	// Calculate distance covariance
	dcov := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			dcov += distX[i][j] * distY[i][j]
		}
	}
	dcov = math.Sqrt(dcov / float64(n*n))

	// Calculate distance variances
	dvarX := 0.0
	dvarY := 0.0
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			dvarX += distX[i][j] * distX[i][j]
			dvarY += distY[i][j] * distY[i][j]
		}
	}
	dvarX = math.Sqrt(dvarX / float64(n*n))
	dvarY = math.Sqrt(dvarY / float64(n*n))

	// Distance correlation
	if dvarX == 0 || dvarY == 0 {
		return 0
	}

	return dcov / math.Sqrt(dvarX*dvarY)
}

// MaximalInformationCoefficient calculates MIC
// Finds complex patterns in data
func (asc *AdvancedStatsCalculator) MaximalInformationCoefficient(df1, df2 *state.DataFrame, col1Idx, col2Idx int) float64 {
	vals1 := extractFloatValues(df1, col1Idx)
	vals2 := extractFloatValues(df2, col2Idx)

	if len(vals1) < 10 || len(vals2) < 10 {
		return 0
	}

	// Simplified MIC: try different grid sizes and find max normalized MI
	maxMIC := 0.0

	// Try grid sizes from 2x2 to 5x5
	for gridX := 2; gridX <= 5; gridX++ {
		for gridY := 2; gridY <= 5; gridY++ {
			// Discretize into grid
			bins1 := discretize(vals1, gridX)
			bins2 := discretize(vals2, gridY)

			// Calculate MI for this grid
			mi := asc.calculateMIFromBins(bins1, bins2, gridX, gridY)

			// Normalize by log of grid size
			normalizedMI := mi / math.Log2(float64(minIntStats(gridX, gridY)))

			if normalizedMI > maxMIC {
				maxMIC = normalizedMI
			}
		}
	}

	return math.Min(1.0, maxMIC)
}

// Helper functions

func extractValues(df *state.DataFrame, colIdx int) []string {
	values := []string{}
	for _, row := range df.Rows {
		if colIdx < len(row) && row[colIdx] != "" {
			values = append(values, row[colIdx])
		}
	}
	return values
}

func extractFloatValues(df *state.DataFrame, colIdx int) []float64 {
	values := []float64{}
	for _, row := range df.Rows {
		if colIdx < len(row) {
			if val, err := strconv.ParseFloat(row[colIdx], 64); err == nil {
				values = append(values, val)
			}
		}
	}
	return values
}

func discretize(values []float64, numBins int) []int {
	if len(values) == 0 {
		return []int{}
	}

	// Find min and max
	minVal := values[0]
	maxVal := values[0]
	for _, v := range values {
		if v < minVal {
			minVal = v
		}
		if v > maxVal {
			maxVal = v
		}
	}

	// Create bins
	binWidth := (maxVal - minVal) / float64(numBins)
	if binWidth == 0 {
		binWidth = 1
	}

	bins := make([]int, len(values))
	for i, v := range values {
		bin := int((v - minVal) / binWidth)
		if bin >= numBins {
			bin = numBins - 1
		}
		bins[i] = bin
	}

	return bins
}

func entropy(prob map[int]float64) float64 {
	h := 0.0
	for _, p := range prob {
		if p > 0 {
			h -= p * math.Log2(p)
		}
	}
	return h
}

func doubleCenterMatrix(dist [][]float64) [][]float64 {
	n := len(dist)
	if n == 0 {
		return dist
	}

	// Calculate row means
	rowMeans := make([]float64, n)
	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < n; j++ {
			sum += dist[i][j]
		}
		rowMeans[i] = sum / float64(n)
	}

	// Calculate column means
	colMeans := make([]float64, n)
	for j := 0; j < n; j++ {
		sum := 0.0
		for i := 0; i < n; i++ {
			sum += dist[i][j]
		}
		colMeans[j] = sum / float64(n)
	}

	// Calculate grand mean
	grandMean := 0.0
	for i := 0; i < n; i++ {
		grandMean += rowMeans[i]
	}
	grandMean /= float64(n)

	// Double center
	centered := make([][]float64, n)
	for i := 0; i < n; i++ {
		centered[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			centered[i][j] = dist[i][j] - rowMeans[i] - colMeans[j] + grandMean
		}
	}

	return centered
}

func (asc *AdvancedStatsCalculator) calculateMIFromBins(bins1, bins2 []int, maxBin1, maxBin2 int) float64 {
	jointProb := make(map[[2]int]float64)
	prob1 := make(map[int]float64)
	prob2 := make(map[int]float64)

	n := float64(len(bins1))

	for i := 0; i < len(bins1); i++ {
		b1 := bins1[i]
		b2 := bins2[i]

		jointProb[[2]int{b1, b2}]++
		prob1[b1]++
		prob2[b2]++
	}

	// Normalize
	for key := range jointProb {
		jointProb[key] /= n
	}
	for key := range prob1 {
		prob1[key] /= n
	}
	for key := range prob2 {
		prob2[key] /= n
	}

	// Calculate MI
	mi := 0.0
	for key, pxy := range jointProb {
		px := prob1[key[0]]
		py := prob2[key[1]]

		if pxy > 0 && px > 0 && py > 0 {
			mi += pxy * math.Log2(pxy/(px*py))
		}
	}

	return mi
}

func minIntStats(a, b int) int {
	if a < b {
		return a
	}
	return b
}
