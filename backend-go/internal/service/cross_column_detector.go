package service

import (
	"backend-go/internal/state"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// CrossColumnDetector identifies relationships between multiple columns
type CrossColumnDetector struct{}

// NewCrossColumnDetector creates a new detector
func NewCrossColumnDetector() *CrossColumnDetector {
	return &CrossColumnDetector{}
}

// CompositeKey represents a combination of columns that uniquely identifies rows
type CompositeKey struct {
	Columns     []string `json:"columns"`
	Uniqueness  float64  `json:"uniqueness"`
	IsCandidate bool     `json:"is_candidate"`
}

// DerivedColumn represents a column that is derived from other columns
type DerivedColumn struct {
	TargetColumn  string   `json:"target_column"`
	SourceColumns []string `json:"source_columns"`
	Relationship  string   `json:"relationship"` // "concatenation", "sum", "product", "ratio"
	Confidence    float64  `json:"confidence"`
}

// DetectCompositeKeys finds column combinations that uniquely identify rows
func (ccd *CrossColumnDetector) DetectCompositeKeys(df *state.DataFrame) []CompositeKey {
	results := []CompositeKey{}

	// Test 2-column combinations
	for i := 0; i < len(df.Headers); i++ {
		for j := i + 1; j < len(df.Headers); j++ {
			uniqueness := ccd.calculateCompositeUniqueness(df, []int{i, j})

			composite := CompositeKey{
				Columns:     []string{df.Headers[i], df.Headers[j]},
				Uniqueness:  uniqueness,
				IsCandidate: uniqueness > 0.95,
			}

			if composite.IsCandidate {
				results = append(results, composite)
			}
		}
	}

	// Test 3-column combinations (only if we have enough columns)
	if len(df.Headers) >= 3 && len(df.Headers) <= 10 {
		for i := 0; i < len(df.Headers); i++ {
			for j := i + 1; j < len(df.Headers); j++ {
				for k := j + 1; k < len(df.Headers); k++ {
					uniqueness := ccd.calculateCompositeUniqueness(df, []int{i, j, k})

					composite := CompositeKey{
						Columns:     []string{df.Headers[i], df.Headers[j], df.Headers[k]},
						Uniqueness:  uniqueness,
						IsCandidate: uniqueness > 0.95,
					}

					if composite.IsCandidate {
						results = append(results, composite)
					}
				}
			}
		}
	}

	return results
}

// calculateCompositeUniqueness calculates uniqueness ratio for column combination
func (ccd *CrossColumnDetector) calculateCompositeUniqueness(df *state.DataFrame, colIndices []int) float64 {
	if len(df.Rows) == 0 {
		return 0
	}

	// Create composite values
	compositeValues := make(map[string]bool)

	for _, row := range df.Rows {
		// Build composite key
		parts := []string{}
		for _, idx := range colIndices {
			if idx < len(row) {
				parts = append(parts, row[idx])
			}
		}

		compositeKey := strings.Join(parts, "||")
		compositeValues[compositeKey] = true
	}

	return float64(len(compositeValues)) / float64(len(df.Rows))
}

// DetectDerivedColumns identifies columns derived from other columns
func (ccd *CrossColumnDetector) DetectDerivedColumns(df *state.DataFrame) []DerivedColumn {
	results := []DerivedColumn{}

	// Check for string concatenations
	concatResults := ccd.detectConcatenations(df)
	results = append(results, concatResults...)

	// Check for arithmetic relationships (sum, product, ratio)
	arithResults := ccd.detectArithmetic(df)
	results = append(results, arithResults...)

	return results
}

// detectConcatenations finds columns that are concatenations of others
func (ccd *CrossColumnDetector) detectConcatenations(df *state.DataFrame) []DerivedColumn {
	results := []DerivedColumn{}

	// For each column, check if it's a concatenation of two others
	for targetIdx := 0; targetIdx < len(df.Headers); targetIdx++ {
		for i := 0; i < len(df.Headers); i++ {
			if i == targetIdx {
				continue
			}
			for j := i + 1; j < len(df.Headers); j++ {
				if j == targetIdx {
					continue
				}

				confidence := ccd.testConcatenation(df, targetIdx, i, j)

				if confidence > 0.8 {
					results = append(results, DerivedColumn{
						TargetColumn:  df.Headers[targetIdx],
						SourceColumns: []string{df.Headers[i], df.Headers[j]},
						Relationship:  "concatenation",
						Confidence:    confidence,
					})
				}
			}
		}
	}

	return results
}

// testConcatenation tests if target = source1 + source2
func (ccd *CrossColumnDetector) testConcatenation(df *state.DataFrame, targetIdx, src1Idx, src2Idx int) float64 {
	matches := 0
	total := 0

	sampleSize := 50
	if len(df.Rows) < sampleSize {
		sampleSize = len(df.Rows)
	}

	for i := 0; i < sampleSize; i++ {
		row := df.Rows[i]

		if targetIdx >= len(row) || src1Idx >= len(row) || src2Idx >= len(row) {
			continue
		}

		target := strings.ToLower(strings.TrimSpace(row[targetIdx]))
		src1 := strings.ToLower(strings.TrimSpace(row[src1Idx]))
		src2 := strings.ToLower(strings.TrimSpace(row[src2Idx]))

		if target == "" || src1 == "" || src2 == "" {
			continue
		}

		total++

		// Test various concatenation patterns
		if target == src1+src2 ||
			target == src1+" "+src2 ||
			target == src1+","+src2 ||
			target == src1+", "+src2 ||
			target == src2+" "+src1 ||
			target == src2+", "+src1 {
			matches++
		}
	}

	if total == 0 {
		return 0
	}

	return float64(matches) / float64(total)
}

// detectArithmetic finds columns with arithmetic relationships
func (ccd *CrossColumnDetector) detectArithmetic(df *state.DataFrame) []DerivedColumn {
	results := []DerivedColumn{}

	numericCols := df.GetNumericColumnIndices()
	numericIndices := []int{}
	for idx, isNumeric := range numericCols {
		if isNumeric {
			numericIndices = append(numericIndices, idx)
		}
	}

	// Need at least 3 numeric columns to detect relationships
	if len(numericIndices) < 3 {
		return results
	}

	// Test sum relationships: target = src1 + src2
	for _, targetIdx := range numericIndices {
		for i, src1Idx := range numericIndices {
			if src1Idx == targetIdx {
				continue
			}
			for j := i + 1; j < len(numericIndices); j++ {
				src2Idx := numericIndices[j]
				if src2Idx == targetIdx {
					continue
				}

				confidence := ccd.testSum(df, targetIdx, src1Idx, src2Idx)
				if confidence > 0.9 {
					results = append(results, DerivedColumn{
						TargetColumn:  df.Headers[targetIdx],
						SourceColumns: []string{df.Headers[src1Idx], df.Headers[src2Idx]},
						Relationship:  "sum",
						Confidence:    confidence,
					})
				}
			}
		}
	}

	// Test product relationships: target = src1 * src2
	for _, targetIdx := range numericIndices {
		for i, src1Idx := range numericIndices {
			if src1Idx == targetIdx {
				continue
			}
			for j := i + 1; j < len(numericIndices); j++ {
				src2Idx := numericIndices[j]
				if src2Idx == targetIdx {
					continue
				}

				confidence := ccd.testProduct(df, targetIdx, src1Idx, src2Idx)
				if confidence > 0.9 {
					results = append(results, DerivedColumn{
						TargetColumn:  df.Headers[targetIdx],
						SourceColumns: []string{df.Headers[src1Idx], df.Headers[src2Idx]},
						Relationship:  "product",
						Confidence:    confidence,
					})
				}
			}
		}
	}

	return results
}

// testSum tests if target ≈ src1 + src2
func (ccd *CrossColumnDetector) testSum(df *state.DataFrame, targetIdx, src1Idx, src2Idx int) float64 {
	matches := 0
	total := 0

	for _, row := range df.Rows {
		if targetIdx >= len(row) || src1Idx >= len(row) || src2Idx >= len(row) {
			continue
		}

		target, err1 := strconv.ParseFloat(row[targetIdx], 64)
		src1, err2 := strconv.ParseFloat(row[src1Idx], 64)
		src2, err3 := strconv.ParseFloat(row[src2Idx], 64)

		if err1 != nil || err2 != nil || err3 != nil {
			continue
		}

		total++

		// Check if target ≈ src1 + src2 (within 1% tolerance)
		expected := src1 + src2
		if math.Abs(target-expected) < math.Abs(expected)*0.01 {
			matches++
		}
	}

	if total == 0 {
		return 0
	}

	return float64(matches) / float64(total)
}

// testProduct tests if target ≈ src1 * src2
func (ccd *CrossColumnDetector) testProduct(df *state.DataFrame, targetIdx, src1Idx, src2Idx int) float64 {
	matches := 0
	total := 0

	for _, row := range df.Rows {
		if targetIdx >= len(row) || src1Idx >= len(row) || src2Idx >= len(row) {
			continue
		}

		target, err1 := strconv.ParseFloat(row[targetIdx], 64)
		src1, err2 := strconv.ParseFloat(row[src1Idx], 64)
		src2, err3 := strconv.ParseFloat(row[src2Idx], 64)

		if err1 != nil || err2 != nil || err3 != nil {
			continue
		}

		total++

		// Check if target ≈ src1 * src2 (within 1% tolerance)
		expected := src1 * src2
		if expected == 0 {
			continue
		}
		if math.Abs(target-expected) < math.Abs(expected)*0.01 {
			matches++
		}
	}

	if total == 0 {
		return 0
	}

	return float64(matches) / float64(total)
}

// BuildDependencyGraph creates a graph of column dependencies
func (ccd *CrossColumnDetector) BuildDependencyGraph(derivedCols []DerivedColumn) string {
	// Simple text representation of dependency graph
	graph := "Column Dependency Graph:\n"

	for _, derived := range derivedCols {
		sources := strings.Join(derived.SourceColumns, " + ")
		graph += fmt.Sprintf("  %s = %s (%s, %.0f%% confidence)\n",
			derived.TargetColumn, sources, derived.Relationship, derived.Confidence*100)
	}

	return graph
}
