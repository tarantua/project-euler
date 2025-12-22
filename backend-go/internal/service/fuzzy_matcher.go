package service

import (
	"hash/fnv"
	"math"
	"strings"
	"unicode"
)

// FuzzyMatcher provides fast approximate string matching
type FuzzyMatcher struct {
	lshBuckets   map[uint64][]string // LSH buckets for fast lookup
	soundexCache map[string]string   // Cache for soundex codes
}

// NewFuzzyMatcher creates a new fuzzy matcher
func NewFuzzyMatcher() *FuzzyMatcher {
	return &FuzzyMatcher{
		lshBuckets:   make(map[uint64][]string),
		soundexCache: make(map[string]string),
	}
}

// LSHMatch performs locality-sensitive hashing for fast approximate matching
func (fm *FuzzyMatcher) LSHMatch(query string, candidates []string, threshold float64) []string {
	results := []string{}

	// Generate MinHash signature for query
	queryHash := fm.minHash(query)

	// Find candidates in same LSH bucket
	bucket := fm.lshBuckets[queryHash]

	// For each candidate in bucket, verify with actual similarity
	for _, candidate := range bucket {
		sim := fm.jaccardSimilarity(query, candidate)
		if sim >= threshold {
			results = append(results, candidate)
		}
	}

	return results
}

// minHash generates a MinHash signature for LSH
func (fm *FuzzyMatcher) minHash(s string) uint64 {
	// Generate character 3-grams
	grams := fm.generateNGrams(s, 3)

	// Hash each gram and take minimum
	minHashVal := uint64(math.MaxUint64)

	for _, gram := range grams {
		h := fnv.New64a()
		h.Write([]byte(gram))
		hashVal := h.Sum64()

		if hashVal < minHashVal {
			minHashVal = hashVal
		}
	}

	return minHashVal
}

// generateNGrams creates character n-grams
func (fm *FuzzyMatcher) generateNGrams(s string, n int) []string {
	s = strings.ToLower(s)
	grams := []string{}

	if len(s) < n {
		return []string{s}
	}

	for i := 0; i <= len(s)-n; i++ {
		grams = append(grams, s[i:i+n])
	}

	return grams
}

// jaccardSimilarity calculates Jaccard similarity of character n-grams
func (fm *FuzzyMatcher) jaccardSimilarity(s1, s2 string) float64 {
	grams1 := fm.generateNGrams(s1, 3)
	grams2 := fm.generateNGrams(s2, 3)

	set1 := make(map[string]bool)
	set2 := make(map[string]bool)

	for _, g := range grams1 {
		set1[g] = true
	}
	for _, g := range grams2 {
		set2[g] = true
	}

	// Calculate intersection
	intersection := 0
	for g := range set1 {
		if set2[g] {
			intersection++
		}
	}

	// Calculate union
	union := len(set1) + len(set2) - intersection

	if union == 0 {
		return 0
	}

	return float64(intersection) / float64(union)
}

// PhoneticMatch uses phonetic algorithms for name matching
func (fm *FuzzyMatcher) PhoneticMatch(s1, s2 string) float64 {
	// Get Soundex codes
	soundex1 := fm.Soundex(s1)
	soundex2 := fm.Soundex(s2)

	// Exact match on soundex
	if soundex1 == soundex2 {
		return 1.0
	}

	// Also try Metaphone for better accuracy
	metaphone1 := fm.Metaphone(s1)
	metaphone2 := fm.Metaphone(s2)

	if metaphone1 == metaphone2 {
		return 0.9
	}

	return 0.0
}

// Soundex implements the Soundex phonetic algorithm
func (fm *FuzzyMatcher) Soundex(s string) string {
	// Check cache
	if cached, ok := fm.soundexCache[s]; ok {
		return cached
	}

	s = strings.ToUpper(s)
	if len(s) == 0 {
		return "0000"
	}

	// Keep first letter
	result := string(s[0])

	// Soundex mapping
	mapping := map[rune]rune{
		'B': '1', 'F': '1', 'P': '1', 'V': '1',
		'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
		'D': '3', 'T': '3',
		'L': '4',
		'M': '5', 'N': '5',
		'R': '6',
	}

	prevCode := '0'
	for _, char := range s[1:] {
		if code, ok := mapping[char]; ok {
			if code != prevCode {
				result += string(code)
				prevCode = code
			}
		} else {
			prevCode = '0'
		}

		if len(result) >= 4 {
			break
		}
	}

	// Pad with zeros
	for len(result) < 4 {
		result += "0"
	}

	soundex := result[:4]
	fm.soundexCache[s] = soundex
	return soundex
}

// Metaphone implements a simplified Metaphone algorithm
func (fm *FuzzyMatcher) Metaphone(s string) string {
	s = strings.ToUpper(s)
	if len(s) == 0 {
		return ""
	}

	result := ""

	// Remove non-letters
	cleaned := ""
	for _, r := range s {
		if unicode.IsLetter(r) {
			cleaned += string(r)
		}
	}

	if len(cleaned) == 0 {
		return ""
	}

	// Simplified Metaphone rules
	for i := 0; i < len(cleaned); i++ {
		char := cleaned[i]

		switch char {
		case 'A', 'E', 'I', 'O', 'U':
			if i == 0 {
				result += string(char)
			}
		case 'B':
			if i == len(cleaned)-1 && i > 0 && cleaned[i-1] == 'M' {
				// Silent B after M at end
			} else {
				result += "B"
			}
		case 'C':
			if i+1 < len(cleaned) && cleaned[i+1] == 'H' {
				result += "X"
				i++
			} else {
				result += "K"
			}
		case 'D':
			result += "T"
		case 'G':
			if i+1 < len(cleaned) && cleaned[i+1] == 'H' {
				result += "K"
				i++
			} else {
				result += "K"
			}
		case 'H':
			// Keep H if at start or after vowel
			if i == 0 {
				result += "H"
			}
		case 'K':
			result += "K"
		case 'P':
			if i+1 < len(cleaned) && cleaned[i+1] == 'H' {
				result += "F"
				i++
			} else {
				result += "P"
			}
		case 'Q':
			result += "K"
		case 'S':
			if i+1 < len(cleaned) && cleaned[i+1] == 'H' {
				result += "X"
				i++
			} else {
				result += "S"
			}
		case 'T':
			if i+1 < len(cleaned) && cleaned[i+1] == 'H' {
				result += "0" // TH sound
				i++
			} else {
				result += "T"
			}
		case 'V':
			result += "F"
		case 'W', 'Y':
			if i == 0 {
				result += string(char)
			}
		case 'X':
			result += "KS"
		case 'Z':
			result += "S"
		default:
			result += string(char)
		}
	}

	return result
}

// FastFuzzyMatch combines LSH and phonetic for optimal speed and accuracy
func (fm *FuzzyMatcher) FastFuzzyMatch(s1, s2 string, threshold float64) float64 {
	// Quick exact match check
	if strings.EqualFold(s1, s2) {
		return 1.0
	}

	// Try phonetic match first (fastest)
	phoneticSim := fm.PhoneticMatch(s1, s2)
	if phoneticSim > 0 {
		return phoneticSim
	}

	// Fall back to Jaccard similarity
	return fm.jaccardSimilarity(s1, s2)
}

// IndexForLSH indexes strings into LSH buckets for fast lookup
func (fm *FuzzyMatcher) IndexForLSH(strings []string) {
	fm.lshBuckets = make(map[uint64][]string)

	for _, s := range strings {
		hash := fm.minHash(s)
		fm.lshBuckets[hash] = append(fm.lshBuckets[hash], s)
	}
}
