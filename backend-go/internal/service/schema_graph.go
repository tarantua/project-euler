package service

// SchemaGraph represents the correlation graph
type SchemaGraph struct {
	Nodes []GraphNode
	Edges []GraphEdge
}

// GraphNode represents a column in the graph
type GraphNode struct {
	ID         string  `json:"id"`
	Label      string  `json:"label"`
	File       string  `json:"file"`
	Centrality float64 `json:"centrality"`
	Community  int     `json:"community"`
}

// GraphEdge represents a correlation between columns
type GraphEdge struct {
	Source string  `json:"source"`
	Target string  `json:"target"`
	Weight float64 `json:"weight"` // Confidence score
}

// GraphAnalyzer provides graph-based correlation analysis
type GraphAnalyzer struct{}

// NewGraphAnalyzer creates a new analyzer
func NewGraphAnalyzer() *GraphAnalyzer {
	return &GraphAnalyzer{}
}

// BuildSchemaGraph creates a graph from correlation results
func (ga *GraphAnalyzer) BuildSchemaGraph(correlations []SimilarityResult, file1Cols, file2Cols []string) *SchemaGraph {
	graph := &SchemaGraph{
		Nodes: []GraphNode{},
		Edges: []GraphEdge{},
	}

	// Create nodes for all columns
	for _, col := range file1Cols {
		graph.Nodes = append(graph.Nodes, GraphNode{
			ID:    "f1_" + col,
			Label: col,
			File:  "file1",
		})
	}
	for _, col := range file2Cols {
		graph.Nodes = append(graph.Nodes, GraphNode{
			ID:    "f2_" + col,
			Label: col,
			File:  "file2",
		})
	}

	// Create edges from correlations
	for _, corr := range correlations {
		if corr.Confidence > 30 { // Threshold
			graph.Edges = append(graph.Edges, GraphEdge{
				Source: "f1_" + corr.File1Column,
				Target: "f2_" + corr.File2Column,
				Weight: corr.Confidence / 100.0,
			})
		}
	}

	return graph
}

// CommunityDetection finds groups of related columns using Louvain algorithm
func (ga *GraphAnalyzer) CommunityDetection(graph *SchemaGraph) {
	// Simplified Louvain: assign communities based on edge weights
	nodeIndex := make(map[string]int)
	for i, node := range graph.Nodes {
		nodeIndex[node.ID] = i
	}

	// Initialize each node in its own community
	for i := range graph.Nodes {
		graph.Nodes[i].Community = i
	}

	// Iteratively merge communities
	improved := true
	for improved {
		improved = false

		for i := range graph.Nodes {
			bestCommunity := graph.Nodes[i].Community
			bestGain := 0.0

			// Try moving to neighbor communities
			neighbors := ga.getNeighborCommunities(graph, i)
			for community, weight := range neighbors {
				gain := weight
				if gain > bestGain {
					bestGain = gain
					bestCommunity = community
				}
			}

			if bestCommunity != graph.Nodes[i].Community {
				graph.Nodes[i].Community = bestCommunity
				improved = true
			}
		}
	}
}

// getNeighborCommunities finds communities of neighboring nodes
func (ga *GraphAnalyzer) getNeighborCommunities(graph *SchemaGraph, nodeIdx int) map[int]float64 {
	nodeID := graph.Nodes[nodeIdx].ID
	communities := make(map[int]float64)

	for _, edge := range graph.Edges {
		if edge.Source == nodeID {
			targetIdx := ga.findNodeIndex(graph, edge.Target)
			if targetIdx >= 0 {
				community := graph.Nodes[targetIdx].Community
				communities[community] += edge.Weight
			}
		} else if edge.Target == nodeID {
			sourceIdx := ga.findNodeIndex(graph, edge.Source)
			if sourceIdx >= 0 {
				community := graph.Nodes[sourceIdx].Community
				communities[community] += edge.Weight
			}
		}
	}

	return communities
}

// findNodeIndex finds the index of a node by ID
func (ga *GraphAnalyzer) findNodeIndex(graph *SchemaGraph, nodeID string) int {
	for i, node := range graph.Nodes {
		if node.ID == nodeID {
			return i
		}
	}
	return -1
}

// CalculateCentrality computes PageRank-style centrality for each node
func (ga *GraphAnalyzer) CalculateCentrality(graph *SchemaGraph) {
	n := len(graph.Nodes)
	if n == 0 {
		return
	}

	// Initialize centrality scores
	centrality := make([]float64, n)
	for i := range centrality {
		centrality[i] = 1.0 / float64(n)
	}

	// Build adjacency structure
	outgoing := make(map[int][]int)
	weights := make(map[[2]int]float64)

	for _, edge := range graph.Edges {
		srcIdx := ga.findNodeIndex(graph, edge.Source)
		tgtIdx := ga.findNodeIndex(graph, edge.Target)

		if srcIdx >= 0 && tgtIdx >= 0 {
			outgoing[srcIdx] = append(outgoing[srcIdx], tgtIdx)
			weights[[2]int{srcIdx, tgtIdx}] = edge.Weight
		}
	}

	// Power iteration
	dampingFactor := 0.85
	iterations := 20

	for iter := 0; iter < iterations; iter++ {
		newCentrality := make([]float64, n)

		for i := range newCentrality {
			newCentrality[i] = (1 - dampingFactor) / float64(n)
		}

		for src, targets := range outgoing {
			if len(targets) == 0 {
				continue
			}

			contribution := centrality[src] / float64(len(targets))
			for _, tgt := range targets {
				weight := weights[[2]int{src, tgt}]
				newCentrality[tgt] += dampingFactor * contribution * weight
			}
		}

		centrality = newCentrality
	}

	// Assign centrality scores to nodes
	for i := range graph.Nodes {
		graph.Nodes[i].Centrality = centrality[i]
	}
}

// FindTransitivePaths finds indirect relationships through intermediate columns
func (ga *GraphAnalyzer) FindTransitivePaths(graph *SchemaGraph, source, target string, maxDepth int) [][]string {
	paths := [][]string{}
	visited := make(map[string]bool)
	currentPath := []string{source}

	ga.dfsPath(graph, source, target, maxDepth, currentPath, visited, &paths)

	return paths
}

// dfsPath performs depth-first search for paths
func (ga *GraphAnalyzer) dfsPath(graph *SchemaGraph, current, target string, depth int, path []string, visited map[string]bool, paths *[][]string) {
	if current == target {
		// Found a path
		pathCopy := make([]string, len(path))
		copy(pathCopy, path)
		*paths = append(*paths, pathCopy)
		return
	}

	if depth == 0 {
		return
	}

	visited[current] = true

	// Find neighbors
	for _, edge := range graph.Edges {
		var next string
		if edge.Source == current {
			next = edge.Target
		} else if edge.Target == current {
			next = edge.Source
		} else {
			continue
		}

		if !visited[next] {
			newPath := append(path, next)
			ga.dfsPath(graph, next, target, depth-1, newPath, visited, paths)
		}
	}

	visited[current] = false
}
