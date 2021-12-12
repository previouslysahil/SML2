//
//  ComputationalGraph.swift
//
//
//  Created by Sahil Srivastava on 11/27/21.
//

import Foundation
import SwiftUI

// Need to make a new class that can handle matrix + matrix, scalar + matrix, scalar + scalar, [matrix] + [matrix], [matrix] + scalar, [matrix] + matrix

// MARK: SMLGraph
class SMLGraph {
    
    // Nodes sorted in high to low dependencies order
    // (root is first element AKA most dependent)
    var nodes: [SMLUnit] = []
    // FOR PARALLEL
    // Keys for each depth, less dependencies as depth increases
    // (root is at first depth AKA most dependent)
    // This could potentially be optimized to use pointers? instead of actual object ref
    var nodes_parallel: [Int: [(node: SMLUnit, parent: SMLUnit?, gradIdx: Int)]] = [:]
    // Parallel toggle
    var parallel: Bool = false
    // Seed
    var seed: Tensor?
    
    // MARK: Init
    init(_ root: SMLUnit, seed: Tensor, parallel: Bool = false) {
        set(root, seed: seed, parallel: parallel)
    }
    
    init() {}
    
    func set(_ root: SMLUnit, seed: Tensor, parallel: Bool = false) {
        self.parallel = parallel
        self.seed = seed
        // Sort graph based on paralleization
        if parallel {
            topology_sort_parallel(root)
        } else {
            topology_sort(root)
        }
    }
    
    // MARK: Topology Sort (BFS)
    // DOES NOT CHECK FOR CYCLES
    private func topology_sort(_ root: SMLUnit) {
        // Make empty queue
        var queue = [SMLUnit]()
        // Add root to our queue
        queue.append(root)
        // Dequeue until no more children
        while !queue.isEmpty {
            // Remove first element in our queue
            let curr = queue.removeFirst()
            // Add to our nodes
            nodes.append(curr)
            // Queue curr's children
            for child in curr.inputs {
                queue.append(child)
            }
        }
    }
    
    // MARK: Topology Sort (BFS) PARALLEL
    // DOES NOT CHECK FOR CYCLES
    private func topology_sort_parallel(_ root: SMLUnit) {
        // Make empty queue
        var queue = [(node: SMLUnit, depth: Int, parent: SMLUnit?, gradIdx: Int)]()
        // Add root to our queue
        queue.append((root, 1, nil, -1))
        // Dequeue until no more children
        while !queue.isEmpty {
            // Remove first element in our queue
            let curr = queue.removeFirst()
            // Add to our nodes
            if nodes_parallel[curr.depth] == nil {
                // First node so make array
                nodes_parallel[curr.depth] = [(curr.node, curr.parent, curr.gradIdx)]
            } else {
                // ith node so append to existing array
                nodes_parallel[curr.depth]!.append((curr.node, curr.parent, curr.gradIdx))
            }
            // Queue curr's children
            var i = 0
            for child in curr.node.inputs {
                queue.append((child, curr.depth + 1, curr.node, i))
                i += 1
            }
        }
    }
    
    // MARK: Forward Exposed
    @discardableResult
    func fwd() -> Tensor {
        // Run fwd based on paralleization
        if parallel {
            return forward_parallel()
        } else {
            return forward()
        }
    }
    
    // MARK: Forward
    private func forward() -> Tensor {
        // Set tracking if we already forwarded one of the nodes
        var forwarded = Set<SMLUnit>()
        // Go through all nodes starting at lowest dependecy and forward
        for node in nodes.reversed() {
            // check since the same node dependency may have been duplicated
            if !forwarded.contains(node) {
                node.forward()
                forwarded.insert(node)
            }
        }
        return nodes.first!.out!
    }
    
    // MARK: Forward PARALLEL
    private func forward_parallel() -> Tensor {
        // Set tracking if we already forwarded one of the nodes
        var forwarded = Set<SMLUnit>()
        // Go through all nodes starting at lowest dependecy and forward
        // This sorted keys could be optimized since it is used in forward_parallel and backward_bfs_parallel
        for depth in nodes_parallel.keys.sorted().reversed() {
            // Make our queue for nodes to be forwarded
            var queue = [SMLUnit]()
            // Queue nodes that should be forwarded
            for (node, _, _) in nodes_parallel[depth]! {
                // check since the same node dependency may have been forwarded at a lower depth
                if !forwarded.contains(node) {
                    // Queue this node
                    queue.append(node)
                    // Insert it in forwarded
                    forwarded.insert(node)
                }
            }
            // Concurrently forwarded our queued nodes
            DispatchQueue.concurrentPerform(iterations: queue.count) { i in
                queue[i].forward()
            }
        }
        // dict[1] should always be just our root node
        return nodes_parallel[1]!.first!.node.out!
    }
    
    // MARK: Backward Exposed
    @discardableResult
    func bwd() -> [SMLUnit: Tensor] {
        // Run bwd based on paralleization
        if parallel {
            return backward_parallel(seed: seed!)
        } else {
            return backward(seed: seed!)
        }
    }
    
    // MARK: Backward
    private func backward(seed: Tensor) -> [SMLUnit: Tensor] {
        // Make set to contain visited
        var grads = [SMLUnit: Tensor]()
        // BFS from root and propagate gradient backwards
        backward_bfs(seed: seed, grads: &grads)
        return grads
    }
    
    // MARK: Backward BFS
    // DOES NOT CHECK FOR CYCLES
    private func backward_bfs(seed: Tensor, grads: inout [SMLUnit: Tensor]) {
        // Make empty queue
        var queue = [(SMLUnit, Tensor)]()
        // Add our first node (root AKA most dependent) and the first gradient (usually 1) to queue
        queue.append((nodes.first!, seed))
        // BFS until queue is empty
        while !queue.isEmpty {
            // Pop our first item in the queue
            let (curr, dOut) = queue.removeFirst()
            // Add curr to our visited
            if grads[curr] != nil {
                // Increment gradient if already contributing to final rate of change
                grads[curr]! = grads[curr]! + dOut
            } else {
                // Set gradient if not already contributing to final rate of change
                grads[curr] = dOut
            }
            // Backpropagate dOut for curr and then recursively go through curr's childs
            curr.backward(dOut: dOut)
            // inputs and gradients will (should) always be perfect arrays since each gradient corresponds to each child
            for (child, grad) in zip(curr.inputs, curr.grads) {
                // Add this child to queue so we can continue our BFS backpropagation
                queue.append((child, grad))
            }
        }
    }
    
    // MARK: Backward PARALLEL
    private func backward_parallel(seed: Tensor) -> [SMLUnit: Tensor] {
        // Make set to contain visited
        var grads = [SMLUnit: Tensor]()
        // BFS from root and propagate gradient backwards
        backward_bfs_parallel(seed: seed, grads: &grads)
        return grads
    }
    
    // MARK: Backward BFS PARALLEL
    // DOES NOT CHECK FOR CYCLES
    private func backward_bfs_parallel(seed: Tensor, grads: inout [SMLUnit: Tensor]) {
        // Iterate from our lowest depth (lowest dependency)
        // This sorted keys could be optimized since it is used in forward_parallel and backward_bfs_parallel
        for depth in nodes_parallel.keys.sorted() {
            // Special solution if first depth (only root, wrt to itself)
            if depth == 1 {
                // Get our root
                let (root, _, _) = nodes_parallel[depth]!.first!
                // roots dOut will always be our seed (usually 1)
                root.backward(dOut: seed)
                // Add root to our visited and set gradient
                grads[root] = seed
            } else {
                // Make our queues
                var queues = [[(node: SMLUnit, dOut: Tensor)]]()
                // For each node at each depth calculate it's chain ruled gradient
                for (node, parent, idx) in nodes_parallel[depth]! {
                    // Find this nodes dOut through it parents gradients
                    let dOut = parent!.grads[idx]
                    // Entered check to see if we need to make a new queue
                    var entered = false
                    // Check all our queues for no node duplicates
                    for i in 0..<queues.count {
                        // Queue this node to be forwarded if no duplicates in this queue
                        // This contains could potentially be optimized since it's O(n) vs a Set contains O(1)
                        if !queues[i].contains(where: { $0.node == node }) {
                            // Queue node and dOut
                            queues[i].append((node, dOut))
                            // Set entered check
                            entered = true
                            break
                        }
                    }
                    // Make a new queue with this node if entered not checked
                    if entered == false {
                        // Queue node and dOut in new queue
                        queues.append([(node, dOut)])
                    }
                    // Add node to our visited
                    if grads[node] != nil {
                        // Increment gradient if already contributing to final rate of change
                        grads[node]! = grads[node]! + dOut
                    } else {
                        // Set gradient if not already contributing to final rate of change
                        grads[node] = dOut
                    }
                }
                // Go through each queue which will contain no duplicate nodes
                for queue in queues {
                    // Dispatch concurrently since we have no duplicates
                    DispatchQueue.concurrentPerform(iterations: queue.count) { i in
                        // Break tuple into node and dOut
                        let (node, dOut) = queue[i]
                        // Backward dOut for this node
                        node.backward(dOut: dOut)
                    }
                }
            }
        }
    }
}

// MARK: SMLNode
class SMLUnit: Hashable {
    
    // Hashable conformance
    static func == (lhs: SMLUnit, rhs: SMLUnit) -> Bool {
        return ObjectIdentifier(lhs) == ObjectIdentifier(rhs)
    }
    
    func hash(into hasher: inout Hasher) { return hasher.combine(ObjectIdentifier(self)) }
    
    var inputs: [SMLUnit]
    var out: Tensor?
    // This will contain the chain ruled gradients for the inputs to the unit
    var grads: [Tensor] = []
    var tag: String
    
    init(_ out: Tensor? = nil, inputs: [SMLUnit] = [], tag: String = "") {
        self.inputs = inputs
        self.out = out
        self.tag = tag
    }
    
    init(_ out: Double, inputs: [SMLUnit] = [], tag: String = "") {
        self.inputs = inputs
        self.out = Tensor(out)
        self.tag = tag
    }
    
    func forward() { }
    // dOut is the gradient for this node wrt to the root of the graph
    func backward(dOut: Tensor?) {}
    func add(to graph: SMLGraph) -> Self {
        graph.nodes.append(self)
        return self
    }
}

// MARK: SMLBinary
class SMLBinary: SMLUnit {
    
    init(_ a: SMLUnit, _ b: SMLUnit, tag: String = "") {
        super.init(inputs: [a, b], tag: tag)
    }
}

// MARK: SMLUnary
class SMLUnary: SMLUnit {
    
    init(_ a: SMLUnit, tag: String = "") {
        super.init(inputs: [a], tag: tag)
    }
}

// MARK: SMLAdd
class SMLAdd: SMLBinary {
    
    override func forward() {
        out = inputs[0].out! + inputs[1].out!
    }
    
    override func backward(dOut: Tensor?) {
        // Clarify a and b
        let a = inputs[0].out!
        let b = inputs[1].out!
        // Gradient for input_nodes[0] AKA a
        var gradA = dOut!
        var axis = 0
        // Assumes a.shape.count == dOut!.shape.count
        // Condense gradients for potential vector * matrix math
        for dim in a.shape {
            if dim == 1 { gradA = gradA.sum(axis: axis, keepDim: true) }
            axis += 1
        }
        grads.append(gradA)
        // Gradient for input_nodes[1] AKA b
        var gradB = dOut!
        axis = 0
        // Assumes b.shape.count == dOut!.shape.count
        // Condense gradients for potential vector * matrix math
        for dim in b.shape {
            if dim == 1 { gradB = gradB.sum(axis: axis, keepDim: true) }
            axis += 1
        }
        grads.append(gradB)
    }
}

// MARK: SMLMul
class SMLMul: SMLBinary {
    
    override func forward() {
        out = inputs[0].out! * inputs[1].out!
    }
    
    override func backward(dOut: Tensor?) {
        // Clarify a and b
        let a = inputs[0].out!
        let b = inputs[1].out!
        // Gradient for input_nodes[0] AKA a
        var gradA = dOut! * b
        var axis = 0
        // Assumes a.shape.count == dOut!.shape.count
        // Condense gradients for potential vector * matrix math
        for dim in a.shape {
            if dim == 1 { gradA = gradA.sum(axis: axis, keepDim: true) }
            axis += 1
        }
        grads.append(gradA)
        // Gradient for input_nodes[1] AKA b
        var gradB = dOut! * a
        axis = 0
        // Assumes b.shape.count == dOut!.shape.count
        // Condense gradients for potential vector * matrix math
        for dim in b.shape {
            if dim == 1 { gradB = gradB.sum(axis: axis, keepDim: true) }
            axis += 1
        }
        grads.append(gradB)
    }
}

// MARK: SMLMatMul
class SMLMatMul: SMLBinary {
    
    override func forward() {
        out = inputs[0].out! <*> inputs[1].out!
    }
    
    override func backward(dOut: Tensor?) {
        // Clarify a and b
        let a = inputs[0].out!
        let b = inputs[1].out!
        // Gradient for input_nodes[0] aka a
        let gradA = b.transpose()
        grads.append(dOut! <*> gradA)
        // Gradient for input_nodes[1] aka b
        let gradB = a.transpose()
        grads.append(gradB <*> dOut!)
    }
}

// MARK: SMLInv
class SMLInv: SMLUnary {
    
    override func forward() {
        out = 1.0 / inputs[0].out!
    }
    
    override func backward(dOut: Tensor?) {
        // Clarify a
        let a = inputs[0].out!
        // Gradient for input_nodes[0] AKA a
        let gradA = -1.0 / a.pow(2.0)
        grads.append(dOut! * gradA)
    }
}

// MARK: SMLNeg
class SMLNeg: SMLUnary {
    
    override func forward() {
        out = -inputs[0].out!
    }
    
    override func backward(dOut: Tensor?) {
        // Gradient for input_nodes[0] AKA a
        let gradA = -1.0
        grads.append(dOut! * gradA)
    }
}

// MARK: SMLSin
class SMLSin: SMLUnary {
    
    override func forward() {
        out = inputs[0].out!.sin()
    }
    
    override func backward(dOut: Tensor?) {
        // Clarify a
        let a = inputs[0].out!
        // Gradient for input_nodes[0] AKA a
        let gradA = a.cos()
        grads.append(dOut! * gradA)
    }
}

// MARK: SMLExp
class SMLExp: SMLUnary {
    
    override func forward() {
        out = inputs[0].out!.exp()
    }
    
    override func backward(dOut: Tensor?) {
        // Gradient for input_nodes[0] AKA a
        let gradA = out!
        grads.append(dOut! * gradA)
    }
}

// MARK: SMLLog
class SMLLog: SMLUnary {
    
    override func forward() {
        out = (inputs[0].out! + 0.00000001).log()
    }
    
    override func backward(dOut: Tensor?) {
        // Clarify a
        let a = inputs[0].out!
        // Gradient for input_nodes[0] AKA a
        let gradA = 1.0 / a
        grads.append(dOut! * gradA)
    }
}

// MARK: SMLSquare
class SMLSquare: SMLUnary {
    
    override func forward() {
        out = inputs[0].out!.pow(2)
    }
    
    override func backward(dOut: Tensor?) {
        // Clarify a
        let a = inputs[0].out!
        // Gradient for input_nodes[0] AKA a
        let gradA = 2.0 * a
        grads.append(dOut! * gradA)
    }
}

// MARK: SMLPow
class SMLPow: SMLUnary {
    
    var p: Double?
    
    override func forward() {
        out = inputs[0].out!.pow(p!)
    }
    
    override func backward(dOut: Tensor?) {
        // Clarify a
        let a = inputs[0].out!
        // Gradient for input_nodes[0] AKA a
        let gradA = p! * a.pow(p! - 1)
        grads.append(dOut! * gradA)
    }
}

// MARK: SMLSum
class SMLSum: SMLUnary {
    
    override func forward() {
        out = Tensor(inputs[0].out!.sum())
    }
    
    override func backward(dOut: Tensor?) {
        precondition(dOut!.shape.count == 0)
        // Clarify a
        let a = inputs[0].out!
        // Gradient for input_nodes[0] AKA a
        let gradA = Tensor(shape: a.shape, repeating: dOut!.grid.first!)
        grads.append(gradA)
    }
}

// MARK: SML Convience Notation
func + (lhs: SMLUnit, rhs: SMLUnit) -> SMLAdd {
    return SMLAdd(lhs, rhs)
}

func * (lhs: SMLUnit, rhs: SMLUnit) -> SMLMul {
    return SMLMul(lhs, rhs)
}

func / (lhs: SMLUnit, rhs: SMLUnit) -> SMLMul {
    return SMLMul(lhs, SMLInv(rhs))
}

func - (lhs: SMLUnit, rhs: SMLUnit) -> SMLAdd {
    return SMLAdd(lhs, SMLNeg(rhs))
}

func <*> (lhs: SMLUnit, rhs: SMLUnit) -> SMLMatMul {
    return SMLMatMul(lhs, rhs)
}

extension SMLUnit {
    func smlsin() -> SMLSin {
        return SMLSin(self)
    }
    func smlexp() -> SMLExp {
        return SMLExp(self)
    }
    func smllog() -> SMLLog {
        return SMLLog(self)
    }
    func smlsum() -> SMLSum {
        return SMLSum(self)
    }
    func smlsquare() -> SMLSquare {
        return SMLSquare(self)
    }
    func smlpow(_ a: Double) -> SMLPow {
        let unit = SMLPow(self)
        unit.p = a
        return unit
    }
}

// MARK: SMLSession
class Session {
    
    var graph: SMLGraph = SMLGraph()
    var parallel: Bool = false
    
    init(parallel: Bool = false) {
        self.parallel = parallel
    }
    
    // seed == scalar basically means we are assuming we are ADing a function that outputs a scalar (all cost functions should do this)
    func run(_ root: SMLUnit, seed: Tensor = Tensor(1)) -> (out: Tensor, grads: [SMLUnit: Tensor]) {
        graph.set(root, seed: seed, parallel: parallel)
        let out = graph.fwd()
        let grads = graph.bwd()
        return (out, grads)
    }
}
