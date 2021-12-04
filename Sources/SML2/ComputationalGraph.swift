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
    
    // MARK: Init
    init(_ root: SMLUnit) {
        topology_sort(root)
    }
    
    init() {}
    
    func set(_ root: SMLUnit) {
        topology_sort(root)
    }
    
    // MARK: Topology Sort (BFS)
    // DOES NOT CHECK FOR CYCLES
    func topology_sort(_ root: SMLUnit) {
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
    
    // MARK: Forward
    @discardableResult
    func forward() -> Double {
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
    
    // MARK: Backward
    @discardableResult
    func backward(seed: Double = 1.0) -> [SMLUnit: Double] {
        // Make set to contain visited
        var grads = [SMLUnit: Double]()
        // BFS from root and propagate gradient backwards
        backward_bfs(seed: seed, grads: &grads)
        return grads
    }
    
    // MARK: Backward BFS
    // DOES NOT CHECK FOR CYCLES
    private func backward_bfs(seed: Double, grads: inout [SMLUnit: Double]) {
        // Make empty queue
        var queue = [(SMLUnit, Double)]()
        // Add our first node (root AKA most dependent) and the first gradient (usually 1) to queue
        queue.append((nodes.first!, seed))
        // BFS until queue is empty
        while !queue.isEmpty {
            // Pop our first item in the queue
            let (curr, dOut) = queue.removeFirst()
            // Add curr to our visited
            if grads[curr] != nil {
                // Increment gradient if already contributing to final rate of change
                grads[curr]! += dOut
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
}

// MARK: SMLNode
class SMLUnit: Hashable {
    
    // Hashable conformance
    static func == (lhs: SMLUnit, rhs: SMLUnit) -> Bool {
        return ObjectIdentifier(lhs) == ObjectIdentifier(rhs)
    }
    
    func hash(into hasher: inout Hasher) { return hasher.combine(ObjectIdentifier(self)) }
    
    var inputs: [SMLUnit]
    var out: Double?
    // This will contain the chain ruled gradients for the inputs to the unit
    var grads: [Double] = []
    var tag: String
    
    init(_ out: Double? = nil, inputs: [SMLUnit] = [], tag: String = "") {
        self.inputs = inputs
        self.out = out
        self.tag = tag
    }
    
    func forward() { }
    // dOut is the gradient for this node wrt to the root of the graph
    func backward(dOut: Double?) {}
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
    
    override func backward(dOut: Double?) {
        // Gradient for input_nodes[0] AKA a
        let gradA = 1.0
        grads.append(dOut! * gradA)
        // Gradient for input_nodes[1] AKA b
        let gradB = 1.0
        grads.append(dOut! * gradB)
    }
}

// MARK: SMLMul
class SMLMul: SMLBinary {
    
    override func forward() {
        out = inputs[0].out! * inputs[1].out!
    }
    
    override func backward(dOut: Double?) {
        // Clarify a and b
        let a = inputs[0]
        let b = inputs[1]
        // Gradient for input_nodes[0] AKA a
        let gradA = b.out!
        grads.append(dOut! * gradA)
        // Gradient for input_nodes[0] AKA a
        let gradB = a.out!
        grads.append(dOut! * gradB)
    }
}

// MARK: SMLInv
class SMLInv: SMLUnary {
    
    override func forward() {
        out = 1.0 / inputs[0].out!
    }
    
    override func backward(dOut: Double?) {
        // Clarify a
        let a = inputs[0]
        // Gradient for input_nodes[0] AKA a
        let gradA = -1.0 / pow(a.out!, 2.0)
        grads.append(dOut! * gradA)
    }
}

// MARK: SMLNeg
class SMLNeg: SMLUnary {
    
    override func forward() {
        out = -inputs[0].out!
    }
    
    override func backward(dOut: Double?) {
        // Gradient for input_nodes[0] AKA a
        let gradA = -1.0
        grads.append(dOut! * gradA)
    }
}

// MARK: SMLSin
class SMLSin: SMLUnary {
    
    override func forward() {
        out = sin(inputs[0].out!)
    }
    
    override func backward(dOut: Double?) {
        // Clarify a
        let a = inputs[0].out!
        // Gradient for input_nodes[0] AKA a
        let gradA = cos(a)
        grads.append(dOut! * gradA)
    }
}

// MARK: SMLExp
class SMLExp: SMLUnary {
    
    override func forward() {
        out = exp(inputs[0].out!)
    }
    
    override func backward(dOut: Double?) {
        // Gradient for input_nodes[0] AKA a
        let gradA = out!
        grads.append(dOut! * gradA)
    }
}

// MARK: SMLLog
class SMLLog: SMLUnary {
    
    override func forward() {
        out = log(inputs[0].out! + 0.00000001)
    }
    
    override func backward(dOut: Double?) {
        // Clarify a
        let a = inputs[0].out!
        // Gradient for input_nodes[0] AKA a
        let gradA = 1.0 / a
        grads.append(dOut! * gradA)
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
}

// MARK: SMLSession
class Session {
    
    var graph: SMLGraph = SMLGraph()
    
    init() {}
    
    func run(_ root: SMLUnit) -> (out: Double, grads: [SMLUnit: Double]) {
        graph.set(root)
        let out = graph.forward()
        let grads = graph.backward()
        return (out, grads)
    }
}
