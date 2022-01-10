//
//  ComputationalGraph.swift
//
//
//  Created by Sahil Srivastava on 11/27/21.
//

import Foundation
import SwiftUI

// Need to make a new class that can handle matrix + matrix, scalar + matrix, scalar + scalar, [matrix] + [matrix], [matrix] + scalar, [matrix] + matrix

// MARK: CGraph
public final class CGraph {
    
    // Nodes sorted in high to low dependencies order
    // (root is first element AKA most dependent)
    private var nodes: [Variable] = []
    // FOR PARALLEL
    // Keys for each depth, less dependencies as depth increases
    // (root is at first depth AKA most dependent)
    // This could potentially be optimized to use pointers? instead of actual object ref
    private var nodes_parallel: [Int: [(node: Variable, parent: Variable?, gradIdx: Int)]] = [:]
    // Parallel toggle
    private var parallel: Bool = false
    // Seed
    private var seed: DTensor?
    
    // MARK: Init
    public init(_ root: Variable, seed: DTensor, parallel: Bool = false) {
        build(root, seed: seed, parallel: parallel)
    }
    
    public init() {}
    
    public func build(_ root: Variable, seed: DTensor, parallel: Bool = false) {
        self.parallel = parallel
        self.seed = seed
        // Sort graph based on paralleization
        if parallel {
            topology_sort_parallel(root)
        } else {
            topology_sort(root)
        }
    }
    
    public func pass(_ data: [Variable: DTensor]) {
        for (node, out) in data {
            // Set the corresponding placeholder for our key's value (now treated like a variable)
            if node.type == .placeholder {
                node.out = out
            }
        }
    }
    
    // MARK: Topology Sort (BFS)
    // DOES NOT CHECK FOR CYCLES
    private func topology_sort(_ root: Variable) {
        // Make empty queue
        var queue = [Variable]()
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
    private func topology_sort_parallel(_ root: Variable) {
        // Make empty queue
        var queue = [(node: Variable, depth: Int, parent: Variable?, gradIdx: Int)]()
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
    public func fwd(till: Variable? = nil) -> DTensor {
        // Run fwd based on paralleization
        if parallel {
            // till not implemented yet
            return forward_parallel(till: till)
        } else {
            return forward(till: till)
        }
    }
    
    // MARK: Forward
    private func forward(till: Variable? = nil) -> DTensor {
        // Set tracking if we already forwarded one of the nodes
        var forwarded = Set<Variable>()
        // Go through all nodes starting at lowest dependecy and forward
        for node in nodes.reversed() {
            // check since the same node dependency may have been duplicated
            if !forwarded.contains(node) {
                node.forward()
                if let till = till, node == till { return node.out! }
                forwarded.insert(node)
            }
        }
        return nodes.first!.out!
    }
    
    // MARK: Forward PARALLEL
    private func forward_parallel(till: Variable? = nil) -> DTensor {
        // Set tracking if we already forwarded one of the nodes
        var forwarded = Set<Variable>()
        // Go through all nodes starting at lowest dependecy and forward
        // This sorted keys could be optimized since it is used in forward_parallel and backward_bfs_parallel
        for depth in nodes_parallel.keys.sorted().reversed() {
            // Make our queue for nodes to be forwarded
            var queue = [Variable]()
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
            var until: Int? = nil
            // Concurrently forwarded our queued nodes
            DispatchQueue.concurrentPerform(iterations: queue.count) { i in
                queue[i].forward()
                if let till = till, queue[i] == till { until = i }
            }
            if let until = until { return queue[until].out! }
        }
        // dict[1] should always be just our root node
        return nodes_parallel[1]!.first!.node.out!
    }
    
    // MARK: Backward Exposed
    @discardableResult
    public func bwd() -> [Variable: DTensor] {
        // Run bwd based on paralleization
        if parallel {
            return backward_parallel(seed: seed!)
        } else {
            return backward(seed: seed!)
        }
    }
    
    // MARK: Backward
    private func backward(seed: DTensor) -> [Variable: DTensor] {
        // Make set to contain visited
        var grads = [Variable: DTensor]()
        // BFS from root and propagate gradient backwards
        backward_bfs(seed: seed, grads: &grads)
        return grads
    }
    
    // MARK: Backward BFS
    // DOES NOT CHECK FOR CYCLES
    private func backward_bfs(seed: DTensor, grads: inout [Variable: DTensor]) {
        // Make empty queue
        var queue = [(Variable, DTensor)]()
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
    private func backward_parallel(seed: DTensor) -> [Variable: DTensor] {
        // Make set to contain visited
        var grads = [Variable: DTensor]()
        // BFS from root and propagate gradient backwards
        backward_bfs_parallel(seed: seed, grads: &grads)
        return grads
    }
    
    // MARK: Backward BFS PARALLEL
    // DOES NOT CHECK FOR CYCLES
    private func backward_bfs_parallel(seed: DTensor, grads: inout [Variable: DTensor]) {
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
                var queues = [[(node: Variable, dOut: DTensor)]]()
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

// MARK: CGraphTypes
public enum CGGraphTypes: String {
    case variable = "variable"
    case placeholder = "placeholder"
    case constant = "constant"
    case operation = "operation"
}

// MARK: Variable
public class Variable: Hashable, Comparable {
    
    // Comparable conformance
    public static func < (lhs: Variable, rhs: Variable) -> Bool {
        return ObjectIdentifier(lhs) < ObjectIdentifier(rhs)
    }
    
    // Hashable conformance
    public static func == (lhs: Variable, rhs: Variable) -> Bool {
        return ObjectIdentifier(lhs) == ObjectIdentifier(rhs)
    }
    
    public final func hash(into hasher: inout Hasher) { return hasher.combine(ObjectIdentifier(self)) }
    
    public final var inputs: [Variable]
    public final var out: DTensor?
    // This will contain the chain ruled gradients for the inputs to the unit
    public final var grads: [DTensor]
    public final var tag: String
    
    public final var type: CGGraphTypes
    
    public init(_ out: DTensor? = nil, inputs: [Variable] = [], tag: String = "") {
        self.inputs = inputs
        self.out = out
        self.grads = []
        self.tag = tag
        self.type = .variable
    }
    
    public init(_ out: Double, inputs: [Variable] = [], tag: String = "") {
        self.inputs = inputs
        self.out = DTensor(out)
        self.grads = []
        self.tag = tag
        self.type = .variable
    }
    
    public func forward() { }
    // dOut is the gradient for this node wrt to the root of the graph
    public func backward(dOut: DTensor?) {}
}

// MARK: Placeholder
public final class Placeholder: Variable {
    
    public init(tag: String = "") {
        super.init(tag: tag)
        self.type = .placeholder
    }
}

// MARK: Constant
public final class Constant: Variable {
    
    public init(_ out: DTensor, tag: String = "") {
        super.init(out, tag: tag)
        self.type = .constant
    }
    
    public init(_ out: Double, tag: String = "") {
        super.init(out, tag: tag)
        self.type = .constant
    }
}

// MARK: BinaryOp
public class BinaryOp: Variable {
    
    public init(_ a: Variable, _ b: Variable, tag: String = "") {
        super.init(inputs: [a, b], tag: tag)
        grads = Array(repeating: DTensor(shape: [], grid: []), count: 2)
        self.type = .operation
    }
}

// MARK: UnaryOp
public class UnaryOp: Variable {
    
    public init(_ a: Variable, tag: String = "") {
        super.init(inputs: [a], tag: tag)
        grads = Array(repeating: DTensor(shape: [], grid: []), count: 1)
        self.type = .operation
    }
}

// MARK: Add
public final class Add: BinaryOp {
    
    public override func forward() {
        out = inputs[0].out! + inputs[1].out!
    }
    
    public override func backward(dOut: DTensor?) {
        // Clarify a and b
        let a = inputs[0].out!
        let b = inputs[1].out!
        // Gradient for input_nodes[0] AKA a
        var gradA = dOut!
        // Assumes a.shape.count == dOut!.shape.count
        while gradA.shape.count > a.shape.count {
            gradA = gradA.sum(axis: 0)
        }
        var axis = 0
        // Condense gradients for potential vector * matrix math
        for dim in a.shape {
            if dim == 1 { gradA = gradA.sum(axis: axis, keepDim: true) }
            axis += 1
        }
        grads[0] = gradA
        // Gradient for input_nodes[1] AKA b
        var gradB = dOut!
        // Assumes b.shape.count == dOut!.shape.count
        while gradB.shape.count > b.shape.count {
            gradB = gradB.sum(axis: 0)
        }
        axis = 0
        // Condense gradients for potential vector * matrix math
        for dim in b.shape {
            if dim == 1 { gradB = gradB.sum(axis: axis, keepDim: true) }
            axis += 1
        }
        grads[1] = gradB
    }
}

// MARK: Mul
public final class Mul: BinaryOp {
    
    public override func forward() {
        out = inputs[0].out! * inputs[1].out!
    }
    
    public override func backward(dOut: DTensor?) {
        // Clarify a and b
        let a = inputs[0].out!
        let b = inputs[1].out!
        // Gradient for input_nodes[0] AKA a
        var gradA = dOut! * b
        // Assumes a.shape.count == dOut!.shape.count
        while gradA.shape.count > a.shape.count {
            gradA = gradA.sum(axis: 0)
        }
        var axis = 0
        // Condense gradients for potential vector * matrix math
        for dim in a.shape {
            if dim == 1 { gradA = gradA.sum(axis: axis, keepDim: true) }
            axis += 1
        }
        grads[0] = gradA
        // Gradient for input_nodes[1] AKA b
        var gradB = dOut! * a
        // Assumes b.shape.count == dOut!.shape.count
        while gradB.shape.count > b.shape.count {
            gradB = gradB.sum(axis: 0)
        }
        axis = 0
        // Condense gradients for potential vector * matrix math
        for dim in b.shape {
            if dim == 1 { gradB = gradB.sum(axis: axis, keepDim: true) }
            axis += 1
        }
        grads[1] = gradB
    }
}

// MARK: MatMul
public final class MatMul: BinaryOp {
    
    public override func forward() {
        out = inputs[0].out! <*> inputs[1].out!
    }
    
    public override func backward(dOut: DTensor?) {
        // Clarify a and b
        let a = inputs[0].out!
        let b = inputs[1].out!
        // Gradient for input_nodes[0] aka a
        let gradA = b.transpose()
        grads[0] = dOut! <*> gradA
        // Gradient for input_nodes[1] aka b
        let gradB = a.transpose()
        grads[1] = gradB <*> dOut!
    }
}

// MARK: MatTran (not tested but like.... c'mon)
public final class MatTran: UnaryOp {
    
    public override func forward() {
        out = inputs[0].out!.transpose()
    }
    
    public override func backward(dOut: DTensor?) {
        grads[0] = dOut!.transpose()
    }
}

// MARK: Inv
public final class Inv: UnaryOp {
    
    public override func forward() {
        out = 1.0 / inputs[0].out!
    }
    
    public override func backward(dOut: DTensor?) {
        // Clarify a
        let a = inputs[0].out!
        // Gradient for input_nodes[0] AKA a
        let gradA = -1.0 / a.pow(2.0)
        grads[0] = dOut! * gradA
    }
}

// MARK: Negate
public final class Negate: UnaryOp {
    
    public override func forward() {
        out = -inputs[0].out!
    }
    
    public override func backward(dOut: DTensor?) {
        // Gradient for input_nodes[0] AKA a
        let gradA = -1.0
        grads[0] = dOut! * gradA
    }
}

// MARK: Sin
public final class Sin: UnaryOp {
    
    public override func forward() {
        out = inputs[0].out!.sin()
    }
    
    public override func backward(dOut: DTensor?) {
        // Clarify a
        let a = inputs[0].out!
        // Gradient for input_nodes[0] AKA a
        let gradA = a.cos()
        grads[0] = dOut! * gradA
    }
}

// MARK: Exp
public final class Exp: UnaryOp {
    
    public override func forward() {
        out = inputs[0].out!.exp()
    }
    
    public override func backward(dOut: DTensor?) {
        // Gradient for input_nodes[0] AKA a
        let gradA = out!
        grads[0] = dOut! * gradA
    }
}

// MARK: Log
public final class Log: UnaryOp {
    
    public override func forward() {
        out = (inputs[0].out! + 0.00000001).log()
    }
    
    public override func backward(dOut: DTensor?) {
        // Clarify a
        let a = inputs[0].out!
        // Gradient for input_nodes[0] AKA a
        let gradA = 1.0 / a
        grads[0] = dOut! * gradA
    }
}

// MARK: Square
public final class Square: UnaryOp {
    
    public override func forward() {
        out = inputs[0].out!.pow(2)
    }
    
    public override func backward(dOut: DTensor?) {
        // Clarify a
        let a = inputs[0].out!
        // Gradient for input_nodes[0] AKA a
        let gradA = 2.0 * a
        grads[0] = dOut! * gradA
    }
}

// MARK: Pow
public final class Pow: UnaryOp {
    
    var p: Double?
    
    public override func forward() {
        out = inputs[0].out!.pow(p!)
    }
    
    public override func backward(dOut: DTensor?) {
        // Clarify a
        let a = inputs[0].out!
        // Gradient for input_nodes[0] AKA a
        let gradA = p! * a.pow(p! - 1)
        grads[0] = dOut! * gradA
    }
}

// MARK: Sum
public final class Sum: UnaryOp {
    
    public override func forward() {
        out = DTensor(inputs[0].out!.sum())
    }
    
    public override func backward(dOut: DTensor?) {
        precondition(dOut!.shape.count == 0)
        // Clarify a
        let a = inputs[0].out!
        // Gradient for input_nodes[0] AKA a
        let gradA = DTensor(shape: a.shape, repeating: dOut!.grid.first!)
        grads[0] = gradA
    }
}

// MARK: SumDiag
public final class SumDiag: UnaryOp {
    
    public override func forward() {
        out = DTensor(inputs[0].out!.sumDiag())
    }
    
    public override func backward(dOut: DTensor?) {
        precondition(dOut!.shape.count == 0)
        // Clarify a
        let a = inputs[0].out!
        // Gradient for input_nodes[0] AKA a
        let gradA = DTensor(shape: a.shape, repeating: dOut!.grid.first!).diag()
        grads[0] = gradA
    }
}

// MARK: Op Convience Notation
public func + (lhs: Variable, rhs: Variable) -> Add {
    return Add(lhs, rhs)
}

public func * (lhs: Variable, rhs: Variable) -> Mul {
    return Mul(lhs, rhs)
}

public func / (lhs: Variable, rhs: Variable) -> Mul {
    return Mul(lhs, Inv(rhs))
}

public func - (lhs: Variable, rhs: Variable) -> Add {
    return Add(lhs, Negate(rhs))
}

public func <*> (lhs: Variable, rhs: Variable) -> MatMul {
    return MatMul(lhs, rhs)
}

extension Variable {
    public func transpose() -> MatTran {
        return MatTran(self)
    }
    public func sin() -> Sin {
        return Sin(self)
    }
    public func exp() -> Exp {
        return Exp(self)
    }
    public func log() -> Log {
        return Log(self)
    }
    public func sum() -> Sum {
        return Sum(self)
    }
    public func sumDiag() -> SumDiag {
        return SumDiag(self)
    }
    public func square() -> Square {
        return Square(self)
    }
    public func pow(_ a: Double) -> Pow {
        let unit = Pow(self)
        unit.p = a
        return unit
    }
}

// MARK: Session
public final class Session {
    
    public var graph: CGraph = CGraph()
    public var parallel: Bool = false
    
    public init(parallel: Bool = false) {
        self.parallel = parallel
    }
    
    // seed == scalar basically means we are assuming we are ADing a function that outputs a scalar (all cost functions should do this)
    public func build(_ root: Variable, seed: DTensor = DTensor(1)) {
        // Builds the graph (currently just topolgical sort, maybe offset some other work here?)
        graph.build(root, seed: seed, parallel: parallel)
    }
    
    public func pass(_ data: [Variable: DTensor]) {
        // Populates our placeholders
        graph.pass(data)
    }
    
    public func run(_ root: Variable, till: Variable? = nil, seed: DTensor = DTensor(1)) -> (out: DTensor, grads: [Variable: DTensor]) {
        if till != nil {
            // Forward to get our answer
            let out = graph.fwd(till: till)
            return (out, [:])
        }
        // Forward to get our answer
        let out = graph.fwd()
        // Backward to get our gradients/ derivatives
        let grads = graph.bwd()
        return (out, grads)
    }
    
    public func descend(grads: [Variable: DTensor], optim: Optimizer, lr: Double) {
        var types = [String: Int]()
        // Two empty soon to be perfect arrays
        var params = [Variable]()
        var optim_grads = [DTensor]()
        // Store the computed gradients for our parameters
        for (node, grad) in grads.sorted(by: { $0.key < $1.key }) {
            if node.type == .variable {
                params.append(node)
                optim_grads.append(grad)
//                print(node.out!.shape)
            }
            let str = node.type.rawValue
            if types[str] != nil {
                types[str]! += 1
            } else {
                types[str] = 1
            }
        }
//        for (key, val) in types {
//            print("\(key): \(val)")
//        }
        // Use chosen optimizer to find out optimized gradients
        optim_grads.withUnsafeBufferPointer { param_gradsPtr in
            optim_grads = optim.gradients(grads_ptr: param_gradsPtr)
        }
        // Now take our gradient step for our params
        for i in 0..<params.count {
            params[i].out! = params[i].out! - lr * optim_grads[i]
        }
    }
}

// MARK: Optimizer
public protocol Optimizer {
    
    func gradients(grads_ptr: UnsafeBufferPointer<DTensor>) -> [DTensor]
}

// MARK: ADAM
public final class Adam: Optimizer {
    
    private var m = [DTensor]()
    private var v = [DTensor]()
    
    private var b1: Double
    private var b2: Double
    private var t: Int
    
    public init(b1: Double = 0.9, b2: Double = 0.999) {
        self.b1 = b1
        self.b2 = b2
        self.t = 0
    }
    
    public func inc() {
        // Add to t on each epoch
        t += 1
    }
    
    public func gradients(grads_ptr: UnsafeBufferPointer<DTensor>) -> [DTensor] {
        // Get our mean moments for our grads
        m = mean(derivatives: grads_ptr, m: m, b1: b1)
        // Get our variance moments for our grads
        v = variance(derivatives: grads_ptr, v: v, b2: b2)
        
        // Make our gradients for our weights and biases from our mean and variance moments
        let grads = grads(m: m, v: v, t: t)
        return grads
    }
    
    private func grads(m: [DTensor], v: [DTensor], t: Int) -> [DTensor] {
        return zip(m, v).map { m_l, v_l in
            // Get corrected m and v
            let mc_l: DTensor = m_l / (1 - pow(0.9, Double(t)))
            let vc_l: DTensor = v_l / (1 - pow(0.999, Double(t)))
            // Define our gradient
            return mc_l / (vc_l.sqrt() + 0.00000001)
        }
    }
    
    private func mean(derivatives: UnsafeBufferPointer<DTensor>, m: [DTensor], b1: Double) -> [DTensor] {
        // First set up empty gradient array to store each layers gradient matrix
        var firstMoments = [DTensor]()
        // On nth iteration get the gradient with momentum for each layer and add it to our gradient matrix
        for l in 0..<derivatives.count {
            let firstMoment: DTensor
            if m.isEmpty {
                // Make our gradient for this layer which is just our derivative * lr this time
                firstMoment = (1 - b1) * derivatives[l]
            } else {
                // Make our gradient for this layer using our last gradient and the derivative * lr
                firstMoment = b1 * m[l] + (1 - b1) * derivatives[l]
            }
            // Add this layers gradient to the gradient array
            firstMoments.append(firstMoment)
        }
        // At this point gradient[l] is a matrix of the rolling averages of the previous derivatives and this iterations derivative so add it to our gradients to be used for our next gradient calculation
        return firstMoments
    }
    
    private func variance(derivatives: UnsafeBufferPointer<DTensor>, v: [DTensor], b2: Double) -> [DTensor] {
        // First set up empty gradient array to store each layers gradient matrix
        var secondMoments = [DTensor]()
        // On nth iteration get the gradient with momentum for each layer and add it to our gradient matrix
        for l in 0..<derivatives.count {
            let secondMoment: DTensor
            if v.isEmpty {
                // Make our gradient for this layer which is just our derivative * lr this time
                secondMoment = (1 - b2) * derivatives[l].pow(2)
            } else {
                // Make our gradient for this layer using our last gradient and the derivative * lr
                secondMoment = b2 * v[l] + (1 - b2) * derivatives[l].pow(2)
            }
            // Add this layers gradient to the gradient array
            secondMoments.append(secondMoment)
        }
        // At this point gradient[l] is a matrix of the rolling averages of the previous derivatives and this iterations derivative so add it to our gradients to be used for our next gradient calculation
        return secondMoments
    }
}
