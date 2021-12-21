//
//  NeuralNetwork.swift
//  
//
//  Created by Sahil Srivastava on 12/13/21.
//

import Foundation

// MARK: Sequence
public struct Sequence {
    
    public private (set)var layers: [Layer]
    
    public var input: Variable {
        return layers.first!.input
    }
    
    public var weights: [Variable] {
        var weights = [Variable]()
        for layer in layers {
            if let layer = layer as? Linear {
                weights.append(layer.weight)
            }
        }
        return weights
    }
    
    public var regularizer: Variable {
        var reg: Variable? = nil
        for weight in weights {
            if reg == nil {
                reg = weight.pow(2).sum()
            } else {
                reg = reg! + weight.pow(2).sum()
            }
        }
        return reg!
    }
    
    public var batch_norms: [BatchNorm] {
        var batch_norms = [BatchNorm]()
        for layer in layers {
            if let layer = layer as? BatchNorm {
                batch_norms.append(layer)
            }
        }
        return batch_norms
    }
    
    public var predicted: Variable {
        return layers.last!
    }
    
    public init(_ layers: [Layer]) {
        // Link linear layers if we need to
        for i in 0..<layers.count {
            if i == 0 { continue }
            let prev_layer = layers[i - 1]
            let layer = layers[i]
            layer.input = prev_layer
        }
        self.layers = layers
    }
}

// MARK: Layer
public class Layer: Variable {
    
    final var input: Variable {
        get {
            return inputs[0]
        }
        set {
            inputs[0] = newValue
        }
    }
    
    public init(inputs: [Variable], tag: String = "") {
        precondition(inputs.count >= 1, "A layer must have at least one input")
        super.init(inputs: inputs, tag: tag)
        self.type = .operation
    }
}

// MARK: Linear
public final class Linear: Layer {
    
    private var combination: Tensor?
    
    public var weight: Variable {
        return inputs[1]
    }
    public var bias: Variable {
        return inputs[2]
    }
    
    // Placeholder in case input currently uknown (first input, or we haven't fed forward)
    public init(_ input: Variable = Placeholder(), to: Int, out: Int, tag: String = "") {
        let weight = Variable(Tensor.random_xavier(shape: [out, to], ni: to, no: out))
        let bias = Variable(Tensor(shape: [out, 1], repeating: 0))
        super.init(inputs: [input, weight, bias], tag: tag)
        grads = Array(repeating: Tensor(shape: [], grid: []), count: 3)
    }
    
    public override func forward() {
        // Clarify inputs
        let data = inputs[0].out!
        let weight = inputs[1].out!
        let bias = inputs[2].out!
        // Make our linear combination
        combination = weight <*> data
        out = combination! + bias
    }
    
    public override func backward(dOut: Tensor?) {
        // Clarify inputs
        let data = inputs[0].out!
        let weight = inputs[1].out!
        let combination = combination!
        let bias = inputs[2].out!
        // Grad of combination
        var gradCombination = dOut!
        var axis = 0
        // Assumes combination.shape.count == dOut!
        // Condense gradients for potential vector * matrix math
        for dim in combination.shape {
            if dim == 1 { gradCombination = gradCombination.sum(axis: axis, keepDim: true) }
            axis += 1
        }
        // Grad of input
        let gradInput = weight.transpose() <*> gradCombination
        // Grad of weight
        let gradWeight = gradCombination <*> data.transpose()
        // Grad of bias
        var gradBias = dOut!
        axis = 0
        // Assumes bias.shape.count == dOut!
        // Condense gradients for potential vector * matrix math
        for dim in bias.shape {
            if dim == 1 { gradBias = gradBias.sum(axis: axis, keepDim: true) }
            axis += 1
        }
        // Now set our grads
        grads[0] = gradInput
        grads[1] = gradWeight
        grads[2] = gradBias
    }
}

// MARK: Sigmoid
public final class Sigmoid: Layer {
    
    // Placeholder in case input currently unknown (we haven't fed forward)
    public init(_ input: Variable = Placeholder(), tag: String = "") {
        super.init(inputs: [input], tag: tag)
        grads = Array(repeating: Tensor(shape: [], grid: []), count: 1)
    }
    
    public override func forward() {
        // Clarify inputs
        let input = inputs[0].out!
        // Run through sigmoid func
        out = input.sigmoid()
    }
    
    public override func backward(dOut: Tensor?) {
        // Clarify inputs
        let out = out!
        // Grad on input
        let gradInput = out * (1 - out)
        // Now set our grads (dont forget chainrule)
        grads[0] = dOut! * gradInput
    }
}

// MARK: ReLU
public final class ReLU: Layer {
    
    // Placeholder in case input currently unknown (we haven't fed forward)
    public init(_ input: Variable = Placeholder(), tag: String = "") {
        super.init(inputs: [input], tag: tag)
        grads = Array(repeating: Tensor(shape: [], grid: []), count: 1)
    }
    
    public override func forward() {
        // Clarify inputs
        let input = inputs[0].out!
        // Run through sigmoid func
        out = input.relu()
    }
    
    public override func backward(dOut: Tensor?) {
        // Clarify inputs
        let input = inputs[0].out!
        // Grad on input
        let gradInput = input.drelu()
        // Now set our grads (dont forget chainrule)
        grads[0] = dOut! * gradInput
    }
}

// MARK: LReLU
public final class LReLU: Layer {
    
    // Placeholder in case input currently unknown (we haven't fed forward)
    public init(_ input: Variable = Placeholder(), tag: String = "") {
        super.init(inputs: [input], tag: tag)
        grads = Array(repeating: Tensor(shape: [], grid: []), count: 1)
    }
    
    public override func forward() {
        // Clarify inputs
        let input = inputs[0].out!
        // Run through sigmoid func
        out = input.lrelu()
    }
    
    public override func backward(dOut: Tensor?) {
        // Clarify inputs
        let input = inputs[0].out!
        // Grad on input
        let gradInput = input.dlrelu()
        // Now set our grads (dont forget chainrule)
        grads[0] = dOut! * gradInput
    }
}

// MARK: BatchNorm
public final class BatchNorm: Layer {
    
    private var data_norm: Tensor?
    private var running_mean: Tensor?
    private var running_std: Tensor?
    private var sample_std: Tensor?
    private var momentum: Double
    
    public var training: Bool
    
    public var gamma: Variable {
        return inputs[1]
    }
    public var beta: Variable {
        return inputs[2]
    }
    
    // Placeholder in case input currently unknown (we haven't fed forward)
    public init(_ input: Variable = Placeholder(), to: Int, momentum: Double = 0.9, training: Bool = true, tag: String = "") {
        let gamma = Variable(Tensor(shape: [to, 1], repeating: 4))
        let beta = Variable(Tensor(shape: [to, 1], repeating: 5))
        self.training = training
        self.momentum = momentum
        super.init(inputs: [input, gamma, beta], tag: tag)
        grads = Array(repeating: Tensor(shape: [], grid: []), count: 3)
    }
    
    public override func forward() {
        // Clarify inputs
        let data = inputs[0].out!
        let gamma = inputs[1].out!
        let beta = inputs[2].out!
        if training {
            // Make our normalized and scaled output (transpose so we get mean/std for each feature)
            var (data_norm, sample_mean, sample_std) = data.transpose().zscore()
            data_norm = data_norm.transpose()
            out = gamma * data_norm + beta
            // Running mean and std
            if let running_mean = self.running_mean, let running_std = self.running_std {
                self.running_mean = momentum * running_mean + (1 - momentum) * sample_mean
                self.running_std = momentum * running_std + (1 - momentum) * sample_std
            } else {
                self.running_mean = (1 - momentum) * sample_mean
                self.running_std = (1 - momentum) * sample_std
            }
            // Cache
            self.data_norm = data_norm
            self.sample_std = sample_std // needed for backprop
        } else {
            let data_norm = (data - self.running_mean!.transpose()) / self.running_std!.transpose()
            out = gamma * data_norm + beta
        }
    }
    
    public override func backward(dOut: Tensor?) {
        // Clarify inputs
        let data_norm = data_norm!
        let gamma = inputs[1].out!
        let sample_std = sample_std!.transpose()
        // Get grad for data
        let gradData_norm = dOut! * gamma
        let m = Double(dOut!.shape[1])
        // Simplified equation
        let gradData = (1.0 / m) / sample_std * (m * gradData_norm - gradData_norm.sum(axis: 1, keepDim: true) - data_norm * (gradData_norm * data_norm).sum(axis: 1, keepDim: true))
        // Get grad for gamma
        let gradGamma = (dOut! * data_norm).sum(axis: 1, keepDim: true)
        // Get grad for beta
        let gradBeta = dOut!.sum(axis: 1, keepDim: true)
        // Now set our grads
        grads[0] = gradData
        grads[1] = gradGamma
        grads[2] = gradBeta
    }
}

// MARK: Process
public final class Process {
    
    public func shuffle(data: [[Double]], labels: [[Double]]) -> (shuffledData: [[Double]], shuffledLabels: [[Double]]) {
        var merged = [[Double]]()
        var breaks = [Int]()
        for i in 0..<min(data.count, labels.count) {
            // Make the 1D array first
            merged.append([])
            // Add our data and labels row
            merged[i].append(contentsOf: data[i])
            merged[i].append(contentsOf: labels[i])
            // Get the break between our labels and data
            breaks.append(data[i].count)
        }
        // Shuffle
        merged.shuffle()
        // Make our new data and labels
        var shuffledData = [[Double]]()
        var shuffledLabels = [[Double]]()
        for i in 0..<merged.count {
            // Store our shuffled data row
            shuffledData.append([])
            shuffledData[i].append(contentsOf: Array(merged[i][0..<breaks[i]]))
            // Store our shuffled labels row
            shuffledLabels.append([])
            shuffledLabels[i].append(contentsOf: Array(merged[i][breaks[i]..<merged[i].count]))
        }
        return (shuffledData, shuffledLabels)
    }
    
    private var mean = Tensor(shape: [], grid: [])
    private var std = Tensor(shape: [], grid: [])
    
    public func zscore(_ input: Tensor, type: ProcessType) -> Tensor {
        switch type {
        case .data:
            let (norm, mean, std) = input.zscore()
            self.mean = mean
            self.std = std
            return norm
        case .pred:
            return (input - mean.transpose()) / std.transpose()
        }
    }
}

public enum ProcessType {
    case data
    case pred
}
