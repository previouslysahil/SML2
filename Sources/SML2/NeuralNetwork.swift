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
        let bias = Variable(Tensor(shape: [out, 1], repeating: 0.01))
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
        while gradCombination.shape.count > combination.shape.count {
            gradCombination = gradCombination.sum(axis: 0)
        }
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

// MARK: Conv2D
public final class Conv2D: Layer {
    
    public var kernel: Variable {
        return inputs[1]
    }
    
    public var bias: Variable {
        return inputs[2]
    }
    
    private let pad: Bool
    
    // Placeholder in case input currently unknown (we haven't fed forward)
    public init(_ input: Variable = Placeholder(), to: Int, out: Int, size: Int, pad: Bool = false, tag: String = "") {
        var kernelShape: [Int]
        // Extra shape here would mess up dimensions
        kernelShape = to == 1 ? [out, size, size] : [out, to, size, size]
        // No need for extra shape out as well
        if out == 1 { kernelShape.removeFirst() }
        // Set kernel shape (maybe different random? xavier?
        let kernel = Variable(Tensor.random(shape: kernelShape))
        let bias = Variable(Tensor(shape: [out], repeating: 0.01))
        // Not implemented yet
        self.pad = pad
        super.init(inputs: [input, kernel, bias], tag: tag)
        grads = Array(repeating: Tensor(shape: [], grid: []), count: 3)
    }
    
    public override func forward() {
        // Clarify inputs
        let data = inputs[0].out!
        let kernel = inputs[1].out!
        let bias = inputs[2].out!
        // Check for multiple kernels
        if kernel.shape.main.count == 4 && data.shape.main.count == 3 && kernel.shape.main[1] == data.shape.main[0] {
            // Multiple kernels with depth > 1
            // First get the shape of our 2D Tensor after convolution
            let (_, mat_reshaped) = Array(data.shape.main[1...2]).conv2D_shape(with: Array(kernel.shape.main[2...3]), type: .valid).seperate()
            // Now we can make the out shape using the 2D Tensor (matrix) with a depth of the number of kernels since out must have the same depth as the number of kernels
            out = Tensor(shape: [kernel.shape.main[0], mat_reshaped[0], mat_reshaped[1]], repeating: 0.0)
            // Now for each kernel we convolve with our data to produce our dth depth for out
            for d in 0..<kernel.shape.main[0] {
                // Get the dth kernel
                let kernelD = kernel[t3D: d]
                // Now convolve this kernel with our data, since both kernel and data are 3D Tensors we convolve the corresponding depth of data with that of kernelD
                for m in 0..<kernel.shape.main[1] {
                    out![mat: d] = out![mat: d] + data[mat: m].conv2D(with: kernelD[mat: m], type: .valid)
                }
                // Add the bias for the dth depth of out which corresponds to the dth kernel
                out![mat: d] = out![mat: d] + bias[d]
            }
            // For more clarity, this is essentially the following forward calculation (One kernel with depth > 1) except we have multiple kernels that contribute to out so we need to do convolutions for each kernel with data which will produce a new depth for out (making out a multi depth output)
        } else if kernel.shape.main.count == 3 && data.shape.main.count == 3 && kernel.shape.main[0] == data.shape.main[0] {
            // One kernel with depth > 1
            // First get the shape of our 2D Tensor after convolution
            let (_, mat_reshaped) = Array(data.shape.main[1...2]).conv2D_shape(with: Array(kernel.shape.main[1...2]), type: .valid).seperate()
            // Now we can make the out shape using the 2D Tensor (matrix), since we only have one kernel out only has a depth of 1 which we can omit making out 2D instead of 3D
            out = Tensor(shape: [mat_reshaped[0], mat_reshaped[1]], repeating: 0.0)
            // Now convolve this kernel with our data, since both kernel and data are 3D Tensors we convolve the corresponding depth of data with that of the kernel
            for m in 0..<kernel.shape.main[0] {
                out = out! + data[mat: m].conv2D(with: kernel[mat: m], type: .valid)
            }
            // Add the singular bias unit since we have only one kernel
            out = out! + bias[0]
        } else if kernel.shape.main.count == 3 && data.shape.main.count == 2 {
            // Multiple kernels with depth 1
            // First get the shape of our 2D Tensor after convolution
            let (_, mat_reshaped) = Array(data.shape.main[0...1]).conv2D_shape(with: Array(kernel.shape.main[1...2]), type: .valid).seperate()
            // Now we can make the out shape using the 2D Tensor (matrix) with a depth of the number of kernels since out must have the same depth as the number of kernels
            out = Tensor(shape: [kernel.shape.main[0], mat_reshaped[0], mat_reshaped[1]], repeating: 0.0)
            // Now we convolve each depth of out with each depth of our kernel to get our final out
            for m in 0..<kernel.shape.main[0] {
                // Each mth depth of out corresponds to each kernel and has its own bias
                out![mat: m] = data.conv2D(with: kernel[mat: m], type: .valid) + bias[m]
            }
            // For more clarity, this is essentially the following forward calculation (One kernel with depth 1) except we have multiple kernels that contribute to out so we need to do convolutions for each kernel with data which will produce a new depth for out (making out a multi depth output)
        } else if kernel.shape.main.count == 2 && data.shape.main.count == 2 {
            // One kernel with depth 1
            // Out only has one depth so we only convolve for this depth and omit the extra shape, making out a 2D Tensor
            out = data.conv2D(with: kernel, type: .valid) + bias[0]
        } else {
            fatalError("Data and kernels are incompatible")
        }
    }
    
    public override func backward(dOut: Tensor?) {
        // Clarify inputs
        let data = inputs[0].out!
        let kernel = inputs[1].out!
        let bias = inputs[2].out!
        // Declare grads
        var gradData: Tensor
        var gradKernel: Tensor
        var gradBias: Tensor
        // Get grad for data, kernel, and bias (valid) SHOULD ROTATE KERNEL 180
        // Check for multiple kernels
        if kernel.shape.main.count == 4 && data.shape.main.count == 3 && kernel.shape.main[1] == data.shape.main[0] {
            // Multiple kernels with depth > 1
            
            // *** GRADDATA ***
            // gradData should be the same shape as our data
            gradData = Tensor(shape: data.shape, repeating: 0.0)
            // Each depth of gradData is influenced by every kernel that was convoluted over data so we must convolve each depth of dOut with the corresponding depth of each kernel
            for d in 0..<kernel.shape.main[0] {
                // For each kernel we are adding, its mth depth convolved with the respective dth dOut that is influenced by this kernel, with the respective gradData mth depth
                for m in 0..<kernel.shape.main[1] {
                    // Each kernel only influences a single depth of dOut, but each depth of the kernel influences each depth of gradData, leading to this syntax
                    gradData[mat: m] = gradData[mat: m] + kernel[t3D: d][mat: m].conv2D(with: dOut![mat: d], type: .full)
                }
            }
            // For more clarity, this is essentially the following gradient calculation (One kernel with depth > 1) except we have multiple kernels that contribute to gradData so we need to add their convolved gradients to gradData as well
            
            // *** GRADKERNEL ***
            // gradKernel should be the same shape as our kernels tensor
            gradKernel = Tensor(shape: kernel.shape, repeating: 0)
            // Now we calculate the influence of each kernel
            for d in 0..<kernel.shape.main[0] {
                // For a single kernel its mth depths influence is found by the convolution with the mth depth of the data (since this depth only convolves with the respective data mth depth) and the dOut depth d that corresponds to this kernel (since each kernel only influences on depth of dOut)
                for m in 0..<kernel.shape.main[1] {
                    // Here we are finding the partial derivative for the mth depth of kernel d
                    gradKernel[t3D: d][mat: m] = data[mat: m].conv2D(with: dOut![mat: d], type: .valid)
                }
            }
            // For more clarity, this is essentially the following gradient calculation (One kernel with depth > 1) except we have multiple kernels that contribute to gradKernel so we need to calcaulte the convolved gradients for each different kernel to find the total gradKernel
            
            // *** GRADBIAS ***
            gradBias = bias
            // Do you realy need an explanation?
            for d in 0..<kernel.shape.main[0] {
                gradBias[d] = dOut![mat: d].sum()
            }
        } else if kernel.shape.main.count == 3 && data.shape.main.count == 3 && kernel.shape.main[0] == data.shape.main[0] {
            // One kernel with depth > 1
            
            // *** GRADDATA ***
            // gradData should be the same shape as our data
            gradData = Tensor(shape: data.shape, repeating: 0.0)
            // We are adding this kernels mth depth with the respective gradData depth
            for m in 0..<kernel.shape.main[0] {
                // dOut is a 2D Tensor here since we only have one kernel
                gradData[mat: m] = kernel[mat: m].conv2D(with: dOut!, type: .full)
            }
            
            // *** GRADKERNEL ***
            // gradKernel should be the same shape as our kernels tensor
            gradKernel = Tensor(shape: kernel.shape, repeating: 0)
            // To calculate the influence of this kernel we find the influence of each of mth depth
            for m in 0..<kernel.shape.main[0] {
                // Since dOut is a 2D Matrix (we only have one kernel) we convolve each depth of data with this dOut to get the partial of each depth of the kernel (since the kernel's mth depth only interacts with data's mth depth)
                gradKernel[mat: m] = data[mat: m].conv2D(with: dOut!, type: .valid)
            }
            
            // *** GRADBIAS ***
            gradBias = bias
            // If you got the last this is easier...
            gradBias[0] = dOut!.sum()
        } else if kernel.shape.main.count == 3 && data.shape.main.count == 2 {
            // Multiple kernels with depth 1
            
            // *** GRADDATA ***
            // gradData should be the same shape as our data
            gradData = Tensor(shape: data.shape, repeating: 0.0)
            // Since we have multiple kernels we need to add their partials to gradData
            for m in 0..<kernel.shape.main[0] {
                // gradData is influenced by kernel and the corresponding dOout that the kernels convolution with data created
                gradData = gradData + kernel[mat: m].conv2D(with: dOut![mat: m], type: .full)
            }
            // For more clarity, this is essentially the following gradient calculation (One kernel with depth 1) except we have multiple kernels that contribute to gradData so we need to add their convolved gradients to gradData as well
            
            // *** GRADKERNEL ***
            // gradKernel should be the same shape as our kernels tensor
            gradKernel = Tensor(shape: kernel.shape, repeating: 0.0)
            // Each kernels partial is made with convolving with the respective mth depth dOut!
            for m in 0..<kernel.shape.main[0] {
                // Since each depth of dOut is influenced by a single kernel we convolve data with this mth dOut depth to get the partial of this mth kernel
                gradKernel[mat: m] = data.conv2D(with: dOut![mat: m], type: .valid)
            }
            // For more clarity, this is essentially the following gradient calculation (One kernel with depth 1) except we have multiple kernels that contribute to gradKernel so we need to calcaulte the convolved gradients for each different kernel to find the total gradKernel
            
            // *** GRADBIAS ***
            gradBias = bias
            // Pretty simple too! just think!
            for d in 0..<kernel.shape.main[0] {
                gradBias[d] = dOut![mat: d].sum()
            }
        } else if kernel.shape.main.count == 2 && data.shape.main.count == 2 {
            // One kernel with depth 1
            
            // *** GRADDATA ***
            // We only have one kernel with depth one and data with depth one so dOut has depth one, resulting in the simplest case of a basic full convolution to get gradData
            gradData = kernel.conv2D(with: dOut!, type: .full)
            
            // *** GRADKERNEL ***
            // We only have one kernel with depth one and data with depth one so dOut has depth one, resulting in the simplest case of a basic full convolution to get gradKernel
            gradKernel = data.conv2D(with: dOut!, type: .valid)
            
            // *** GRADBIAS ***
            gradBias = bias
            // This is the easiest one...
            gradBias[0] = dOut!.sum()
        } else {
            fatalError("Data and kernels are incompatible")
        }
        // Now set our grads
        grads[0] = gradData
        grads[1] = gradKernel
        grads[2] = gradBias
    }
}

public enum ConvType {
    case valid
    case same
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
