//
//  NeuralNetwork.swift
//  
//
//  Created by Sahil Srivastava on 12/13/21.
//

import Foundation

// MARK: Sequence
public struct Sequence<Tensor: Tensorable> {
    
    public private (set)var layers: [Layer<Tensor>]
    
    public var input: Variable<Tensor> {
        return layers.first!.input
    }
    
    public var weights: [Variable<Tensor>] {
        var weights = [Variable<Tensor>]()
        for layer in layers {
            if let layer = layer as? Linear {
                weights.append(layer.weight)
            }
        }
        return weights
    }
    
    public var kernels: [Variable<Tensor>] {
        var kernels = [Variable<Tensor>]()
        for layer in layers {
            if let layer = layer as? Conv2D {
                kernels.append(layer.kernel)
            }
        }
        return kernels
    }
    
    public var regularizer: Variable<Tensor> {
        var reg: Variable<Tensor>? = nil
        for weight in weights {
            if reg == nil {
                reg = weight.pow(2).sum()
            } else {
                reg = reg! + weight.pow(2).sum()
            }
        }
        for kernel in kernels {
            if reg == nil {
                reg = kernel.pow(2).sum()
            } else {
                reg = reg! + kernel.pow(2).sum()
            }
        }
        return reg!
    }
    
    public var batch_norms: [BatchNorm<Tensor>] {
        var batch_norms = [BatchNorm<Tensor>]()
        for layer in layers {
            if let layer = layer as? BatchNorm {
                batch_norms.append(layer)
            }
        }
        return batch_norms
    }
    
    public var batch_norms2D: [BatchNorm2D<Tensor>] {
        var batch_norms = [BatchNorm2D<Tensor>]()
        for layer in layers {
            if let layer = layer as? BatchNorm2D {
                batch_norms.append(layer)
            }
        }
        return batch_norms
    }
    
    public var predicted: Variable<Tensor> {
        return layers.last!
    }
    
    public init(_ layers: [Layer<Tensor>]) {
        // Link layers if we need to
        for i in 0..<layers.count {
            if i == 0 { continue }
            let prev_layer = layers[i - 1]
            let layer = layers[i]
            layer.input = prev_layer
        }
        self.layers = layers
    }
    
    public func encode_params() -> Data? {
        var params = [SequenceParam<Tensor>]()
        for (idx, layer) in layers.enumerated() {
            if let layer = layer as? Linear {
                let weight = layer.weight.out!
                let bias = layer.bias.out!
                params.append(SequenceParam(layer: idx, name: "weight", grid: weight.grid, shape: weight.shape.strip()))
                params.append(SequenceParam(layer: idx, name: "weight bias", grid: bias.grid, shape: bias.shape.strip()))
            } else if let layer = layer as? Conv2D {
                let kernel = layer.kernel.out!
                let bias = layer.bias.out!
                params.append(SequenceParam(layer: idx, name: "kernel", grid: kernel.grid, shape: kernel.shape.strip()))
                params.append(SequenceParam(layer: idx, name: "kernel bias", grid: bias.grid, shape: bias.shape.strip()))
            }
        }
        return try? JSONEncoder().encode(params)
    }
    
    public func decode_params(_ data: Data) {
        if let params = try? JSONDecoder().decode([SequenceParam<Tensor>].self, from: data) {
            for param in params {
                let tensor = Tensor(shape: param.shape, grid: param.grid)
                if param.name.contains("weight") {
                    let layer = layers[param.layer] as! Linear
                    if param.name.contains("bias") {
                        layer.bias.out! = tensor
                    } else {
                        layer.weight.out! = tensor
                    }
                } else if param.name.contains("kernel") {
                    let layer = layers[param.layer] as! Conv2D
                    if param.name.contains("bias") {
                        layer.bias.out! = tensor
                    } else {
                        layer.kernel.out! = tensor
                    }
                } else {
                    print("Unknown param")
                }
            }
        } else {
            print("Unable to decode params")
        }
    }
}
public struct SequenceParam<Tensor: Tensorable>: Codable {
    public var layer: Int
    public var name: String
//    public var idx: Int
    public var grid: [Tensor.Scalar]
    public var shape: [Int]
}

// MARK: Layer
public class Layer<Tensor: Tensorable>: Variable<Tensor> {
    
    final var input: Variable<Tensor> {
        get {
            return inputs[0]
        }
        set {
            inputs[0] = newValue
        }
    }
    
    public init(inputs: [Variable<Tensor>], tag: String = "") {
        precondition(inputs.count >= 1, "A layer must have at least one input")
        super.init(inputs: inputs, tag: tag)
        self.type = .operation
    }
}

// MARK: Linear
public final class Linear<Tensor: Tensorable>: Layer<Tensor> {
    
    private var combination: Tensor?
    
    public var weight: Variable<Tensor> {
        return inputs[1]
    }
    public var bias: Variable<Tensor> {
        return inputs[2]
    }
    
    // Placeholder in case input currently uknown (first input, or we haven't fed forward)
    public init(_ input: Variable<Tensor> = Placeholder(), to: Int, out: Int, tag: String = "") {
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
            gradCombination = gradCombination.sum(axis: 0, keepDim: false)
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
public final class Sigmoid<Tensor: Tensorable>: Layer<Tensor> {
    
    // Placeholder in case input currently unknown (we haven't fed forward)
    public init(_ input: Variable<Tensor> = Placeholder(), tag: String = "") {
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
public final class ReLU<Tensor: Tensorable>: Layer<Tensor> {
    
    // Placeholder in case input currently unknown (we haven't fed forward)
    public init(_ input: Variable<Tensor> = Placeholder(), tag: String = "") {
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
public final class LReLU<Tensor: Tensorable>: Layer<Tensor> {
    
    // Placeholder in case input currently unknown (we haven't fed forward)
    public init(_ input: Variable<Tensor> = Placeholder(), tag: String = "") {
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
public final class BatchNorm<Tensor: Tensorable>: Layer<Tensor> {
    
    private var data_norm: Tensor?
    private var running_mean: Tensor?
    private var running_std: Tensor?
    private var sample_std: Tensor?
    private var sample_mean: Tensor?
    private var momentum: Tensor.Scalar
    
    public var training: Bool
    
    public var gamma: Variable<Tensor> {
        return inputs[1]
    }
    public var beta: Variable<Tensor> {
        return inputs[2]
    }
    
    // Placeholder in case input currently unknown (we haven't fed forward)
    public init(_ input: Variable<Tensor> = Placeholder(), to: Int, momentum: Tensor.Scalar = 0.99, training: Bool = true, tag: String = "") {
        let gamma = Variable(Tensor(shape: [to, 1], repeating: 1))
        let beta = Variable(Tensor(shape: [to, 1], repeating: 0.01))
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
            self.sample_std = sample_std
            self.sample_mean = sample_mean
        } else {
            let data_norm = (data - self.running_mean!.transpose()) / self.running_std!.transpose()
            out = gamma * data_norm + beta
        }
    }
    
    public override func backward(dOut: Tensor?) {
        // Clarify inputs
        let data = inputs[0].out!
        let gamma = inputs[1].out!
        let sample_mean = sample_mean!.transpose()
        let sample_std = sample_std!.transpose()
        let data_norm = data_norm!
        let data_centered = data - sample_mean
        
        // Duplicate use
        let m = Tensor.Scalar(data.shape[1])
        let div_std = 1.0 / (sample_std + 0.0001).sqrt()
        let dub_data_centered = 2.0 * data_centered
        // Get grad for data
        let gradData_norm = dOut! * gamma
        // grad std calculation
        let gradStd = (gradData_norm * data_centered).sum(axis: 1, keepDim: true) * (-0.5 * (sample_std + 0.0001).pow(-1.5))
//        let gradStd = (gradData_norm * data_centered * (-0.5 * (sample_std + 0.0001).pow(-1.5))).sum(axis: 1, keepDim: true)
        // grad mean calculation
        let gradMean1 = (gradData_norm * (-1.0 * div_std)).sum(axis: 1, keepDim: true)
        let gradMean2 = gradStd * (-dub_data_centered).sum(axis: 1, keepDim: true) / m
        let gradMean = gradMean1 + gradMean2
        // final grad data calculation
        let gradData1 = gradData_norm * div_std
        let gradData2 = gradStd * dub_data_centered / m
        let gradData3 = gradMean / m
        let gradData = gradData1 + gradData2 + gradData3
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

// MARK: BatchNorm2D
public final class BatchNorm2D<Tensor: Tensorable>: Layer<Tensor> {
    
    private var data_norm: Tensor?
    fileprivate var running_mean: Tensor?
    fileprivate var running_std: Tensor?
    private var sample_std: Tensor?
    private var sample_mean: Tensor?
    private var momentum: Tensor.Scalar
    
    public var training: Bool
    
    public var gamma: Variable<Tensor> {
        return inputs[1]
    }
    public var beta: Variable<Tensor> {
        return inputs[2]
    }
    
    // Placeholder in case input currently unknown (we haven't fed forward)
    public init(_ input: Variable<Tensor> = Placeholder(), to: Int, momentum: Tensor.Scalar = 0.99, training: Bool = true, tag: String = "") {
        let gamma = Variable(Tensor(shape: [to], repeating: 1))
        let beta = Variable(Tensor(shape: [to], repeating: 0.01))
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
            let (data_norm, sample_mean, sample_std) = data.zscore_image()
            out = cadd(cmul(data_norm, with: gamma), with: beta)
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
            self.sample_mean = sample_mean
        } else {
            let data_norm = data.zscore_image_norm(mean: self.running_mean!, std: self.running_std!)
            out = cadd(cmul(data_norm, with: gamma), with: beta)
        }
    }
    
    public override func backward(dOut: Tensor?) {
        // Clarify inputs
        let data = inputs[0].out!
        let gamma = inputs[1].out!
        let sample_mean = sample_mean!
        let sample_std = sample_std!
        let data_norm = data_norm!
        let data_centered = cadd(data, with: -1.0 * sample_mean)

        // Duplicate use
        let m = Tensor.Scalar(data.shape[0] * data.shape[2] * data.shape[3])
        let div_std = 1.0 / (sample_std + 0.0001).sqrt()
        let dub_data_centered = 2.0 * data_centered
        // Get grad for data
        let gradData_norm = cmul(dOut!, with: gamma)
        // grad std calculation
        let gradStd = (gradData_norm * data_centered).sum(axis: 0, keepDim: false).sum(axes: 1...2, keepDim: false) * -0.5 * (sample_std + 0.0001).pow(-1.5)
//        let gradStd = (gradData_norm * cmul(data_centered, with: -0.5 * (sample_std + 0.0001).pow(-1.5))).sum(axis: 0, keepDim: false).sum(axes: 1...2, keepDim: false)
        // grad mean calculation
        let gradMean1 = cmul(gradData_norm, with: -1.0 * div_std).sum(axis: 0, keepDim: false).sum(axes: 1...2, keepDim: false)
        let gradMean2 = ((-dub_data_centered).sum(axis: 0, keepDim: false).sum(axes: 1...2, keepDim: false) / m) * gradStd
//        let gradMean2 = cmul((-dub_data_centered).sum(axis: 0, keepDim: false), with: gradStd).sum(axes: 1...2, keepDim: false) / m
        let gradMean = (gradMean1 + gradMean2)
        // final grad data calculation
        let gradData1 = cmul(gradData_norm, with: div_std)
        let gradData2 = cmul(dub_data_centered, with: gradStd) / m
        let gradData3 = gradMean / m
        let gradData = cadd(gradData1 + gradData2, with: gradData3)
        // Get grad for gamma
        let gradGamma = (dOut! * data_norm).sum(axis: 0, keepDim: false).sum(axes: 1...2, keepDim: false)
        // Get grad for beta
        let gradBeta = dOut!.sum(axis: 0, keepDim: false).sum(axes: 1...2, keepDim: false)
        // Now set our grads
        grads[0] = gradData
        grads[1] = gradGamma
        grads[2] = gradBeta
    }
    
    private func cmul(_ nd: Tensor, with vec: Tensor) -> Tensor {
        if nd.shape.count == 4 && vec.shape.count == 1 && vec.shape[0] == nd.shape[1] {
            var t = nd
            for n in 0..<nd.shape[0] {
                let nthImage = nd[t3D: n]
                for d in 0..<nd.shape[1] {
                    t[t3D: n][mat: d] = nthImage[mat: d] * vec[val: d]
                }
            }
            return t
        } else if nd.shape.count == 3 && vec.shape.count == 1 && vec.shape[0] == nd.shape[0] {
            var t = nd
            for d in 0..<nd.shape[0] {
                t[mat: d] = nd[mat: d] * vec[val: d]
            }
            return t
        }
        fatalError("Incompatible type for channel wise multiplication")
    }
    
    private func cadd(_ nd: Tensor, with vec: Tensor) -> Tensor {
        if nd.shape.count == 4 && vec.shape.count == 1 && vec.shape[0] == nd.shape[1] {
            var t = nd
            for n in 0..<nd.shape[0] {
                let nthImage = nd[t3D: n]
                for d in 0..<nd.shape[1] {
                    t[t3D: n][mat: d] = nthImage[mat: d] + vec[val: d]
                }
            }
            return t
        } else if nd.shape.count == 3 && vec.shape.count == 1 && vec.shape[0] == nd.shape[0] {
            var t = nd
            for d in 0..<nd.shape[0] {
                t[mat: d] = nd[mat: d] + vec[val: d]
            }
            return t
        }
        fatalError("Incompatible type for channel wise addition")
    }
}

// MARK: Conv2D
public final class Conv2D<Tensor: Tensorable>: Layer<Tensor> {
    
    public var kernel: Variable<Tensor> {
        return inputs[1]
    }
    
    public var bias: Variable<Tensor> {
        return inputs[2]
    }
    
    private let pad: Bool
    
    // Placeholder in case input currently unknown (we haven't fed forward)
    public init(_ input: Variable<Tensor> = Placeholder(), to: Int, out: Int, size: Int, pad: Bool = false, tag: String = "") {
        // Check if this is really xavier
        let kernel = Variable(Tensor.random_xavier(shape: [out, to, size, size], ni: size * size * to, no: size * size * out))
        let bias = Variable(Tensor(shape: [out], repeating: 0.01))
        self.pad = pad
        super.init(inputs: [input, kernel, bias], tag: tag)
        grads = Array(repeating: Tensor(shape: [], grid: []), count: 3)
    }
    
    public override func forward() {
        // Clarify inputs
        let data = inputs[0].out!
        let kernel = inputs[1].out!
        let bias = inputs[2].out!
        // Convolve!
        if kernel.shape.count == 4 && data.shape.count == 3 && kernel.shape[1] == data.shape[0] {
            // First get the shape of our 2D Tensor after convolution
            // pad to make shape a same convolution if we have padding
            let mat_shape = pad ? Array(data.shape[1...2]).pad_shape((kernel.shape[2] - 1) / 2, (kernel.shape[3] - 1) / 2).conv2D_shape(with: Array(kernel.shape[2...3]), type: .valid) : Array(data.shape[1...2]).conv2D_shape(with: Array(kernel.shape[2...3]), type: .valid)
            // Now we can make the out shape using the 2D Tensor (matrix) with a depth of the number of kernels since out must have the same depth as the number of kernels
            out = Tensor(shape: [kernel.shape[0], mat_shape[0], mat_shape[1]], repeating: 0.0)
            // Now for each kernel we convolve with our data to produce our dth depth for out
            for d in 0..<kernel.shape[0] {
                // Used repeatedly in following for loop so cache
                let dthKernel = kernel[t3D: d]
                // Now convolve this kernel with our data, since both kernel and data are 3D Tensors we convolve the corresponding depth of data with that of kernelD
                for m in 0..<kernel.shape[1] {
                    // pad to make same convolution if we have padding
                    out![mat: d] = pad ? out![mat: d] + data[mat: m].pad((kernel.shape[2] - 1) / 2, (kernel.shape[3] - 1) / 2).conv2D(with: dthKernel[mat: m], type: .valid) : out![mat: d] + data[mat: m].conv2D(with: dthKernel[mat: m], type: .valid)
                }
                // Add the bias for the dth depth of out which corresponds to the dth kernel
                out![mat: d] = out![mat: d] + bias[d]
            }
            // For more clarity, this is essentially the following forward calculation (One kernel with depth > 1) except we have multiple kernels that contribute to out so we need to do convolutions for each kernel with data which will produce a new depth for out (making out a multi depth output)
        } else if kernel.shape.count == 4 && data.shape.count == 4 && kernel.shape[1] == data.shape[1] {
            // First get the shape of our 2D Tensor after convolution
            // pad to make shape a same convolution if we have padding
            let mat_shape = pad ? Array(data.shape[2...3]).pad_shape((kernel.shape[2] - 1) / 2, (kernel.shape[3] - 1) / 2).conv2D_shape(with: Array(kernel.shape[2...3]), type: .valid) : Array(data.shape[2...3]).conv2D_shape(with: Array(kernel.shape[2...3]), type: .valid)
            // Now we can make the out shape using the 2D Tensor (matrix) with a depth of the number of kernels since out must have the same depth as the number of kernels and a count of the number of input images since we are outting the same number of images
            out = Tensor(shape: [data.shape[0], kernel.shape[0], mat_shape[0], mat_shape[1]], repeating: 0.0)
            // Now for each image we do a convolution
            for n in 0..<data.shape[0] {
                // Used repeatedly in following for loop so cache
                let nthData = data[t3D: n]
                // Now for each kernel we convolve with our data to produce our dth depth for out
                for d in 0..<kernel.shape[0] {
                    // Used repeatedly in following for loop so cache
                    let dthKernel = kernel[t3D: d]
                    // Now convolve this kernel with our data, since both kernel and data are 3D Tensors we convolve the corresponding depth of data with that of dth kernel
                    for m in 0..<kernel.shape[1] {
                        // pad to make same convolution if we have padding
                        out![t3D: n][mat: d] = pad ? out![t3D: n][mat: d] + nthData[mat: m].pad((kernel.shape[2] - 1) / 2, (kernel.shape[3] - 1) / 2).conv2D(with: dthKernel[mat: m], type: .valid) : out![t3D: n][mat: d] + nthData[mat: m].conv2D(with: dthKernel[mat: m], type: .valid)
                    }
                    // Add the bias for the dth depth of out which corresponds to the dth kernel
                    out![t3D: n][mat: d] = out![t3D: n][mat: d] + bias[d]
                }
            }
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
        // Convolve! Backwards...? D;
        if kernel.shape.count == 4 && data.shape.count == 3 && kernel.shape[1] == data.shape[0] {
            // *** GRADDATA ***
            // gradData should be the same shape as our data
            gradData = Tensor(shape: data.shape, repeating: 0.0)
            // Each depth of gradData is influenced by every kernel that was convoluted over data so we must convolve each depth of dOut with the corresponding depth of each kernel
            for d in 0..<kernel.shape[0] {
                // Used repeatedly in following for loop so cache
                let dthKernel = kernel[t3D: d]
                let dthDout = dOut![mat: d]
                // For each kernel we are adding, its mth depth convolved with the respective dth dOut that is influenced by this kernel, with the respective gradData mth depth
                for m in 0..<kernel.shape[1] {
                    // Each kernel only influences a single depth of dOut, but each depth of the kernel influences each depth of gradData, leading to this syntax
                    // trim gradData since padded areas dont contribute to gradient if we have padding
                    gradData[mat: m] = pad ? gradData[mat: m] + dthKernel[mat: m].conv2D(with: dthDout, type: .full).trim((kernel.shape[2] - 1) / 2, (kernel.shape[3] - 1) / 2) : gradData[mat: m] + dthKernel[mat: m].conv2D(with: dthDout, type: .full)
                }
            }
            // For more clarity look at the previous github commit
            
            // *** GRADKERNEL ***
            // gradKernel should be the same shape as our kernels tensor
            gradKernel = Tensor(shape: kernel.shape, repeating: 0)
            // Now we calculate the influence of each kernel
            for d in 0..<kernel.shape[0] {
                // Used repeatedly in following for loop so cache
                let dthDout = dOut![mat: d]
                // For a single kernel its mth depths influence is found by the convolution with the mth depth of the data (since this depth only convolves with the respective data mth depth) and the dOut depth d that corresponds to this kernel (since each kernel only influences on depth of dOut)
                for m in 0..<kernel.shape[1] {
                    // Here we are finding the partial derivative for the mth depth of kernel d
                    // pad data just as we pad in forward if we have padding
                    gradKernel[t3D: d][mat: m] = pad ? data[mat: m].pad((kernel.shape[2] - 1) / 2, (kernel.shape[3] - 1) / 2).conv2D(with: dthDout, type: .valid) : data[mat: m].conv2D(with: dthDout, type: .valid)
                }
            }
            // For more clarity look at the previous github commit
            
            // *** GRADBIAS ***
            gradBias = bias
            // Do you realy need an explanation?
            for d in 0..<kernel.shape[0] {
                gradBias[d] = dOut![mat: d].sum()
            }
        } else if kernel.shape.count == 4 && data.shape.count == 4 && kernel.shape[1] == data.shape[1] {
            // *** GRADDATA ***
            // gradData should be the same shape as our data
            gradData = Tensor(shape: data.shape, repeating: 0.0)
            // Average each images dOut
            for n in 0..<data.shape[0] {
                // Used repeatedly in following for loop so cache
                let nthDout = dOut![t3D: n]
                // Each depth of gradData is influenced by every kernel that was convoluted over data so we must convolve each depth of dOut with the corresponding depth of each kernel
                for d in 0..<kernel.shape[0] {
                    // Used repeatedly in following for loop so cache
                    let dthKernel = kernel[t3D: d]
                    let nthDthDout = nthDout[mat: d]
                    // For each kernel we are adding, its mth depth convolved with the respective dth dOut that is influenced by this kernel, with the respective gradData mth depth
                    for m in 0..<kernel.shape[1] {
                        // Each kernel only influences a single depth of dOut, but each depth of the kernel influences each depth of gradData, leading to this syntax
                        // trim gradData since padded areas dont contribute to gradient if we have padding
                        gradData[t3D: n][mat: m] = pad ? gradData[t3D: n][mat: m] + dthKernel[mat: m].conv2D(with: nthDthDout, type: .full).trim((kernel.shape[2] - 1) / 2, (kernel.shape[3] - 1) / 2) : gradData[t3D: n][mat: m] + dthKernel[mat: m].conv2D(with: nthDthDout, type: .full)
                    }
                }
            }
            // For more clarity look at the previous github commit
            
            // *** GRADKERNEL ***
            // gradKernel should be the same shape as our kernels tensor
            gradKernel = Tensor(shape: kernel.shape, repeating: 0)
            // Average each images dOut
            for n in 0..<data.shape[0] {
                // Used repeatedly in following for loop so cache
                let nthData = data[t3D: n]
                let nthDout = dOut![t3D: n]
                // Now we calculate the influence of each kernel
                for d in 0..<kernel.shape[0] {
                    // Used repeatedly in following for loop so cache
                    let dthGradKernel = gradKernel[t3D: d]
                    let nthDthDout = nthDout[mat: d]
                    // For a single kernel its mth depths influence is found by the convolution with the mth depth of the data (since this depth only convolves with the respective data mth depth) and the dOut depth d that corresponds to this kernel (since each kernel only influences on depth of dOut)
                    for m in 0..<kernel.shape[1] {
                        // Here we are finding the partial derivative for the mth depth of kernel d
                        // pad data just as we pad in forward if we have padding
                        gradKernel[t3D: d][mat: m] = pad ? dthGradKernel[mat: m] + nthData[mat: m].pad((kernel.shape[2] - 1) / 2, (kernel.shape[3] - 1) / 2).conv2D(with: nthDthDout, type: .valid) : dthGradKernel[mat: m] + nthData[mat: m].conv2D(with: nthDthDout, type: .valid)
                    }
                }
            }
            // For more clarity look at the previous github commit
            
            // *** GRADBIAS ***
            gradBias = Tensor(shape: bias.shape, repeating: 0.0)
            // Average each images dOut
            for n in 0..<data.shape[0] {
                // Used repeatedly in following for loop so cache
                let nthDout = dOut![t3D: n]
                // Do you realy need an explanation?
                for d in 0..<kernel.shape[0] {
                    gradBias[d] = gradBias[d] + nthDout[mat: d].sum()
                }
            }
        } else {
            fatalError("Data and kernels are incompatible")
        }
        // Now set our grads
        grads[0] = gradData
        grads[1] = gradKernel
        grads[2] = gradBias
    }
}

// MARK: Pool2DMax
public final class Pool2DMax<Tensor: Tensorable>: Layer<Tensor> {
    
    private let size: Int
    private let stride: Int
    private var n_positions: [[[Int]]]?
    
    // Placeholder in case input currently unknown (we haven't fed forward)
    public init(_ input: Variable<Tensor> = Placeholder(), size: Int, stride: Int, tag: String = "") {
        self.size = size
        self.stride = stride
        super.init(inputs: [input], tag: tag)
        grads = Array(repeating: Tensor(shape: [], grid: []), count: 1)
    }
    
    public override func forward() {
        // Clarify inputs
        let input = inputs[0].out!
        // Max pool!
        if input.shape.count == 4 {
            // Make empty out same size as input after pool, mat_shape is the shape of a matrix in our tensor
            let mat_shape = Array(input.shape[2...3]).pool2D_max_shape(size: size, strd: stride)
            // Now we can make the out shape using the 2D Tensor (matrix) with a depth of our input depth since pooling always maintains depth and count of our input count
            out = Tensor(shape: [input.shape[0], input.shape[1], mat_shape[0], mat_shape[1]], repeating: 0.0)
            // positions will have 2D arrays as long as input count
            n_positions = Array(repeating: Array(repeating: [], count: input.shape[1]), count: input.shape[0])
            // Now for each nth image pool
            for n in 0..<input.shape[0] {
                // Used repeatedly in following for loop so cache
                let nthInput = input[t3D: n]
                // Now pool each depth in the nth image input
                for d in 0..<input.shape[1] {
                    // Pool and get positions for backpropagation
                    let (pooled, positions) = nthInput[mat: d].pool2D_max(size: size, strd: stride)
                    out![t3D: n][mat: d] = pooled
                    // Cache this depths positions
                    n_positions![n][d] = positions
                }
            }
        } else if input.shape.count == 3 {
            // Make empty out same size as input after pool, mat_shape is the shape of a matrix in our tensor
            let mat_shape = Array(input.shape[1...2]).pool2D_max_shape(size: size, strd: stride)
            // Now we can make the out shape using the 2D Tensor (matrix) with a depth of our input depth since pooling always maintains depth
            out = Tensor(shape: [input.shape[0], mat_shape[0], mat_shape[1]], repeating: 0.0)
            // positions will only have one 2D array
            n_positions = Array(repeating: Array(repeating: [], count: input.shape[0]), count: 1)
            // Now pool each depth in input
            for d in 0..<input.shape[0] {
                // Pool and get positions for backpropagation
                let (pooled, positions) = input[mat: d].pool2D_max(size: size, strd: stride)
                out![mat: d] = pooled
                // Cache this depths position
                n_positions![0][d] = positions
            }
        } else {
            fatalError("Incompatible dimensions for pooling")
        }
    }
    
    public override func backward(dOut: Tensor?) {
        // Clarify inputs
        let input = inputs[0].out!
        // Backprop
        if input.shape.count == 4 {
            // Make empty tensor of input shape
            var gradInput = Tensor(shape: input.shape, repeating: 0.0)
            // For each input image use its cached positions
            for n in 0..<n_positions!.count {
                // Used repeatedly in following for loop so cache
                let nthGradInput = gradInput[t3D: n]
                let nthDout = dOut![t3D: n]
                // The cached positions contains where our gradients in this nth gradInput should be
                for (d, positions) in n_positions![n].enumerated() {
                    // Used repeatedly in following for loop so cache
                    var nthDthGradInput = nthGradInput[mat: d]
                    let nthDthDout = nthDout[mat: d]
                    for (idx, pos) in positions.enumerated() {
                        // positions is as long as the grid of the dth depth of the nth dOut and each index in positions corresponds to a pos in the grid of the dth depth of the nth gradInput that should receive a derivative
                        nthDthGradInput.grid[pos] = nthDthDout.grid[idx]
                    }
                    gradInput[t3D: n][mat: d] = nthDthGradInput
                }
            }
            // Set grad
            grads[0] = gradInput
        } else if input.shape.count == 3 {
            // Make empty tensor of input shape
            var gradInput = Tensor(shape: input.shape, repeating: 0.0)
            // The cached position contains where our gradients in gradInput should be
            for (d, positions) in n_positions![0].enumerated() {
                // Used repeatedly in following for loop so cache
                var dthGradInput = gradInput[mat: d]
                let dthDout = dOut![mat: d]
                for (idx, pos) in positions.enumerated() {
                    // positions is as long as the grid of the dth depth of dOut and each index in positions corresponds to a pos in the grid of the dth depth gradInput that should receive a derivative
                    dthGradInput.grid[pos] = dthDout.grid[idx]
                }
                gradInput[mat: d] = dthGradInput
            }
            // Set grad
            grads[0] = gradInput
        } else {
            fatalError("Incompatible dimensions for pooling")
        }
    }
}

// MARK: Flatten
public final class Flatten<Tensor: Tensorable>: Layer<Tensor> {
    
    private var input_shape: Shape?
    
    // Placeholder in case input currently unknown (we haven't fed forward)
    public init(_ input: Variable<Tensor> = Placeholder(), tag: String = "") {
        super.init(inputs: [input], tag: tag)
        grads = Array(repeating: Tensor(shape: [], grid: []), count: 1)
    }
    
    public override func forward() {
        // Clarify inputs
        let input = inputs[0].out!
        // If 4D-Tensor make into matrix
        // If 3D-Tensor make into vector
        if input.shape.count == 4 {
            // Rows is number of images, columns is all other dimensions multiplied
            // Transpose to fit linear layer
            out = Tensor(shape: [input.shape[0], input.shape[1] * input.shape[2] * input.shape[3]], grid: input.grid).transpose()
            // Cache input shape
            input_shape = input.shape
        } else if input.shape.count == 3 {
            // Rows is 1, columns is all dimensions multiplied
            // Transpose to fit linear layer
            out = Tensor(shape: [1, input.shape[0] * input.shape[1] * input.shape[2]], grid: input.grid).transpose()
            // Cache input shape
            input_shape = input.shape
        } else {
            fatalError("Input must be 4D or 3D Tensor")
        }
    }
    
    public override func backward(dOut: Tensor?) {
        // Grad, transpose dOut since we transpose in forward
        let gradInput = Tensor(shape: input_shape!, grid: dOut!.transpose().grid)
        // Set grad
        grads[0] = gradInput
    }
}

public enum ConvType {
    case valid
    case same
}

// MARK: Process
public final class Process<Tensor: Tensorable> {
    
    public func shuffle(data: [[Tensor.Scalar]], labels: [[Tensor.Scalar]]) -> (shuffledData: [[Tensor.Scalar]], shuffledLabels: [[Tensor.Scalar]]) {
        var merged = [[Tensor.Scalar]]()
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
        var shuffledData = [[Tensor.Scalar]]()
        var shuffledLabels = [[Tensor.Scalar]]()
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
    
    public func zscore_image(_ input: Tensor, type: ProcessType) -> Tensor {
        switch type {
        case .data:
            let (norm, mean, std) = input.zscore_image()
            self.mean = mean
            self.std = std
            return norm
        case .pred:
            return input.zscore_image_norm(mean: mean, std: std)
        }
    }
}

public enum ProcessType {
    case data
    case pred
}
