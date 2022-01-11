//
//  Metal.swift
//  
//
//  Created by Sahil Srivastava on 1/9/22.
//

import Foundation
import MetalPerformanceShaders
import Accelerate

@available(macOS 10.13, *)
func prepare_metal() -> (device: MTLDevice, commandQueue: MTLCommandQueue) {
    let device = MTLCreateSystemDefaultDevice()!
    let commandQueue = device.makeCommandQueue()!
    return (device, commandQueue)
}
@available(macOS 10.13, *)
func matrix_mul_metal(a: FTensor, b: FTensor, device: MTLDevice, commandBuffer: MTLCommandBuffer) -> FTensor {
    var a_shape = a.shape
    var b_shape = b.shape
    if a.shape.count == 1 {
        a_shape.insert(1, at: 0)
    }
    if b.shape.count == 1 {
        b_shape.insert(1, at: 0)
    }
    precondition(a_shape.count == 2 && b_shape.count == 2, "Tensor lhs \(a.shape) and Tensor rhs \(b.shape) not compatible for matrix multiplication")
    precondition(a_shape[1] == b_shape[0], "Matrix lhs \(a.shape) and Matrix rhs \(b.shape) not compatible for matrix multiplication")
    
    let mmKernel = MPSMatrixMultiplication(device: device, transposeLeft: false, transposeRight: false, resultRows: a_shape[0], resultColumns: b_shape[1], interiorColumns: a_shape[1], alpha: 1.0, beta: 0.0)
    
    var res = FTensor(shape: [a_shape[0], b_shape[1]], repeating: 0.0)
    let res_N = res.grid.count
    let a_N = a.grid.count
    let b_N = b.grid.count
    a.grid.withUnsafeBufferPointer { aPtr in
        b.grid.withUnsafeBufferPointer { bPtr in
            let totalBytesA = MemoryLayout<Float32>.stride * a_N
            let bufferA = device.makeBuffer(bytes: a.grid, length: totalBytesA, options: .storageModeShared)
            let descriptorA = MPSMatrixDescriptor(rows: a_shape[0], columns: a_shape[1], rowBytes: totalBytesA / a_shape[0], dataType: .float32)
            let A = MPSMatrix(buffer: bufferA!, descriptor: descriptorA)
            
            let totalBytesB = MemoryLayout<Float32>.stride * b_N
            let bufferB = device.makeBuffer(bytes: b.grid, length: totalBytesB, options: .storageModeShared)
            let descriptorB = MPSMatrixDescriptor(rows: b_shape[0], columns: b_shape[1], rowBytes: totalBytesB / b_shape[0], dataType: .float32)
            let B = MPSMatrix(buffer: bufferB!, descriptor: descriptorB)
            
            let totalBytesC = MemoryLayout<Float32>.stride * A.rows * B.columns
            let bufferC = device.makeBuffer(length: totalBytesC, options: .storageModeShared)
            let descriptorC = MPSMatrixDescriptor(rows: A.rows, columns: B.columns, rowBytes: totalBytesC / A.rows, dataType: .float32)
            let C = MPSMatrix(buffer: bufferC!, descriptor: descriptorC)
            
            mmKernel.encode(commandBuffer: commandBuffer, leftMatrix: A, rightMatrix: B, resultMatrix: C)
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
            
            let rawPointer = C.data.contents()
            let typePointer = rawPointer.bindMemory(to: Float32.self, capacity: A.rows * B.columns)
            let bufferPointer = UnsafeBufferPointer(start: typePointer, count: A.rows * B.columns)
            res.grid.withUnsafeMutableBufferPointer { resPtr in
                cblas_scopy(Int32(res_N), bufferPointer.baseAddress!, 1, resPtr.baseAddress!, 1)
            }
        }
    }
    return res
}
