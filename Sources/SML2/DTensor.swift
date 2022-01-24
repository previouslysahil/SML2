//
//  DTensor.swift
//
//
//  Created by Sahil Srivastava on 1/10/22.
//
//  IMPORTANT NOTES: Only math arithmetic ignores extra shape
//
//  This Tensor struct is heavily influenced by the Matrix
//  struct created by Matthijs Hollemans and Mattt Thompson,
//  I do not know how copyrights and that stuff works but
//  you can find their Matrix.swift file at this link:
//  https://github.com/hollance/Matrix/blob/master/Matrix.swift
//  thank you so much to both of you for writing code using
//  Accelerate that makes it incredibly easy to decipher
//  what each Accelerate method is doing, it really helped
//  write this struct (and just improve my understanding
//  of coding).
//
//  Handy References:
//  Fig 1.0: https://github.com/scikit-learn/scikit-learn/blob/7389dbac82d362f296dc2746f10e43ffa1615660/sklearn/preprocessing/data.py#L70
//  Fig 2.0: https://github.com/keras-team/keras/blob/v2.7.0/keras/layers/normalization/batch_normalization.py#L885
//  Fig 3.0: https://github.com/keras-team/keras/blob/998efc04eefa0c14057c1fa87cab71df5b24bf7e/keras/initializations.py#L41

import Foundation
import Accelerate

public struct DTensor: Equatable, Tensorable {
    public var shape: Shape
    public var grid: [Double]
    
    public init(shape: [Int], grid: [Double]) {
        self.init(shape: Shape(shape), grid: grid)
    }
    
    public init(shape: Shape, grid: [Double]) {
        self.shape = shape
        self.grid = grid
    }
}
// MARK: - Inits
extension DTensor {
    public init(shape: [Int], repeating: Double) {
        self.init(shape: Shape(shape), repeating: repeating)
    }
    public init(shape: Shape, repeating: Double) {
        self.shape = shape
        self.grid = Array(repeating: repeating, count: shape.reduce())
    }
    public init(_ val: Double) {
        self.init(shape: [], repeating: val)
    }
    public init(_ vec: [Double]) {
        self.init(shape: [vec.count], repeating: 0.0)
        vec.withUnsafeBufferPointer { vecPtr in
            grid.withUnsafeMutableBufferPointer { gridPtr in
                cblas_dcopy(Int32(shape[0]), vecPtr.baseAddress!, 1, gridPtr.baseAddress!, 1)
            }
        }
    }
    public init(_ vec: [Double], type: VectorType) {
        switch type {
        case .row:
            self.init(shape: [1, vec.count], repeating: 0.0)
            vec.withUnsafeBufferPointer { vecPtr in
                grid.withUnsafeMutableBufferPointer { gridPtr in
                    cblas_dcopy(Int32(shape[1]), vecPtr.baseAddress!, 1, gridPtr.baseAddress!, 1)
                }
            }
        case .column:
            self.init(shape: [vec.count, 1], repeating: 0.0)
            vec.withUnsafeBufferPointer { vecPtr in
                grid.withUnsafeMutableBufferPointer { gridPtr in
                    cblas_dcopy(Int32(shape[0]), vecPtr.baseAddress!, 1, gridPtr.baseAddress!, 1)
                }
            }
        }
    }
    public init(_ mat: [[Double]]) {
        self.init(shape: [mat.count, mat.first!.count], repeating: 0.0)
        var i = 0
        for row in mat {
            row.withUnsafeBufferPointer { rowPtr in
                grid.withUnsafeMutableBufferPointer { gridPtr in
                    cblas_dcopy(Int32(shape[1]), rowPtr.baseAddress!, 1, gridPtr.baseAddress! + i * shape[1], 1)
                }
            }
            i += 1
        }
    }
    public init(_ t3D: [[[Double]]]) {
        self.init(shape: [t3D.count, t3D.first!.count, t3D.first!.first!.count], repeating: 0.0)
        var j = 0
        for mat in t3D {
            for row in mat {
                row.withUnsafeBufferPointer { rowPtr in
                    grid.withUnsafeMutableBufferPointer { gridPtr in
                        cblas_dcopy(Int32(shape[2]), rowPtr.baseAddress!, 1, gridPtr.baseAddress! + j * shape[2], 1)
                    }
                }
                j += 1
            }
        }
    }
    public init(_ t4D: [[[[Double]]]]) {
        self.init(shape: [t4D.count, t4D.first!.count, t4D.first!.first!.count, t4D.first!.first!.first!.count], repeating: 0.0)
        var j = 0
        for t3D in t4D {
            for mat in t3D {
                for row in mat {
                    row.withUnsafeBufferPointer { rowPtr in
                        grid.withUnsafeMutableBufferPointer { gridPtr in
                            cblas_dcopy(Int32(shape[3]), rowPtr.baseAddress!, 1, gridPtr.baseAddress! + j * shape[3], 1)
                        }
                    }
                    j += 1
                }
            }
        }
    }
}
// MARK: - Query
extension DTensor {
    public subscript(pos: Int...) -> Double {
        get {
            precondition(pos.count == shape.count, "Position must account for all dimensions of our tensor")
            var idx = 0
            for i in (0..<pos.count).reversed() {
                if i == pos.count - 1 {
                    idx += pos[i]
                } else {
                    idx += pos[i] * shape[(i + 1)..<shape.count].reduce(1) { $0 * $1 }
                }
            }
            return grid[idx]
        }
        set {
            precondition(pos.count == shape.count, "Position must account for all dimensions of our tensor")
            var idx = 0
            for i in (0..<pos.count).reversed() {
                if i == pos.count - 1 {
                    idx += pos[i]
                } else {
                    idx += pos[i] * shape[(i + 1)..<shape.count].reduce(1) { $0 * $1 }
                }
            }
            grid[idx] = newValue
        }
    }
    public subscript(t3D d: Int) -> DTensor {
        get {
            precondition(shape.count == 4, "Must be a 4D-Tensor")
            var t = DTensor(shape: [shape[1], shape[2], shape[3]], repeating: 0)
            grid.withUnsafeBufferPointer { gridPtr in
                t.grid.withUnsafeMutableBufferPointer { tPtr in
                    cblas_dcopy(Int32(shape[1] * shape[2] * shape[3]), gridPtr.baseAddress! + d * (shape[1] * shape[2] * shape[3]), 1, tPtr.baseAddress!, 1)
                }
            }
            return t
        }
        set(t) {
            precondition(shape.count == 4, "Must be a 4D-Tensor")
            precondition(t.shape.count == 3 && t.shape[0] == shape[1] && t.shape[1] == shape[2] && t.shape[2] == shape[3], "Not compatible tensor dimensions")
            grid.withUnsafeMutableBufferPointer { gridPtr in
                t.grid.withUnsafeBufferPointer { tPtr in
                    cblas_dcopy(Int32(shape[1] * shape[2] * shape[3]), tPtr.baseAddress!, 1, gridPtr.baseAddress! + d * (shape[1] * shape[2] * shape[3]), 1)
                }
            }
        }
    }
    public subscript(t3Ds range: Range<Int>) -> DTensor {
        get {
            precondition(shape.count == 4, "Must be a 4D-Tensor")
            precondition(range.lowerBound >= 0 && range.upperBound <= shape[0], "Invalid range")
            var t = DTensor(shape: [(range.upperBound - range.lowerBound), shape[1], shape[2], shape[3]], repeating: 0)
            grid.withUnsafeBufferPointer { gridPtr in
                t.grid.withUnsafeMutableBufferPointer { tPtr in
                    cblas_dcopy(Int32(shape[1] * shape[2] * shape[3] * range.count), gridPtr.baseAddress! + range.lowerBound * (shape[1] * shape[2] * shape[3]), 1, tPtr.baseAddress!, 1)
                }
            }
            return t
        }
    }
    public subscript(t3Ds range: ClosedRange<Int>) -> DTensor {
        get {
            self[t3Ds: Range(range)]
        }
    }
    public subscript(mats range: Range<Int>) -> DTensor {
        get {
            precondition(shape.count == 3, "Must be a 4D-Tensor")
            precondition(range.lowerBound >= 0 && range.upperBound <= shape[0], "Invalid range")
            var t = DTensor(shape: [(range.upperBound - range.lowerBound), shape[1], shape[2]], repeating: 0)
            grid.withUnsafeBufferPointer { gridPtr in
                t.grid.withUnsafeMutableBufferPointer { tPtr in
                    cblas_dcopy(Int32(shape[1] * shape[2] * range.count), gridPtr.baseAddress! + range.lowerBound * (shape[1] * shape[2]), 1, tPtr.baseAddress!, 1)
                }
            }
            return t
        }
    }
    public subscript(mats range: ClosedRange<Int>) -> DTensor {
        get {
            self[mats: Range(range)]
        }
    }
    public subscript(mat m: Int) -> DTensor {
        get {
            precondition(shape.count == 3, "Must be a 3D-Tensor")
            var t = DTensor(shape: [shape[1], shape[2]], repeating: 0)
            grid.withUnsafeBufferPointer { gridPtr in
                t.grid.withUnsafeMutableBufferPointer { tPtr in
                    cblas_dcopy(Int32(shape[1] * shape[2]), gridPtr.baseAddress! + m * (shape[1] * shape[2]), 1, tPtr.baseAddress!, 1)
                }
            }
            return t
        }
        set(t) {
            precondition(shape.count == 3, "Must be a 3D-Tensor")
            precondition(t.shape.count == 2 && t.shape[0] == shape[1] && t.shape[1] == shape[2], "Not compatible matrix dimensions")
            grid.withUnsafeMutableBufferPointer { gridPtr in
                t.grid.withUnsafeBufferPointer { tPtr in
                    cblas_dcopy(Int32(shape[1] * shape[2]), tPtr.baseAddress!, 1, gridPtr.baseAddress! + m * (shape[1] * shape[2]), 1)
                }
            }
        }
    }
    public subscript(row r: Int) -> DTensor {
        get {
            precondition(shape.count == 2, "Must be a matrix")
            var t = DTensor(shape: [1, shape[1]], repeating: 0)
            grid.withUnsafeBufferPointer { gridPtr in
                t.grid.withUnsafeMutableBufferPointer { tPtr in
                    cblas_dcopy(Int32(shape[1]), gridPtr.baseAddress! + r * shape[1], 1, tPtr.baseAddress!, 1)
                }
            }
            return t
        }
        set(t) {
            if t.shape.count == 1 && t.shape[0] == shape[1] {
                grid.withUnsafeMutableBufferPointer { gridPtr in
                    t.grid.withUnsafeBufferPointer { tPtr in
                        cblas_dcopy(Int32(shape[1]), tPtr.baseAddress!, 1, gridPtr.baseAddress! + r * shape[1], 1)
                    }
                }
            } else {
                precondition(shape.count == 2, "Must be a matrix")
                precondition(t.shape.count == 2 && t.shape[0] == 1 && t.shape[1] == shape[1], "Not compatible vector dimensions")
                grid.withUnsafeMutableBufferPointer { gridPtr in
                    t.grid.withUnsafeBufferPointer { tPtr in
                        cblas_dcopy(Int32(shape[1]), tPtr.baseAddress!, 1, gridPtr.baseAddress! + r * shape[1], 1)
                    }
                }
            }
        }
    }
    public subscript(rows range: Range<Int>) -> DTensor {
        get {
            precondition(shape.count == 2, "Must be a matrix")
            precondition(range.lowerBound >= 0 && range.upperBound <= shape[0], "Invalid range")
            var t = DTensor(shape: [(range.upperBound - range.lowerBound), shape[1]], repeating: 0)
            grid.withUnsafeBufferPointer { gridPtr in
                t.grid.withUnsafeMutableBufferPointer { tPtr in
                    cblas_dcopy(Int32(shape[1] * range.count), gridPtr.baseAddress! + range.lowerBound * shape[1], 1, tPtr.baseAddress!, 1)
                }
            }
            return t
        }
    }
    public subscript(rows range: ClosedRange<Int>) -> DTensor {
        get {
            self[rows: Range(range)]
        }
    }
    public subscript(col c: Int) -> DTensor {
        get {
            precondition(shape.count == 2, "Must be a matrix")
            var t = DTensor(shape: [shape[0], 1], repeating: 0)
            grid.withUnsafeBufferPointer { gridPtr in
                t.grid.withUnsafeMutableBufferPointer { tPtr in
                    cblas_dcopy(Int32(shape[0]), gridPtr.baseAddress! + c, Int32(shape[1]), tPtr.baseAddress!, 1)
                }
            }
            return t
        }
        set(t) {
            precondition(shape.count == 2, "Must be a matrix")
            precondition(t.shape.count == 2 && t.shape[1] == 1 && t.shape[0] == shape[0], "Not compatible vector dimensions")
            grid.withUnsafeMutableBufferPointer { gridPtr in
                t.grid.withUnsafeBufferPointer { tPtr in
                    cblas_dcopy(Int32(shape[0]), tPtr.baseAddress!, 1, gridPtr.baseAddress! + c, Int32(shape[1]))
                }
            }
        }
    }
    public subscript(cols range: Range<Int>) -> DTensor {
        precondition(shape.count == 2, "Must be a matrix")
        precondition(range.lowerBound >= 0 && range.upperBound <= shape[1], "Invalid range")
        var t = DTensor(shape: [shape[0], (range.upperBound - range.lowerBound)], repeating: 0)
        grid.withUnsafeBufferPointer { gridPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                for c in range {
                    cblas_dcopy(Int32(shape[0]), gridPtr.baseAddress! + c, Int32(shape[1]), tPtr.baseAddress! + c - range.lowerBound, Int32(t.shape[1]))
                }
            }
        }
        return t
    }
    public subscript(cols range: ClosedRange<Int>) -> DTensor {
        get {
            self[cols: Range(range)]
        }
    }
    public subscript(val v: Int) -> Double {
        get {
            precondition((shape.count == 2 && shape[1] == 1) || (shape.count == 2 && shape[0] == 1) || (shape.count == 1), "Must be a vector")
            return grid[v]
        }
        set(t) {
            precondition((shape.count == 2 && shape[1] == 1) || (shape.count == 2 && shape[0] == 1) || (shape.count == 1), "Must be a vector")
            grid[v] = t
        }
    }
}
// MARK: - Random Init
extension DTensor {
    public static func random(shape: [Int], min: Double = 0.0, max: Double = 1.0) -> DTensor {
        let count = shape.reduce(1) { $0 * $1 }
        var uniform = [Double]()
        uniform.reserveCapacity(count)
        for _ in 0..<count { uniform.append(Double.random(in: min..<max)) }
        let t = DTensor(shape: shape, grid: uniform)
        return t
    }
    public static func random_xavier(shape: [Int], ni: Int, no: Int) -> DTensor {
        let count = shape.reduce(1) { $0 * $1 }
        let upper: Double = (6.0.squareRoot() / Double(ni + no).squareRoot())
        let lower: Double = -upper
        var uniform = [Double]()
        uniform.reserveCapacity(count)
        for _ in 0..<count { uniform.append(Double.random(in: lower..<upper)) }
        let t = DTensor(shape: shape, grid: uniform)
        return t
    }
}
// MARK: - Matrix Specific
extension DTensor {
    public func matInv() -> DTensor {
        precondition(shape.count == 2, "Must be a matrix")
        precondition(shape[0] == shape[1], "Must be a square matrix")
        var t = self
        t.grid.withUnsafeMutableBufferPointer { ptr in
            var ipiv = [__CLPK_integer](repeating: 0, count: shape[0] * shape[0])
            var lwork = __CLPK_integer(shape[1] * shape[1])
            var work = [CDouble](repeating: 0, count: Int(lwork))
            var error: __CLPK_integer = 0
            var nc = __CLPK_integer(shape[1])
            var m = nc
            var n = nc
            
            dgetrf_(&m, &n, ptr.baseAddress!, &nc, &ipiv, &error)
            dgetri_(&m, ptr.baseAddress!, &nc, &ipiv, &work, &lwork, &error)
            assert(error == 0, "Matrix not invertible")
        }
        return t
    }
    
    public func transpose() -> DTensor {
        // Scalar
        if shape.count == 0 {
            return self
        }
        // Row vector
        if shape.count == 1 {
            let t = DTensor(shape: [shape[0], 1], grid: grid)
            return t
        } else if shape.count == 2 && shape[0] == 1 {
            let t = DTensor(shape: [shape[1], 1], grid: grid)
            return t
        } else if shape.count == 2 && shape[1] == 1 {
            // Column vector
            let t = DTensor(shape: [shape[1], shape[0]], grid: grid)
            return t
        }
        precondition(shape.count == 2, "Must be a matrix or vector")
        var t = DTensor(shape: shape.reversed(), repeating: 0)
        grid.withUnsafeBufferPointer { gridPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                vDSP_mtransD(gridPtr.baseAddress!, 1, tPtr.baseAddress!, 1, vDSP_Length(t.shape[0]), vDSP_Length(t.shape[1]))
            }
        }
        return t
    }
    public func diag() -> DTensor {
        if shape.count == 1 || (shape.count == 2 && shape[0] == 1) {
            // Row vector
            var t = DTensor(shape: shape, repeating: 0)
            t.grid[0] = grid[0]
            return t
        } else if shape.count == 2 && shape[1] == 1 {
            // Column vector
            var t = DTensor(shape: shape, repeating: 0)
            t.grid[0] = grid[0]
            return t
        }
        precondition(shape.count == 2, "Must be a matrix or vector")
        var t = DTensor(shape: shape, repeating: 0)
        grid.withUnsafeBufferPointer { gridPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                cblas_dcopy(Int32(min(shape[0], shape[1])), gridPtr.baseAddress!, Int32(shape[1] + 1), tPtr.baseAddress!, Int32(shape[1] + 1))
            }
        }
        return t
    }
}
// MARK: Image Specific
extension DTensor {
    public func conv2D(with kernel: DTensor, type: TensorConvType) -> DTensor {
        precondition(shape.count == 2 && shape[1] != 1 && kernel.shape.count == 2 && kernel.shape[1] != 1, "Image and kernel must be matrices")
        switch type {
        case .valid:
            return conv2D_valid(with: kernel)
        case .same:
            return conv2D_same(with: kernel)
        case .full:
            return conv2D_full(with: kernel)
        }
    }
    public func conv2D(with kernel: DTensor) -> DTensor {
        precondition(shape.count == 2 && shape[1] != 1 && kernel.shape.count == 2 && kernel.shape[1] != 1, "Image and kernel must be matrices")
        // Padding by vDSP
        let vertPad = (kernel.shape[0] - 1) / 2
        let horzPad = (kernel.shape[1] - 1) / 2
        // Dimension of output sans padding
        let tempShape = [shape[0] - kernel.shape[0] + 1, shape[1] - kernel.shape[1] + 1]
        // Output shape
        let resShape = [tempShape[0] + vertPad * 2, tempShape[1] + horzPad * 2]
        // Output tensor
        var res = DTensor(shape: resShape, repeating: -1)
        // Convolve
        grid.withUnsafeBufferPointer { imagePtr in
            kernel.grid.withUnsafeBufferPointer { kernelPtr in
                res.grid.withUnsafeMutableBufferPointer { resPtr in
                    vDSP_imgfirD(imagePtr.baseAddress!, vDSP_Length(shape[0]), vDSP_Length(shape[1]), kernelPtr.baseAddress!, resPtr.baseAddress!, vDSP_Length(kernel.shape[0]), vDSP_Length(kernel.shape[1]))
                }
            }
        }
        return res
    }
    public func conv2D_mine(with kernel: DTensor) -> DTensor {
        var res = DTensor(shape: [shape[0] - kernel.shape[0] + 1, shape[1] - kernel.shape[1] + 1], repeating: 0.0)
        var idx = 0
        for r in 0..<shape[0] {
            if r < shape[0] && r + kernel.shape[0] <= shape[0] {
                let pools = self[rows: r..<r + kernel.shape[0]]
                for c in 0..<shape[1] {
                    if c < shape[1] && c + kernel.shape[1] <= shape[1] {
                        let pool = pools[cols: c..<c + kernel.shape[1]]
                        // Max of pool
                        res.grid[idx] = (pool * kernel).sum()
                        idx += 1
                    }
                }
            }
        }
        return res
    }
    public func conv2D_valid(with kernel: DTensor) -> DTensor {
        // vDSP_imgfir even kernels wonky behavior....
        if kernel.shape[0] % 2 == 0 || kernel.shape[1] % 2 == 0 {
            return self.pad(0, 1, 0, 1).conv2D(with: kernel.pad(0, 1, 0, 1)).trim((kernel.shape[0]) / 2, (kernel.shape[1]) / 2)
        }
        return self.conv2D(with: kernel).trim((kernel.shape[0] - 1) / 2, (kernel.shape[1] - 1) / 2)
    }
    public func conv2D_same(with kernel: DTensor) -> DTensor {
        return self.pad((kernel.shape[0] - 1) / 2, (kernel.shape[1] - 1) / 2).conv2D_valid(with: kernel)
    }
    public func conv2D_full(with kernel: DTensor) -> DTensor {
        return self.pad(kernel.shape[0] - 1, kernel.shape[1] - 1).conv2D_valid(with: kernel)
    }
    // Could be optimized? 10X slower than vDSP_imgfir (conv)
    public func pool2D_max(size: Int) -> DTensor {
        return pool2D_max(size: size).0
    }
    // Could be optimized? 10X slower than vDSP_imgfir (conv)
    public func pool2D_max(size: Int, strd: Int = 1) -> (DTensor, [Int]) {
        precondition(shape.count == 2 && shape[1] != 1, "Image and kernel must be matrices")
        var res = DTensor(shape: [((shape[0] - size) / strd) + 1, ((shape[1] - size) / strd) + 1], repeating: 0)
        var positions = Array(repeating: 0, count: res.grid.count)
        var idx = 0
        for r in stride(from: 0, through: shape[0] - size, by: strd) {
            for c in stride(from: 0, through: shape[1] - size, by: strd) {
                var pool = Array(repeating: (val: 0.0, idx: -1), count: size * size)
                for rp in r..<r + size {
                    for cp in c..<c + size {
                        let flat_idx = rp * shape[1] + cp
                        pool[(rp * size + cp) - (r * size + c)] = (grid[flat_idx], flat_idx)
                    }
                }
                let max = pool.max(by: { $0.val < $1.val })!
                res.grid[idx] = max.val
                positions[idx] = max.idx
                idx += 1
            }
        }
//        DispatchQueue.concurrentPerform(iterations: shape[0] - size + 1) { r in
//            guard r % strd == 0 else { return }
//            DispatchQueue.concurrentPerform(iterations: shape[1] - size + 1) { c in
//                guard c % strd == 0 else { return }
//                var pool = Array(repeating: (val: 0.0, idx: -1), count: size * size)
//                for rp in r..<r + size {
//                    for cp in c..<c + size {
//                        let flat_idx = rp * shape[1] + cp
//                        pool[(rp * size + cp) - (r * size + c)] = (grid[flat_idx], flat_idx)
//                    }
//                }
//                let max = pool.max(by: { $0.val < $1.val })!
//                let idx = ((r * shape[1] + c) - r * (shape[1] - res.shape[1])) / strd
//                res.grid[idx] = max.val
//                positions[idx] = max.idx
//            }
//        }
        return (res, positions)
    }
    public func pad(_ w: Int, _ h: Int) -> DTensor {
        return pad(w, w, h, h)
    }
    public func pad(_ le: Int, _ ri: Int, _ to: Int, _ bo: Int) -> DTensor {
        precondition(shape.count == 2 && shape[1] != 1, "Must be a matrix")
        var out = DTensor(shape: [shape[0] + le + ri, shape[1] + to + bo], repeating: 0)
        var idx = 0
        for i in 0..<out.grid.count {
            let r = i / out.shape[1]
            let c = i % out.shape[1]
            // Only store non-padding numbers
            if r <= -1 + le || r >= out.shape[0] - ri || c <= -1 + to || c >= out.shape[1] - bo { continue }
            out.grid[c + r * out.shape[1]] = grid[idx]
            idx += 1
        }
        return out
    }
    public func trim(_ w: Int, _ h: Int) -> DTensor {
        return trim(w, w, h, h)
    }
    public func trim(_ le: Int, _ ri: Int, _ to: Int, _ bo: Int) -> DTensor {
        precondition(shape.count == 2 && shape[1] != 1, "Must be a matrix")
        var out = DTensor(shape: [shape[0] - le - ri, shape[1] - to - bo], repeating: 0)
        var idx = 0
        for i in 0..<grid.count {
            let r = i / shape[1]
            let c = i % shape[1]
            // Only store non-padding numbers
            if r <= -1 + le || r >= shape[0] - ri || c <= -1 + to || c >= shape[1] - bo { continue }
            out.grid[idx] = grid[c + r * shape[1]]
            idx += 1
        }
        return out
    }
    public func rot180() -> DTensor {
        precondition(shape.count == 2 && shape[1] != 1, "Must be a matrix")
        return DTensor(shape: shape, grid: grid.reversed())
    }
}
// MARK: - Addition Arithmetic
// SUPPORTS UP TO 3D-Tensor OPERATIONS (same shape up to ND-Tensor)
public func + (lhs: DTensor, rhs: DTensor) -> DTensor {
    // Remove unnecessary dimensions for operation then add them back if necessary
    var res = trueAdd(DTensor(shape: Array(lhs.shape.main.view), grid: lhs.grid), DTensor(shape: Array(rhs.shape.main.view), grid: rhs.grid))
    while res.shape.count < lhs.shape.count || res.shape.count < rhs.shape.count {
        res.shape.insert(1, at: 0)
    }
    return res
}
public func + (lhs: DTensor, rhs: Double) -> DTensor {
    return lhs + DTensor(rhs)
}
public func + (lhs: Double, rhs: DTensor) -> DTensor {
    return DTensor(lhs) + rhs
}
// MARK: Add Arithmetic True
// Assumes extra shape has been rid
fileprivate func trueAdd(_ lhs: DTensor, _ rhs: DTensor) -> DTensor {
    if lhs.shape == rhs.shape {
        // Double + Double or Vector + Vector of same size or Matrix + Matrix of same size or ND-Tensor + ND-Tensor of same size
        var t = rhs
        rhs.grid.withUnsafeBufferPointer { rhsPtr in
            lhs.grid.withUnsafeBufferPointer { lhsPtr in
                t.grid.withUnsafeMutableBufferPointer { tPtr in
                    vDSP_vaddD(rhsPtr.baseAddress!, vDSP_Stride(1), lhsPtr.baseAddress!, vDSP_Stride(1), tPtr.baseAddress!, vDSP_Stride(1), vDSP_Length(rhs.count))
                }
            }
        }
        return t
    } else if lhs.shape.count == 0 && rhs.shape.count >= 1 {
        // Double + Vector or Double + Matrix or Double + ND-Tensor
        var t = rhs
        rhs.grid.withUnsafeBufferPointer { rhsPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                var d = lhs.grid.first!
                vDSP_vsaddD(rhsPtr.baseAddress!, 1, &d, tPtr.baseAddress!, 1, vDSP_Length(rhs.count))
            }
        }
        return t
    } else if lhs.shape.count >= 1 && rhs.shape.count == 0 {
        // Vector + Double or Matrix + Double or ND-Tensor + Double
        return trueAdd(rhs, lhs)
    } else if lhs.shape.count == 1 && rhs.shape.count == 2 {
        // Row vector (1 by N) + Matrix
        if lhs.shape[0] == rhs.shape[1] {
            var t = rhs
            lhs.grid.withUnsafeBufferPointer { lhsPtr in
                rhs.grid.withUnsafeBufferPointer { rhsPtr in
                    t.grid.withUnsafeMutableBufferPointer { tPtr in
                        for c in 0..<rhs.shape[1] {
                            var val = lhsPtr[c]
                            vDSP_vsaddD(rhsPtr.baseAddress! + c, rhs.shape[1], &val, tPtr.baseAddress! + c, rhs.shape[1], vDSP_Length(rhs.shape[0]))
                        }
                    }
                }
            }
            return t
        }
    } else if lhs.shape.count == 2 && rhs.shape.count == 1 {
        // Matrix + Row vector (1 by N)
        return trueAdd(rhs, lhs)
    } else if lhs.shape.count == 2 && rhs.shape.count == 2  && lhs.shape[1] == 1 {
        // Column vector (N by 1) + Matrix
        if lhs.shape[0] == rhs.shape[0] {
            var t = rhs
            lhs.grid.withUnsafeBufferPointer { lhsPtr in
                rhs.grid.withUnsafeBufferPointer { rhsPtr in
                    t.grid.withUnsafeMutableBufferPointer { tPtr in
                        for r in 0..<rhs.shape[0] {
                            var val = lhsPtr[r]
                            vDSP_vsaddD(rhsPtr.baseAddress! + r * rhs.shape[1], 1, &val, tPtr.baseAddress! + r * rhs.shape[1], 1, vDSP_Length(rhs.shape[1]))
                        }
                    }
                }
            }
            return t
        }
    } else if rhs.shape.count == 2 && lhs.shape.count == 2  && rhs.shape[1] == 1 {
        // Matrix + Column vector (N by 1)
        return trueAdd(rhs, lhs)
    } else if lhs.shape.count == 1 && rhs.shape.count == 3 {
        // Row vector + 3D-Tensor
        if lhs.shape[0] == rhs.shape[2] {
            var t = rhs
            for m in 0..<rhs.shape[0] {
                t[mat: m] = trueAdd(lhs, rhs[mat: m])
            }
            return t
        }
    } else if lhs.shape.count == 3 && rhs.shape.count == 1 {
        // 3D-Tensor + Row vector
        return trueAdd(rhs, lhs)
    } else if lhs.shape.count == 2 && rhs.shape.count == 3 {
        if lhs.shape[0] == rhs.shape[1] && lhs.shape[1] == rhs.shape[2] {
            // Matrix + 3D-Tensor
            var t = rhs
            for m in 0..<rhs.shape[0] {
                t[mat: m] = trueAdd(lhs, rhs[mat: m])
            }
            return t
        } else if lhs.shape[0] == rhs.shape[1] && lhs.shape[1] == 1 {
            // Column vector + 3D-Tensor
            var t = rhs
            for m in 0..<rhs.shape[0] {
                t[mat: m] = trueAdd(lhs, rhs[mat: m])
            }
            return t
        }
    } else if lhs.shape.count == 3 && rhs.shape.count == 2 {
        // 3D-Tensor + Matrix
        return trueAdd(rhs, lhs)
    } else if lhs.shape.count == 3 && rhs.shape.count == 4 {
        if lhs.shape[0] == rhs.shape[1] && lhs.shape[1] == rhs.shape[2] && lhs.shape[2] == rhs.shape[3] {
            // 3D-Tensor + 4D-Tensor
            var t = rhs
            for n in 0..<rhs.shape[0] {
                t[t3D: n] = trueAdd(lhs, rhs[t3D: n])
            }
            return t
        }
    } else if lhs.shape.count == 4 && rhs.shape.count == 3 {
        return trueAdd(rhs, lhs)
    }
    fatalError("Cannot elementwise add (or subtract) lhs shape of \(lhs.shape) with rhs shape of \(rhs.shape)")
}
// MARK: - Subtraction Arithmetic
// SUPPORTS UP TO 3D-Tensor OPERATIONS (same shape up to ND-Tensor)
prefix public func -(t: DTensor) -> DTensor {
    var new = t
    new.grid.withUnsafeMutableBufferPointer { newPtr in
        t.grid.withUnsafeBufferPointer { tPtr in
            vDSP_vnegD(tPtr.baseAddress!, 1, newPtr.baseAddress!, 1, vDSP_Length(t.count))
        }
    }
    return new
}
public func - (lhs: DTensor, rhs: DTensor) -> DTensor {
    return lhs + -rhs
}
public func - (lhs: DTensor, rhs: Double) -> DTensor {
    return lhs - DTensor(rhs)
}
public func - (lhs: Double, rhs: DTensor) -> DTensor {
    return DTensor(lhs) - rhs
}
// MARK: - Multiplication Arithmetic
// SUPPORTS UP TO 3D-Tensor OPERATIONS (same shape up to ND-Tensor)
public func * (lhs: DTensor, rhs: DTensor) -> DTensor {
    // Remove unnecessary dimensions for operation then add them back if necessary
    var res = trueMult(DTensor(shape: Array(lhs.shape.main.view), grid: lhs.grid), DTensor(shape: Array(rhs.shape.main.view), grid: rhs.grid))
    while res.shape.count < lhs.shape.count || res.shape.count < rhs.shape.count {
        res.shape.insert(1, at: 0)
    }
    return res
}
public func * (lhs: DTensor, rhs: Double) -> DTensor {
    return lhs * DTensor(rhs)
}
public func * (lhs: Double, rhs: DTensor) -> DTensor {
    return DTensor(lhs) * rhs
}
// MARK: Multiplication Arithmetic True
// Assumes extra shape has been rid
fileprivate func trueMult(_ lhs: DTensor, _ rhs: DTensor) -> DTensor {
    if lhs.shape == rhs.shape {
        // Double * Double or Vector * Vector of same size or Matrix * Matrix of same size or ND-Tensor * ND-Tensor of same size
        var t = rhs
        rhs.grid.withUnsafeBufferPointer { rhsPtr in
            lhs.grid.withUnsafeBufferPointer { lhsPtr in
                t.grid.withUnsafeMutableBufferPointer { tPtr in
                    vDSP_vmulD(rhsPtr.baseAddress!, vDSP_Stride(1), lhsPtr.baseAddress!, vDSP_Stride(1), tPtr.baseAddress!, vDSP_Stride(1), vDSP_Length(rhs.count))
                }
            }
        }
        return t
    } else if lhs.shape.count == 0 && rhs.shape.count >= 1 {
        // Double * Vector or Double * Matrix or Double * ND-Tensor
        var t = rhs
        rhs.grid.withUnsafeBufferPointer { rhsPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                var d = lhs.grid.first!
                vDSP_vsmulD(rhsPtr.baseAddress!, 1, &d, tPtr.baseAddress!, 1, vDSP_Length(rhs.count))
            }
        }
        return t
    } else if lhs.shape.count >= 1 && rhs.shape.count == 0 {
        // Vector * Double or Matrix * Double or ND-Tensor * Double
        return trueMult(rhs, lhs)
    } else if lhs.shape.count == 1 && rhs.shape.count == 2 {
        // Row vector (1 by N) * Matrix
        if lhs.shape[0] == rhs.shape[1] {
            var t = rhs
            lhs.grid.withUnsafeBufferPointer { lhsPtr in
                rhs.grid.withUnsafeBufferPointer { rhsPtr in
                    t.grid.withUnsafeMutableBufferPointer { tPtr in
                        for c in 0..<rhs.shape[1] {
                            var val = lhsPtr[c]
                            vDSP_vsmulD(rhsPtr.baseAddress! + c, rhs.shape[1], &val, tPtr.baseAddress! + c, rhs.shape[1], vDSP_Length(rhs.shape[0]))
                        }
                    }
                }
            }
            return t
        }
    } else if lhs.shape.count == 2 && rhs.shape.count == 1 {
        // Matrix * Row vector (1 by N)
        return trueMult(rhs, lhs)
    } else if lhs.shape.count == 2 && rhs.shape.count == 2  && lhs.shape[1] == 1 {
        // Column vector (N by 1) * Matrix
        if lhs.shape[0] == rhs.shape[0] {
            var t = rhs
            lhs.grid.withUnsafeBufferPointer { lhsPtr in
                rhs.grid.withUnsafeBufferPointer { rhsPtr in
                    t.grid.withUnsafeMutableBufferPointer { tPtr in
                        for r in 0..<rhs.shape[0] {
                            var val = lhsPtr[r]
                            vDSP_vsmulD(rhsPtr.baseAddress! + r * rhs.shape[1], 1, &val, tPtr.baseAddress! + r * rhs.shape[1], 1, vDSP_Length(rhs.shape[1]))
                        }
                    }
                }
            }
            return t
        }
    } else if rhs.shape.count == 2 && lhs.shape.count == 2  && rhs.shape[1] == 1 {
        // Matrix * Column vector (N by 1)
        return trueMult(rhs, lhs)
    } else if lhs.shape.count == 1 && rhs.shape.count == 3 {
        // Row vector * 3D-Tensor
        if lhs.shape[0] == rhs.shape[2] {
            var t = rhs
            for m in 0..<rhs.shape[0] {
                t[mat: m] = trueMult(lhs, rhs[mat: m])
            }
            return t
        }
    } else if lhs.shape.count == 3 && rhs.shape.count == 1 {
        // 3D-Tensor * Row vector
        return trueMult(rhs, lhs)
    } else if lhs.shape.count == 2 && rhs.shape.count == 3 {
        if lhs.shape[0] == rhs.shape[1] && lhs.shape[1] == rhs.shape[2] {
            // Matrix * 3D-Tensor
            var t = rhs
            for m in 0..<rhs.shape[0] {
                t[mat: m] = trueMult(lhs, rhs[mat: m])
            }
            return t
        } else if lhs.shape[0] == rhs.shape[1] && lhs.shape[1] == 1 {
            // Column vector * 3D-Tensor
            var t = rhs
            for m in 0..<rhs.shape[0] {
                t[mat: m] = trueMult(lhs, rhs[mat: m])
            }
            return t
        }
    } else if lhs.shape.count == 3 && rhs.shape.count == 2 {
        // 3D-Tensor * Matrix
        return trueMult(rhs, lhs)
    } else if lhs.shape.count == 3 && rhs.shape.count == 4 {
        if lhs.shape[0] == rhs.shape[1] && lhs.shape[1] == rhs.shape[2] && lhs.shape[2] == rhs.shape[3] {
            // 3D-Tensor * 4D-Tensor
            var t = rhs
            for n in 0..<rhs.shape[0] {
                t[t3D: n] = trueMult(lhs, rhs[t3D: n])
            }
            return t
        }
    } else if lhs.shape.count == 4 && rhs.shape.count == 3 {
        return trueMult(rhs, lhs)
    }
    fatalError("Cannot elementwise multiply (or divide) lhs shape of \(lhs.shape) with rhs shape of \(rhs.shape)")
}
// MARK: - Division Arithmetic
// SUPPORTS UP TO 3D-Tensor OPERATIONS (same shape up to ND-Tensor)
extension DTensor {
    public func inv() -> DTensor {
        var t = self
        grid.withUnsafeBufferPointer { gridPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                var one: Double = 1
                vDSP_svdivD(&one, gridPtr.baseAddress!, vDSP_Stride(1), tPtr.baseAddress!, vDSP_Stride(1), vDSP_Length(count))
            }
        }
        return t
    }
}
public func / (lhs: DTensor, rhs: DTensor) -> DTensor {
    return lhs * rhs.inv()
}
public func / (lhs: DTensor, rhs: Double) -> DTensor {
    return lhs / DTensor(rhs)
}
public func / (lhs: Double, rhs: DTensor) -> DTensor {
    return DTensor(lhs) / rhs
}
// MARK: - Matrix Multiplication Arithmetic
// SUPPORTS Vector by Vector, Matrix by Vector, Matrix by Matrix, Scalar by Matrix
// REVISIT NOT MADE FOR EXTRA SHAPE YET
infix operator <*> : MultiplicationPrecedence
public func <*> (lhs: DTensor, rhs: DTensor) -> DTensor {
    // Remove unnecessary dimensions for operation then add them back if necessary
    var res = trueMatMult(DTensor(shape: Array(lhs.shape.main.view), grid: lhs.grid), DTensor(shape: Array(rhs.shape.main.view), grid: rhs.grid))
    while res.shape.count < lhs.shape.count || res.shape.count < rhs.shape.count {
        res.shape.insert(1, at: 0)
    }
    return res
}
fileprivate func trueMatMult(_ lhs: DTensor, _ rhs: DTensor) -> DTensor {
    // Multiply scalars (just normal multiplication)
    if lhs.shape.count == 0 || rhs.shape.count == 0 {
        return lhs * rhs
    }
    // Convert row vectors to 2D
    var lhs_shape = lhs.shape
    var rhs_shape = rhs.shape
    if lhs.shape.count == 1 {
        lhs_shape.insert(1, at: 0)
    }
    if rhs.shape.count == 1 {
        rhs_shape.insert(1, at: 0)
    }
    // Continue to matrix multiplication!
    precondition(lhs_shape.count == 2 && rhs_shape.count == 2, "Tensor lhs \(lhs.shape) and Tensor rhs \(rhs.shape) not compatible for matrix multiplication")
    precondition(lhs_shape[1] == rhs_shape[0], "Matrix lhs \(lhs.shape) and Matrix rhs \(rhs.shape) not compatible for matrix multiplication")
    var t: DTensor
    t = DTensor(shape: [lhs_shape[0], rhs_shape[1]], repeating: 0)
    lhs.grid.withUnsafeBufferPointer { lhsPtr in
        rhs.grid.withUnsafeBufferPointer { rhsPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(lhs_shape[0]), Int32(rhs_shape[1]), Int32(lhs_shape[1]), 1, lhsPtr.baseAddress!, Int32(lhs_shape[1]), rhsPtr.baseAddress!, Int32(rhs_shape[1]), 0, tPtr.baseAddress!, Int32(t.shape[1]))
            }
        }
    }
    return t
}
// MARK: - Unary Arithmetic
extension DTensor {
    public func exp() -> DTensor {
        var t = self
        grid.withUnsafeBufferPointer { gridPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                var N = Int32(count)
                vvexp(tPtr.baseAddress!, gridPtr.baseAddress!, &N)
            }
        }
        return t
    }
    public func log() -> DTensor {
        var t = self
        grid.withUnsafeBufferPointer { gridPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                var N = Int32(count)
                vvlog(tPtr.baseAddress!, gridPtr.baseAddress!, &N)
            }
        }
        return t
    }
    public func sin() -> DTensor {
        var t = self
        grid.withUnsafeBufferPointer { gridPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                var N = Int32(count)
                vvsin(tPtr.baseAddress!, gridPtr.baseAddress!, &N)
            }
        }
        return t
    }
    public func cos() -> DTensor {
        var t = self
        grid.withUnsafeBufferPointer { gridPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                var N = Int32(count)
                vvcos(tPtr.baseAddress!, gridPtr.baseAddress!, &N)
            }
        }
        return t
    }
    public func pow(_ a: Double) -> DTensor {
        var t = self
        grid.withUnsafeBufferPointer { gridPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                if a == 2 {
                    vDSP_vsqD(gridPtr.baseAddress!, 1, tPtr.baseAddress!, 1, vDSP_Length(count))
                } else {
                    var N = Int32(count)
                    var exp = a
                    vvpows(tPtr.baseAddress!, &exp, gridPtr.baseAddress!, &N)
                }
            }
        }
        return t
    }
    public func sqrt() -> DTensor {
        var t = self
        grid.withUnsafeBufferPointer { gridPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                var N = Int32(count)
                vvsqrt(tPtr.baseAddress!, gridPtr.baseAddress!, &N)
            }
        }
        return t
    }
    public func max() -> Double {
        var result = 0.0
        grid.withUnsafeBufferPointer { gridPtr in
            vDSP_maxvD(gridPtr.baseAddress!, 1, &result, vDSP_Length(count))
        }
        return result
    }
    public func max() -> (Double, Int) {
        var result = 0.0
        var idx: UInt = 0
        grid.withUnsafeBufferPointer { gridPtr in
            vDSP_maxviD(gridPtr.baseAddress!, 1, &result, &idx, vDSP_Length(count))
        }
        return (result, Int(idx))
    }
    public func sum() -> Double {
        var result = 0.0
        grid.withUnsafeBufferPointer { gridPtr in
            vDSP_sveD(gridPtr.baseAddress!, 1, &result, vDSP_Length(count))
        }
        return result
    }
    public func sumDiag() -> Double {
        precondition(shape.count == 2, "Must be a matrix")
        var result = 0.0
        grid.withUnsafeBufferPointer { gridPtr in
            vDSP_sveD(gridPtr.baseAddress!, shape[1] + 1, &result, vDSP_Length(Swift.min(shape[0], shape[1])))
        }
        return result
    }
    public func sum(axis: Int, keepDim: Bool = false) -> DTensor {
        var newShape = shape
        // Check if keeping extra dimension
        if keepDim { newShape[axis] = 1 } else { newShape.remove(at: axis) }
        // Check for summing 1 size axis
        if shape[axis] == 1 { return DTensor(shape: newShape, grid: grid) }
        precondition(axis < shape.count, "Axis not present in this tensor")
        // Different approach for last axis
        if axis == shape.count - 1 {
            // Make new tensor
            var t = DTensor(shape: newShape, repeating: 0.0)
            // size of vectors being summed (also stride)
            let N = shape[axis]
            // Length of tensor
            let count = grid.count
            grid.withUnsafeBufferPointer { gridPtr in
                t.grid.withUnsafeMutableBufferPointer { tPtr in
                    // Incrementer for position in t
                    var inc = 0
                    // Stride through for summation
                    for i in stride(from: 0, to: count, by: N) {
                        var result: Double = 0.0
                        // Sum this N length vector
                        vDSP_sveD(gridPtr.baseAddress! + i, 1, &result, vDSP_Length(N))
                        // Set to our position in t
                        tPtr[inc] = result
                        inc += 1
                    }
                }
            }
            return t
        } else {
            // Make new tensor
            var t = DTensor(shape: newShape, repeating: 0.0)
            // size of vectors being summed
            let N = shape[axis + 1..<shape.count].reduce(1) { $0 * $1 }
            // Length of tensor
            let count = grid.count
            grid.withUnsafeBufferPointer { gridPtr in
                t.grid.withUnsafeMutableBufferPointer { tPtr in
                    // Incrementer for position in t
                    var inc = 0
                    // Stride through for summation
                    for i in stride(from: 0, to: count, by: N) {
                        vDSP_vaddD(gridPtr.baseAddress! + i, 1, tPtr.baseAddress! + inc / shape[axis] * N, 1, tPtr.baseAddress! + inc / shape[axis] * N, 1, vDSP_Length(N))
                        inc += 1
                    }
                }
            }
            return t
        }
    }
    public func sum(axes: ClosedRange<Int>, keepDim: Bool = false) -> DTensor {
        return self.sum(axes: Range(axes), keepDim: keepDim)
    }
    public func sum(axes: Range<Int>, keepDim: Bool = false) -> DTensor {
        precondition(axes.upperBound - axes.lowerBound == axes.count && axes.upperBound <= shape.count && axes.lowerBound >= 0, "Incompatible range for shape")
        // check for all encompassing range, and singular range
        if axes.lowerBound == 0 && axes.upperBound == shape.count {
            // sum everything for all encompassing
            let val = self.sum()
            let newShape = keepDim ? Array(repeating: 1, count: shape.count) : []
            return DTensor(shape: newShape, grid: [val])
        } else if axes.count == 1 {
            // basically just one axis
            return self.sum(axis: axes.lowerBound, keepDim: keepDim)
        }
        // make new shpae
        var newShape = shape
        // type of summation
        if axes.lowerBound == 0 {
            // Check if keeping extra dimension
            if keepDim {
                for axis in axes {
                    newShape[axis] = 1
                }
            } else {
                while newShape.count > (shape.count - axes.count) {
                    newShape.remove(at: 0)
                }
            }
            // leading, make empty tensor
            var t = DTensor(shape: newShape, repeating: 0.0)
            // size of vectors being summed
            let N = shape[axes.upperBound..<shape.count].reduce(1) { $0 * $1 }
            // Length of tensor
            let count = grid.count
            grid.withUnsafeBufferPointer { gridPtr in
                t.grid.withUnsafeMutableBufferPointer { tPtr in
                    // Incrementer for position in t
                    var inc = 0
                    // Stride through for summation
                    for i in stride(from: 0, to: count, by: N) {
                        vDSP_vaddD(gridPtr.baseAddress! + i, 1, tPtr.baseAddress! + inc / shape[axes].reduce(1) { $0 * $1 } * N, 1, tPtr.baseAddress! + inc / shape[axes].reduce(1) { $0 * $1 } * N, 1, vDSP_Length(N))
                        inc += 1
                    }
                }
            }
            return t
        } else if axes.upperBound == shape.count {
            // Check if keeping extra dimension
            if keepDim {
                for axis in axes {
                    newShape[axis] = 1
                }
            } else {
                while newShape.count > (shape.count - axes.count) {
                    newShape.remove(at: newShape.count - 1)
                }
            }
            // trailing, make empty tensor
            var t = DTensor(shape: newShape, repeating: 0.0)
            // size of vectors being summed (also stride)
            let N = shape[axes].reduce(1) { $0 * $1 }
            // Length of tensor
            let count = grid.count
            grid.withUnsafeBufferPointer { gridPtr in
                t.grid.withUnsafeMutableBufferPointer { tPtr in
                    // Incrementer for position in t
                    var inc = 0
                    // Stride through for summation
                    for i in stride(from: 0, to: count, by: N) {
                        var result: Double = 0.0
                        // Sum this N length vector
                        vDSP_sveD(gridPtr.baseAddress! + i, 1, &result, vDSP_Length(N))
                        // Set to our position in t
                        tPtr[inc] = result
                        inc += 1
                    }
                }
            }
            return t
        }
        fatalError("Not implemented")
    }
    // Calculates mean and std for each column!
    public func zscore() -> (norm: DTensor, mean: DTensor, std: DTensor) {
        precondition(shape.count == 2, "Incompatible type for zscore normalization, must be matrix or column vector")
        var t = self
        var mean = DTensor(shape: [1, shape[1]], repeating: 0)
        var std = DTensor(shape: [1, shape[1]], repeating: 0)
        mean.grid.withUnsafeMutableBufferPointer { meanPtr in
            std.grid.withUnsafeMutableBufferPointer { stdPtr in
                for c in 0..<shape[1] {
                    t.grid.withUnsafeMutableBufferPointer { tPtr in
                        grid.withUnsafeBufferPointer { gridPtr in
                            vDSP_normalizeD(gridPtr.baseAddress! + c, shape[1], tPtr.baseAddress! + c, shape[1], meanPtr.baseAddress! + c, stdPtr.baseAddress! + c, vDSP_Length(shape[0]))
                        }
                    }
                }
            }
        }
        return (t, mean, std)
    }
    // Calculates mean and std for each pixel!
    public func zscore_image_norm(mean: DTensor, std: DTensor) -> DTensor {
        if shape.count == 4 && mean.shape.count == 1 && mean.shape[0] == shape[1] && std.shape.count == 1 && std.shape[0] == shape[1] {
            var t = self
            for n in 0..<shape[0] {
                let nthImage = self[t3D: n]
                for d in 0..<shape[1] {
                    t[t3D: n][mat: d] = (nthImage[mat: d] - mean[val: d]) / std[val: d]
                }
            }
            return t
        } else if shape.count == 3 && mean.shape.count == 1 && mean.shape[0] == shape[0] && std.shape.count == 1 && std.shape[0] == shape[0] {
            var t = self
            for d in 0..<shape[0] {
                t[mat: d] = (self[mat: d] - mean[val: d]) / std[val: d]
            }
            return t
        }
        fatalError("Incompatible type for zscore image norm")
    }
    public func zscore_image() -> (norm: DTensor, mean: DTensor, std: DTensor) {
        precondition(shape.count == 4, "Incompatible type for zscore image normalization, must be a 4D-Tensor")
        var channel_avgs = self.sum(axes: 2...3) / Double(shape[2] * shape[3])
        var mean = DTensor(shape: [shape[1]], repeating: 0)
        var std = DTensor(shape: [shape[1]], repeating: 0)
        mean.grid.withUnsafeMutableBufferPointer { meanPtr in
            std.grid.withUnsafeMutableBufferPointer { stdPtr in
                for c in 0..<shape[1] {
                    channel_avgs.grid.withUnsafeMutableBufferPointer { cPtr in
                        // Store channel mean
                        vDSP_meanvD(cPtr.baseAddress! + c, shape[1], meanPtr.baseAddress! + c, vDSP_Length(shape[0]))
                        // Find feature std
                        vDSP_measqvD(cPtr.baseAddress! + c, shape[1], stdPtr.baseAddress! + c, vDSP_Length(shape[0]))
                        stdPtr[c] = (stdPtr[c] - meanPtr[c] * meanPtr[c] + 0.0001).squareRoot()
                    }
                }
            }
        }
        return (self.zscore_image_norm(mean: mean, std: std), mean, std)
    }
}
// MARK: Activation Functions
extension DTensor {
    public func sigmoid() -> DTensor {
        return 1.0 / (1.0 + (-self).exp())
    }
    public func relu() -> DTensor {
        var grid = self.grid
        grid.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<ptr.count {
                if ptr[i] <= 0 {
                    ptr[i] = 0
                }
            }
        }
        return DTensor(shape: shape, grid: grid)
    }
    public func drelu() -> DTensor {
        var grid = self.grid
        grid.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<ptr.count {
                if ptr[i] <= 0 {
                    ptr[i] = 0
                } else {
                    ptr[i] = 1
                }
            }
        }
        return DTensor(shape: shape, grid: grid)
    }
    public func lrelu() -> DTensor {
        var grid = self.grid
        grid.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<ptr.count {
                if ptr[i] <= 0 {
                    ptr[i] = 0.2 * ptr[i]
                }
            }
        }
        return DTensor(shape: shape, grid: grid)
    }
    public func dlrelu() -> DTensor {
        var grid = self.grid
        grid.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<ptr.count {
                if ptr[i] <= 0 {
                    ptr[i] = 0.2
                } else {
                    ptr[i] = 1
                }
            }
        }
        return DTensor(shape: shape, grid: grid)
    }
}
// MARK: - Accessors
extension DTensor {
    public var count: Int {
        return shape.reduce()
    }
    public var type: TensorType {
        if shape.main.count == 0 {
            return .scalar
        } else if shape.main.count == 1 {
            return .row
        } else if shape.main.count == 2 {
            if shape.main[1] == 1 {
                return .column
            }
            return .matrix
        } else if shape.main.count == 3 {
            return .tensor3D
        } else if shape.main.count == 4 {
            return .tensor4D
        } else if shape.main.count >= 5 {
            return .tensorND
        }
        fatalError("Unable to descibe Tensor")
    }
}
