//
//  Tensor.swift
//  
//
//  Created by Sahil Srivastava on 12/4/21.
//
//  IMPORTANT NOTES: Tensor.shape.count == 1 DO NOT EXIST
//  Vectors are treated as matrices with shape = [1, N]
//  for row vectors and shape = [N, 1] for column vectors
//  AKA 1D-Tensors are just 2D-Tensors with one dimension
//  equal to 1. All other Tensor.shape.count >= 0 are valid
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

import Foundation
import Accelerate

public struct Tensor: Equatable {
    public let shape: [Int]
    public var grid: [Double]
    
    public init(shape: [Int], grid: [Double]) {
        self.shape = shape
        self.grid = grid
    }
}
// MARK: - Tensor Enums
extension Tensor {
    public enum TensorVectorType {
        case row
        case column
    }
    public enum TensorType: String {
        case scalar = "scalar"
        case vector = "vector"
        case matrix = "matrix"
        case tensor3D = "tensor3D"
        case tensorND = "tensorND"
    }
}
// MARK: - Inits
extension Tensor {
    public init(shape: [Int], repeating: Double) {
        self.shape = shape
        self.grid = Array(repeating: repeating, count: shape.reduce(1) { $0 * $1 })
    }
    public init(_ val: Double) {
        self.init(shape: [], repeating: val)
    }
    public init(_ vec: [Double], type: TensorVectorType) {
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
    public init(_ mats: [[[Double]]]) {
        self.init(shape: [mats.count, mats.first!.count, mats.first!.first!.count], repeating: 0.0)
        var j = 0
        lo: for mat in mats {
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
}
// MARK: - Query
extension Tensor {
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
    public subscript(mat m: Int) -> Tensor {
        get {
            precondition(shape.count == 3, "Must be a 3D-Tensor")
            var t = Tensor(shape: [shape[1], shape[2]], repeating: 0)
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
    public subscript(row r: Int) -> Tensor {
        get {
            precondition(shape.count == 2, "Must be a matrix")
            var t = Tensor(shape: [1, shape[1]], repeating: 0)
            grid.withUnsafeBufferPointer { gridPtr in
                t.grid.withUnsafeMutableBufferPointer { tPtr in
                    cblas_dcopy(Int32(shape[1]), gridPtr.baseAddress! + r * shape[1], 1, tPtr.baseAddress!, 1)
                }
            }
            return t
        }
        set(t) {
            precondition(shape.count == 2, "Must be a matrix")
            precondition(t.shape.count == 2 && t.shape[0] == 1 && t.shape[1] == shape[1], "Not compatible vector dimensions")
            grid.withUnsafeMutableBufferPointer { gridPtr in
                t.grid.withUnsafeBufferPointer { tPtr in
                    cblas_dcopy(Int32(shape[1]), tPtr.baseAddress!, 1, gridPtr.baseAddress! + r * shape[1], 1)
                }
            }
        }
    }
    public subscript(col c: Int) -> Tensor {
        get {
            precondition(shape.count == 2, "Must be a matrix")
            var t = Tensor(shape: [shape[0], 1], repeating: 0)
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
    public subscript(val v: Int) -> Double {
        get {
            precondition(shape.count == 2 && shape.contains(1), "Must be a vector")
            return grid[v]
        }
        set(t) {
            precondition(shape.count == 2 && shape.contains(1), "Must be a vector")
            grid[v] = t
        }
    }
}
// MARK: - Random Init
extension Tensor {
    public static func random(shape: [Int], min: Double = 0.0, max: Double = 1.0) -> Tensor {
        let count = shape.reduce(1) { $0 * $1 }
        var uniform = [Double]()
        uniform.reserveCapacity(count)
        for _ in 0..<count { uniform.append(Double.random(in: min..<max)) }
        let t = Tensor(shape: shape, grid: uniform)
        return t
    }
    public static func random_xavier(shape: [Int], ni: Int, no: Int) -> Tensor {
        let count = shape.reduce(1) { $0 * $1 }
        let upper: Double = (6.0.squareRoot() / Double(ni + no).squareRoot())
        let lower: Double = -upper
        var uniform = [Double]()
        uniform.reserveCapacity(count)
        for _ in 0..<count { uniform.append(lower + Double.random(in: 0..<1) * (upper - lower)) }
        let t = Tensor(shape: shape, grid: uniform)
        return t
    }
}
// MARK: - Matrix Specific
extension Tensor {
    public func matInv() -> Tensor {
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
    
    public func transpose() -> Tensor {
        precondition(shape.count == 2, "Must be a matrix")
        var t = Tensor(shape: shape.reversed(), repeating: 0)
        grid.withUnsafeBufferPointer { gridPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                vDSP_mtransD(gridPtr.baseAddress!, 1, tPtr.baseAddress!, 1, vDSP_Length(t.shape[0]), vDSP_Length(t.shape[1]))
            }
        }
        return t
    }
}
// MARK: - Addition Arithmetic
// SUPPORTS UP TO 3D-Tensor OPERATIONS (same shape up to ND-Tensor)
public func + (lhs: Tensor, rhs: Tensor) -> Tensor {
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
        return rhs + lhs
    } else if lhs.shape.count == 2 && rhs.shape.count == 2 {
        // Same shape matrices taken care of above, this should be 'Vector' + Matrix
        if lhs.shape[1] == 1 {
            // 'Vector' is a column vector (N by 1)
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
        } else if lhs.shape[0] == 1 {
            // 'Vector' is a row vector (1 by N)
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
        } else if rhs.shape[1] == 1 || rhs.shape[0] == 1 {
            return rhs + lhs
        }
    } else if lhs.shape.count == 2 && rhs.shape.count == 3 {
        // Matrix + 3D-Tensor
        if lhs.shape[0] == rhs.shape[1] && lhs.shape[1] == rhs.shape[2] {
            var t = rhs
            for m in 0..<rhs.shape[0] {
                t[mat: m] = lhs + rhs[mat: m]
            }
            return t
        } else if lhs.shape[1] == 1 {
            if lhs.shape[0] == rhs.shape[1] {
                // 'Vector' is a column vector (N by 1)
                var t = rhs
                for m in 0..<rhs.shape[0] {
                    t[mat: m] = lhs + rhs[mat: m]
                }
                return t
            }
        } else if lhs.shape[0] == 1 {
            if lhs.shape[1] == rhs.shape[2] {
                // 'Vector' is a row vector (1 by N)
                var t = rhs
                for m in 0..<rhs.shape[0] {
                    t[mat: m] = lhs + rhs[mat: m]
                }
                return t
            }
        }
    } else if lhs.shape.count == 3 && rhs.shape.count == 2 {
        // 3D-Tensor + Matrix
        return rhs + lhs
    }
    fatalError("Cannot elementwise add (or subtract) lhs shape of \(lhs.shape) with rhs shape of \(rhs.shape)")
}
public func + (lhs: Tensor, rhs: Double) -> Tensor {
    return lhs + Tensor(rhs)
}
public func + (lhs: Double, rhs: Tensor) -> Tensor {
    return Tensor(lhs) + rhs
}
// MARK: - Subtraction Arithmetic
// SUPPORTS UP TO 3D-Tensor OPERATIONS (same shape up to ND-Tensor)
prefix public func -(t: Tensor) -> Tensor {
    var new = t
    new.grid.withUnsafeMutableBufferPointer { newPtr in
        t.grid.withUnsafeBufferPointer { tPtr in
            vDSP_vnegD(tPtr.baseAddress!, 1, newPtr.baseAddress!, 1, vDSP_Length(t.count))
        }
    }
    return new
}
public func - (lhs: Tensor, rhs: Tensor) -> Tensor {
    return lhs + -rhs
}
public func - (lhs: Tensor, rhs: Double) -> Tensor {
    return lhs - Tensor(rhs)
}
public func - (lhs: Double, rhs: Tensor) -> Tensor {
    return Tensor(lhs) - rhs
}
// MARK: - Multiplication Arithmetic
// SUPPORTS UP TO 3D-Tensor OPERATIONS (same shape up to ND-Tensor)
public func * (lhs: Tensor, rhs: Tensor) -> Tensor {
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
        return rhs * lhs
    } else if lhs.shape.count == 2 && rhs.shape.count == 2 {
        // Same shape matrices taken care of above, this should be 'Vector' + Matrix
        if lhs.shape[1] == 1 {
            // 'Vector' is a column vector (N by 1)
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
        } else if lhs.shape[0] == 1 {
            // 'Vector' is a row vector (1 by N)
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
        } else if rhs.shape[1] == 1 || rhs.shape[0] == 1 {
            return rhs * lhs
        }
    } else if lhs.shape.count == 2 && rhs.shape.count == 3 {
        // Matrix * 3D-Tensor
        if lhs.shape[0] == rhs.shape[1] && lhs.shape[1] == rhs.shape[2] {
            var t = rhs
            for m in 0..<rhs.shape[0] {
                t[mat: m] = lhs * rhs[mat: m]
            }
            return t
        } else if lhs.shape[1] == 1 {
            if lhs.shape[0] == rhs.shape[1] {
                // 'Vector' is a column vector (N by 1)
                var t = rhs
                for m in 0..<rhs.shape[0] {
                    t[mat: m] = lhs * rhs[mat: m]
                }
                return t
            }
        } else if lhs.shape[0] == 1 {
            if lhs.shape[1] == rhs.shape[2] {
                // 'Vector' is a row vector (1 by N)
                var t = rhs
                for m in 0..<rhs.shape[0] {
                    t[mat: m] = lhs * rhs[mat: m]
                }
                return t
            }
        }
    } else if lhs.shape.count == 3 && rhs.shape.count == 2 {
        // 3D-Tensor * Matrix
        return rhs * lhs
    }
    fatalError("Cannot elementwise multiply (or divide) lhs shape of \(lhs.shape) with rhs shape of \(rhs.shape)")
}
public func * (lhs: Tensor, rhs: Double) -> Tensor {
    return lhs * Tensor(rhs)
}
public func * (lhs: Double, rhs: Tensor) -> Tensor {
    return Tensor(lhs) * rhs
}
// MARK: - Division Arithmetic
// SUPPORTS UP TO 3D-Tensor OPERATIONS (same shape up to ND-Tensor)
extension Tensor {
    public func inv() -> Tensor {
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
public func / (lhs: Tensor, rhs: Tensor) -> Tensor {
    return lhs * rhs.inv()
}
public func / (lhs: Tensor, rhs: Double) -> Tensor {
    return lhs / Tensor(rhs)
}
public func / (lhs: Double, rhs: Tensor) -> Tensor {
    return Tensor(lhs) / rhs
}
// MARK: - Matrix Multiplication Arithmetic
// SUPPORTS Vector by Vector, Matrix by Vector, Matrix by Matrix, Scalar by Matrix
infix operator <*> : MultiplicationPrecedence
public func <*> (lhs: Tensor, rhs: Tensor) -> Tensor {
    // If we have a scalar we just treat this as normal multiplication
    if lhs.shape.count == 0 || rhs.shape.count == 0 {
        return lhs * rhs
    }
    // Otherwise continue to matrix multiplication!
    precondition(lhs.shape.count == 2 && rhs.shape.count == 2, "Tensor lhs \(lhs.shape) and Tensor rhs \(rhs.shape) not compatible for matrix multiplication, both Tensor's must have 2 dimensions")
    precondition(lhs.shape[1] == rhs.shape[0], "Matrix lhs \(lhs.shape) and Matrix rhs \(rhs.shape) not compatible for matrix multiplication")
    var t: Tensor
    if lhs.shape[0] == 1 && rhs.shape[1] == 1 {
        t = Tensor(shape: [], repeating: 0)
        lhs.grid.withUnsafeBufferPointer { lhsPtr in
            rhs.grid.withUnsafeBufferPointer { rhsPtr in
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(lhs.shape[0]), Int32(rhs.shape[1]), Int32(lhs.shape[1]), 1, lhsPtr.baseAddress!, Int32(lhs.shape[1]), rhsPtr.baseAddress!, Int32(rhs.shape[1]), 0, &t.grid, Int32(1))
            }
        }
    } else {
        t = Tensor(shape: [lhs.shape[0], rhs.shape[1]], repeating: 0)
        lhs.grid.withUnsafeBufferPointer { lhsPtr in
            rhs.grid.withUnsafeBufferPointer { rhsPtr in
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(lhs.shape[0]), Int32(rhs.shape[1]), Int32(lhs.shape[1]), 1, lhsPtr.baseAddress!, Int32(lhs.shape[1]), rhsPtr.baseAddress!, Int32(rhs.shape[1]), 0, &t.grid, Int32(t.shape[1]))
            }
        }
    }
    return t
}
// MARK: - Unary Arithmetic
extension Tensor {
    public func exp() -> Tensor {
        var t = self
        grid.withUnsafeBufferPointer { gridPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                var N = Int32(count)
                vvexp(tPtr.baseAddress!, gridPtr.baseAddress!, &N)
            }
        }
        return t
    }
    public func log() -> Tensor {
        var t = self
        grid.withUnsafeBufferPointer { gridPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                var N = Int32(count)
                vvlog(tPtr.baseAddress!, gridPtr.baseAddress!, &N)
            }
        }
        return t
    }
    public func sin() -> Tensor {
        var t = self
        grid.withUnsafeBufferPointer { gridPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                var N = Int32(count)
                vvsin(tPtr.baseAddress!, gridPtr.baseAddress!, &N)
            }
        }
        return t
    }
    public func cos() -> Tensor {
        var t = self
        grid.withUnsafeBufferPointer { gridPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                var N = Int32(count)
                vvcos(tPtr.baseAddress!, gridPtr.baseAddress!, &N)
            }
        }
        return t
    }
    public func pow(_ a: Double) -> Tensor {
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
    public func sqrt() -> Tensor {
        var t = self
        grid.withUnsafeBufferPointer { gridPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                var N = Int32(count)
                vvsqrt(tPtr.baseAddress!, gridPtr.baseAddress!, &N)
            }
        }
        return t
    }
    public func sum() -> Double {
        var result = 0.0
        grid.withUnsafeBufferPointer { src in
            vDSP_sveD(src.baseAddress!, 1, &result, vDSP_Length(count))
        }
        return result
    }
    public func sum(axis: Int, keepDim: Bool = false) -> Tensor {
        precondition(axis < shape.count, "Axis not present in this tensor")
        if shape.count == 1 {
            return Tensor(shape: keepDim ? shape: [], grid: [self.sum()])
        } else if shape.count == 2 {
            var newShape = shape
            if keepDim { newShape[axis] = 1 } else { newShape.remove(at: axis) }
            if (axis == 0 && shape[axis] == 1) || (axis == 1 && shape[axis] == 1) { return Tensor(shape: newShape, grid: grid) }
            var t = Tensor(shape: newShape, repeating: 0)
            grid.withUnsafeBufferPointer { gridPtr in
                t.grid.withUnsafeMutableBufferPointer { tPtr in
                    if axis == 0 {
                        for a in 0..<shape[1] {
                            vDSP_sveD(gridPtr.baseAddress! + a, shape[1], tPtr.baseAddress! + a, vDSP_Length(shape[0]))
                        }
                    } else {
                        for a in 0..<shape[0] {
                            vDSP_sveD(gridPtr.baseAddress! + a * shape[1], 1, tPtr.baseAddress! + a, vDSP_Length(shape[1]))
                        }
                    }
                }
            }
            return t
        } else {
            // ND how tf?
            fatalError("ND sum along axis not implemented yet")
//            var newShape = shape
//            let offset = axis == shape.count - 1 ? -1 : 1
//            if keepDim { newShape[axis + offset] = 1 } else { newShape.remove(at: axis + offset) }
//            var t = Tensor(shape: newShape, repeating: 0)
//            grid.withUnsafeBufferPointer { gridPtr in
//                t.grid.withUnsafeMutableBufferPointer { tPtr in
//                    for a in 0..<newShape.reduce(1, { $0 * $1 }) {
//                        vDSP_sveD(gridPtr.baseAddress! + a, newShape.reduce(1) { $0 * $1 }, tPtr.baseAddress! + a, vDSP_Length(newShape.reduce(1) { $0 * $1 }))
//                    }
//                }
//            }
//            return t
        }
    }
}
// MARK: - Accessors
extension Tensor {
    var count: Int {
        return shape.reduce(1) { $0 * $1 }
    }
    var type: TensorType {
        if shape.count == 0 {
            return .scalar
        } else if shape.count == 2 && shape.contains(1) {
            return .vector
        } else if shape.count == 2 && !shape.contains(1) {
            return .matrix
        } else if shape.count == 3 {
            return .tensor3D
        } else {
            return .tensorND
        }
    }
}