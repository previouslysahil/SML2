//
//  Tensor.swift
//  
//
//  Created by Sahil Srivastava on 12/4/21.
//
//  IMPORTANT NOTES: 0_0
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
    public var shape: [Int]
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
        case column = "row vector"
        case row = "column vector"
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
    public init(_ vec: [Double]) {
        self.init(shape: [vec.count], repeating: 0.0)
        vec.withUnsafeBufferPointer { vecPtr in
            grid.withUnsafeMutableBufferPointer { gridPtr in
                cblas_dcopy(Int32(shape[0]), vecPtr.baseAddress!, 1, gridPtr.baseAddress!, 1)
            }
        }
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
        for mat in mats {
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
            let (leftover, reshaped) = extra()
            precondition(reshaped.count == 3, "Must be a 3D-Tensor")
            var t = Tensor(shape: [reshaped[1], reshaped[2]], repeating: 0)
            grid.withUnsafeBufferPointer { gridPtr in
                t.grid.withUnsafeMutableBufferPointer { tPtr in
                    cblas_dcopy(Int32(reshaped[1] * reshaped[2]), gridPtr.baseAddress! + m * (reshaped[1] * reshaped[2]), 1, tPtr.baseAddress!, 1)
                }
            }
            t.shape.insert(contentsOf: leftover.view, at: 0)
            return t
        }
        set(t) {
            let (_, reshaped) = extra()
            precondition(reshaped.count == 3, "Must be a 3D-Tensor")
            precondition(t.shape.count == 2 && t.shape[0] == reshaped[1] && t.shape[1] == reshaped[2], "Not compatible matrix dimensions")
            grid.withUnsafeMutableBufferPointer { gridPtr in
                t.grid.withUnsafeBufferPointer { tPtr in
                    cblas_dcopy(Int32(reshaped[1] * reshaped[2]), tPtr.baseAddress!, 1, gridPtr.baseAddress! + m * (reshaped[1] * reshaped[2]), 1)
                }
            }
        }
    }
    public subscript(row r: Int) -> Tensor {
        get {
            let (leftover, reshaped) = extra()
            precondition(reshaped.count == 2, "Must be a matrix")
            var t = Tensor(shape: [1, reshaped[1]], repeating: 0)
            grid.withUnsafeBufferPointer { gridPtr in
                t.grid.withUnsafeMutableBufferPointer { tPtr in
                    cblas_dcopy(Int32(reshaped[1]), gridPtr.baseAddress! + r * reshaped[1], 1, tPtr.baseAddress!, 1)
                }
            }
            t.shape.insert(contentsOf: leftover.view, at: 0)
            return t
        }
        set(t) {
            let (_, reshaped) = extra()
            if t.shape.count == 1 && t.shape[0] == reshaped[1] {
                grid.withUnsafeMutableBufferPointer { gridPtr in
                    t.grid.withUnsafeBufferPointer { tPtr in
                        cblas_dcopy(Int32(reshaped[1]), tPtr.baseAddress!, 1, gridPtr.baseAddress! + r * reshaped[1], 1)
                    }
                }
            } else {
                precondition(reshaped.count == 2, "Must be a matrix")
                precondition(t.shape.count == 2 && t.shape[0] == 1 && t.shape[1] == reshaped[1], "Not compatible vector dimensions")
                grid.withUnsafeMutableBufferPointer { gridPtr in
                    t.grid.withUnsafeBufferPointer { tPtr in
                        cblas_dcopy(Int32(reshaped[1]), tPtr.baseAddress!, 1, gridPtr.baseAddress! + r * reshaped[1], 1)
                    }
                }
            }
        }
    }
    public subscript(rows range: Range<Int>) -> Tensor {
        get {
            let (leftover, reshaped) = extra()
            precondition(reshaped.count == 2, "Must be a matrix")
            precondition(range.lowerBound >= 0 && range.upperBound <= reshaped[0], "Invalid range")
            var t = Tensor(shape: [(range.upperBound - range.lowerBound), reshaped[1]], repeating: 0)
            grid.withUnsafeBufferPointer { gridPtr in
                t.grid.withUnsafeMutableBufferPointer { tPtr in
                    cblas_dcopy(Int32(reshaped[1] * range.count), gridPtr.baseAddress! + range.lowerBound * reshaped[1], 1, tPtr.baseAddress!, 1)
                }
            }
            t.shape.insert(contentsOf: leftover.view, at: 0)
            return t
        }
    }
    public subscript(rows range: ClosedRange<Int>) -> Tensor {
        get {
            self[rows: Range(range)]
        }
    }
    public subscript(col c: Int) -> Tensor {
        get {
            let (leftover, reshaped) = extra()
            precondition(reshaped.count == 2, "Must be a matrix")
            var t = Tensor(shape: [reshaped[0], 1], repeating: 0)
            grid.withUnsafeBufferPointer { gridPtr in
                t.grid.withUnsafeMutableBufferPointer { tPtr in
                    cblas_dcopy(Int32(reshaped[0]), gridPtr.baseAddress! + c, Int32(reshaped[1]), tPtr.baseAddress!, 1)
                }
            }
            t.shape.insert(contentsOf: leftover.view, at: 0)
            return t
        }
        set(t) {
            let (_, reshaped) = extra()
            precondition(reshaped.count == 2, "Must be a matrix")
            precondition(t.shape.count == 2 && t.shape[1] == 1 && t.shape[0] == reshaped[0], "Not compatible vector dimensions")
            grid.withUnsafeMutableBufferPointer { gridPtr in
                t.grid.withUnsafeBufferPointer { tPtr in
                    cblas_dcopy(Int32(reshaped[0]), tPtr.baseAddress!, 1, gridPtr.baseAddress! + c, Int32(reshaped[1]))
                }
            }
        }
    }
    public subscript(val v: Int) -> Double {
        get {
            let (_, reshaped) = extra()
            precondition((reshaped.count == 2 && reshaped[1] == 1) || (reshaped.count == 1), "Must be a vector")
            return grid[v]
        }
        set(t) {
            let (_, reshaped) = extra()
            precondition((reshaped.count == 2 && reshaped[1] == 1) || (reshaped.count == 1), "Must be a vector")
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
        let (leftover, reshaped) = extra()
        precondition(reshaped.count == 2, "Must be a matrix")
        precondition(reshaped[0] == reshaped[1], "Must be a square matrix")
        var t = self
        t.grid.withUnsafeMutableBufferPointer { ptr in
            var ipiv = [__CLPK_integer](repeating: 0, count: reshaped[0] * reshaped[0])
            var lwork = __CLPK_integer(reshaped[1] * reshaped[1])
            var work = [CDouble](repeating: 0, count: Int(lwork))
            var error: __CLPK_integer = 0
            var nc = __CLPK_integer(reshaped[1])
            var m = nc
            var n = nc
            
            dgetrf_(&m, &n, ptr.baseAddress!, &nc, &ipiv, &error)
            dgetri_(&m, ptr.baseAddress!, &nc, &ipiv, &work, &lwork, &error)
            assert(error == 0, "Matrix not invertible")
        }
        t.shape.insert(contentsOf: leftover.view, at: 0)
        return t
    }
    
    public func transpose() -> Tensor {
        let (leftover, reshaped) = extra()
        // Scalar
        if reshaped.count == 0 {
            return self
        }
        // Row vector
        if reshaped.count == 1 {
            var t = Tensor(shape: [reshaped[0], 1], grid: grid)
            if shape.count > t.shape.count {
                t.shape.insert(contentsOf: leftover.view, at: 0)
            }
            return t
        } else if reshaped.count == 2 && reshaped[1] == 1 {
            // Column vector
            var t = Tensor(shape: [reshaped[1], reshaped[0]], grid: grid)
            t.shape.insert(contentsOf: leftover.view, at: 0)
            return t
        }
        precondition(reshaped.count == 2, "Must be a matrix or vector")
        var t = Tensor(shape: reshaped.view.reversed(), repeating: 0)
        grid.withUnsafeBufferPointer { gridPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                vDSP_mtransD(gridPtr.baseAddress!, 1, tPtr.baseAddress!, 1, vDSP_Length(t.shape[0]), vDSP_Length(t.shape[1]))
            }
        }
        t.shape.insert(contentsOf: leftover.view, at: 0)
        return t
    }
    public func diag() -> Tensor {
        let (leftover, reshaped) = extra()
        if reshaped.count == 1 {
            // Row vector
            var t = Tensor(shape: shape, repeating: 0)
            t.grid[0] = grid[0]
            return t
        } else if reshaped.count == 2 && reshaped[1] == 1 {
            // Column vector
            var t = Tensor(shape: shape, repeating: 0)
            t.grid[0] = grid[0]
            return t
        }
        precondition(reshaped.count == 2, "Must be a matrix or vector")
        var t = Tensor(shape: shape, repeating: 0)
        grid.withUnsafeBufferPointer { gridPtr in
            t.grid.withUnsafeMutableBufferPointer { tPtr in
                cblas_dcopy(Int32(min(reshaped[0], reshaped[1])), gridPtr.baseAddress!, Int32(reshaped[1] + 1), tPtr.baseAddress!, Int32(reshaped[1] + 1))
            }
        }
        t.shape.insert(contentsOf: leftover.view, at: 0)
        return t
    }
}
// MARK: - Addition Arithmetic
// SUPPORTS UP TO 3D-Tensor OPERATIONS (same shape up to ND-Tensor)
public func + (lhs: Tensor, rhs: Tensor) -> Tensor {
    // Remove unnecessary dimensions for operation then add them back if necessary
    let l = lhs.extra()
    let r = rhs.extra()
    var res = trueAdd(Tensor(shape: Array(l.reshaped.view), grid: lhs.grid), Tensor(shape: Array(r.reshaped.view), grid: rhs.grid))
    while res.shape.count < lhs.shape.count || res.shape.count < rhs.shape.count {
        res.shape.insert(1, at: 0)
    }
    return res
}
public func + (lhs: Tensor, rhs: Double) -> Tensor {
    return lhs + Tensor(rhs)
}
public func + (lhs: Double, rhs: Tensor) -> Tensor {
    return Tensor(lhs) + rhs
}
// MARK: Add Arithmetic True
// Assumes extra shape has been rid
fileprivate func trueAdd(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
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
    }
    fatalError("Cannot elementwise add (or subtract) lhs shape of \(lhs.shape) with rhs shape of \(rhs.shape)")
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
    // Remove unnecessary dimensions for operation then add them back if necessary
    let l = lhs.extra()
    let r = rhs.extra()
    var res = trueMult(Tensor(shape: Array(l.reshaped.view), grid: lhs.grid), Tensor(shape: Array(r.reshaped.view), grid: rhs.grid))
    while res.shape.count < lhs.shape.count || res.shape.count < rhs.shape.count {
        res.shape.insert(1, at: 0)
    }
    return res
}
public func * (lhs: Tensor, rhs: Double) -> Tensor {
    return lhs * Tensor(rhs)
}
public func * (lhs: Double, rhs: Tensor) -> Tensor {
    return Tensor(lhs) * rhs
}
// MARK: Multiplication Arithmetic True
// Assumes extra shape has been rid
fileprivate func trueMult(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
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
    }
    fatalError("Cannot elementwise multiply (or divide) lhs shape of \(lhs.shape) with rhs shape of \(rhs.shape)")
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
// REVISIT NOT MADE FOR EXTRA SHAPE YET
infix operator <*> : MultiplicationPrecedence
public func <*> (lhs: Tensor, rhs: Tensor) -> Tensor {
    // Remove unnecessary dimensions for operation then add them back if necessary
    let l = lhs.extra()
    let r = rhs.extra()
    var res = trueMatMult(Tensor(shape: Array(l.reshaped.view), grid: lhs.grid), Tensor(shape: Array(r.reshaped.view), grid: rhs.grid))
    while res.shape.count < lhs.shape.count || res.shape.count < rhs.shape.count {
        res.shape.insert(1, at: 0)
    }
    return res
}
fileprivate func trueMatMult(_ lhs: Tensor, _ rhs: Tensor) -> Tensor {
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
    var t: Tensor
    t = Tensor(shape: [lhs_shape[0], rhs_shape[1]], repeating: 0)
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
    public func sumDiag() -> Double {
        let (_, reshaped) = extra()
        var result = 0.0
        grid.withUnsafeBufferPointer { gridPtr in
            vDSP_sveD(gridPtr.baseAddress!, reshaped[1] + 1, &result, vDSP_Length(Swift.min(reshaped[0], reshaped[1])))
        }
        return result
    }
    public func sum(axis: Int, keepDim: Bool = false) -> Tensor {
        var newShape = shape
        // Keeping dimensions
        if keepDim { newShape[axis] = 1 } else { newShape.remove(at: axis) }
        // Check for summing 1 size axis
        if shape[axis] == 1 { return Tensor(shape: newShape, grid: grid) }
        precondition(axis < shape.count, "Axis not present in this tensor")
        // Remove extra shape
        let (_, reshaped) = extra()
        if reshaped.count == 1 {
            return Tensor(shape: newShape, grid: [self.sum()])
        } else if reshaped.count == 2 {
            var t = Tensor(shape: newShape, repeating: 0)
            grid.withUnsafeBufferPointer { gridPtr in
                t.grid.withUnsafeMutableBufferPointer { tPtr in
                    if axis == 0 {
                        for a in 0..<reshaped[1] {
                            vDSP_sveD(gridPtr.baseAddress! + a, reshaped[1], tPtr.baseAddress! + a, vDSP_Length(reshaped[0]))
                        }
                    } else {
                        for a in 0..<reshaped[0] {
                            vDSP_sveD(gridPtr.baseAddress! + a * reshaped[1], 1, tPtr.baseAddress! + a, vDSP_Length(reshaped[1]))
                        }
                    }
                }
            }
            return t
        } else {
            // ND how tf?
            fatalError("ND sum along axis not implemented yet")
//            var newShape = shape
//            if keepDim { newShape[axis] = 1 } else { newShape.remove(at: axis) }
//            if (axis == 0 && shape[axis] == 1) || (axis == 1 && shape[axis] == 1) { return Tensor(shape: newShape, grid: grid) }
//            var t = Tensor(shape: newShape, repeating: 0)
//            grid.withUnsafeBufferPointer { gridPtr in
//                if axis == 0 {
//                    for a in 0..<shape[0] {
//                        let tmp = t
//                        let N = t.count
//                        t.grid.withUnsafeMutableBufferPointer { tPtr in
//                            tmp.grid.withUnsafeBufferPointer { tmpPtr in
//                                vDSP_vaddD(gridPtr.baseAddress! + a * N, 1, tmpPtr.baseAddress!, 1, tPtr.baseAddress!, 1, vDSP_Length(N))
//                            }
//                        }
//                    }
//                } else if axis == 1 {
//
//                }
//            }
//            return t
        }
    }
    // Calculates mean and std for each column!
    public func zscore() -> (norm: Tensor, mean: Tensor, std: Tensor) {
        let (leftover, reshaped) = extra()
        var t = self
        var mean = Tensor(shape: [1, reshaped[1]], repeating: 0)
        var std = Tensor(shape: [1, reshaped[1]], repeating: 0)
        mean.grid.withUnsafeMutableBufferPointer { meanPtr in
            std.grid.withUnsafeMutableBufferPointer { stdPtr in
                for c in 0..<reshaped[1] {
                    t.grid.withUnsafeMutableBufferPointer { tPtr in
                        grid.withUnsafeBufferPointer { gridPtr in
                            vDSP_normalizeD(gridPtr.baseAddress! + c, reshaped[1], tPtr.baseAddress! + c, reshaped[1], meanPtr.baseAddress! + c, stdPtr.baseAddress! + c, vDSP_Length(reshaped[0]))
                        }
                    }
                }
            }
        }
        t.shape.insert(contentsOf: leftover.view, at: 0)
        return (t, mean, std)
    }
}
// MARK: Activation Functions
extension Tensor {
    public func sigmoid() -> Tensor {
        return 1.0 / (1.0 + (-self).exp())
    }
    public func relu() -> Tensor {
        var grid = self.grid
        grid.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<ptr.count {
                if ptr[i] <= 0 {
                    ptr[i] = 0
                }
            }
        }
        return Tensor(shape: shape, grid: grid)
    }
    public func drelu() -> Tensor {
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
        return Tensor(shape: shape, grid: grid)
    }
    public func lrelu() -> Tensor {
        var grid = self.grid
        grid.withUnsafeMutableBufferPointer { ptr in
            for i in 0..<ptr.count {
                if ptr[i] <= 0 {
                    ptr[i] = 0.2 * ptr[i]
                }
            }
        }
        return Tensor(shape: shape, grid: grid)
    }
    public func dlrelu() -> Tensor {
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
        return Tensor(shape: shape, grid: grid)
    }
}
// MARK: - Accessors
extension Tensor {
    public var count: Int {
        return shape.reduce(1) { $0 * $1 }
    }
    public var type: TensorType {
        let (_, reshaped) = extra()
        if reshaped.count == 0 {
            return .scalar
        } else if reshaped.count == 1 {
            return .row
        } else if reshaped.count == 2 {
            if reshaped[1] == 1 {
                return .column
            }
            return .matrix
        } else if reshaped.count == 3 {
            return .tensor3D
        } else if reshaped.count >= 4 {
            return .tensorND
        }
        fatalError("Unable to descibe Tensor")
    }
    public func extra() -> (leftover: TensorArrayView<Int>, reshaped: TensorArrayView<Int>) {
        var t = 0
        // Should rarely ever even run so shouldnt really hinder performance
        while t < shape.count && shape[t] == 1 {
            t += 1
        }
        // ArraySlice so we arent making unnecessary copies performance
        return (TensorArrayView<Int>(shape.prefix(upTo: t)), TensorArrayView<Int>(shape.suffix(from: t)))
    }
}
// MARK: TensorArrayView
public struct TensorArrayView<T> {
    public var view: ArraySlice<T>
    public var count: Int {
        return view.count
    }
    
    public subscript(i: Int) -> T {
        precondition(view.startIndex + i < view.endIndex, "Index out of ArrayView bounds")
        return view[view.startIndex + i]
    }
    
    public init(_ slice: ArraySlice<T>) {
        self.view = slice
    }
}

// line 683
