//
//  Tensor.swift
//  
//
//  Created by Sahil Srivastava on 12/4/21.
//

import Foundation
import Accelerate

// MARK: Tensorable
public protocol Tensorable {
    associatedtype Scalar: Floatable, Codable
    
    var shape: Shape {get set}
    var grid: [Scalar] {get set}
    
    init(shape: [Int], grid: [Scalar])
    init(shape: Shape, grid: [Scalar])
    
    init(shape: [Int], repeating: Scalar)
    init(shape: Shape, repeating: Scalar)
    init(_ val: Scalar)
    init(_ vec: [Scalar])
    init(_ vec: [Scalar], type: VectorType)
    init(_ mat: [[Scalar]])
    init(_ t3D: [[[Scalar]]])
    init(_ t4D: [[[[Scalar]]]])
    
    subscript(pos: Int...) -> Scalar {get set}
    subscript(t3D d: Int) -> Self {get set}
    subscript(t3Ds range: Range<Int>) -> Self {get}
    subscript(t3Ds range: ClosedRange<Int>) -> Self {get}
    subscript(mats range: Range<Int>) -> Self {get}
    subscript(mats range: ClosedRange<Int>) -> Self {get}
    subscript(mat m: Int) -> Self {get set}
    subscript(row r: Int) -> Self {get set}
    subscript(rows range: Range<Int>) -> Self {get}
    subscript(rows range: ClosedRange<Int>) -> Self {get}
    subscript(col c: Int) -> Self {get set}
    subscript(cols range: Range<Int>) -> Self {get}
    subscript(cols range: ClosedRange<Int>) -> Self {get}
    subscript(val v: Int) -> Scalar {get set}
    
    static func random(shape: [Int], min: Scalar, max: Scalar) -> Self
    static func random_xavier(shape: [Int], ni: Int, no: Int) -> Self
    
    func matInv() -> Self
    func transpose() -> Self
    func diag() -> Self
    
    func conv2D(with kernel: Self, type: TensorConvType) -> Self
    func conv2D(with kernel: Self) -> Self
    func conv2D_mine(with kernel: Self) -> Self
    func conv2D_valid(with kernel: Self) -> Self
    func conv2D_same(with kernel: Self) -> Self
    func conv2D_full(with kernel: Self) -> Self
    func pool2D_max(size: Int) -> Self
    func pool2D_max(size: Int, strd: Int) -> (Self, [Int])
    func pad(_ w: Int, _ h: Int) -> Self
    func pad(_ le: Int, _ ri: Int, _ to: Int, _ bo: Int) -> Self
    func trim(_ w: Int, _ h: Int) -> Self
    func trim(_ le: Int, _ ri: Int, _ to: Int, _ bo: Int) -> Self
    func rot180() -> Self
    
    static func + (lhs: Self, rhs: Self) -> Self
    static func + (lhs: Self, rhs: Scalar) -> Self
    static func + (lhs: Scalar, rhs: Self) -> Self
    
    prefix static func -(t: Self) -> Self
    static func - (lhs: Self, rhs: Self) -> Self
    static func - (lhs: Self, rhs: Scalar) -> Self
    static func - (lhs: Scalar, rhs: Self) -> Self
    
    static func * (lhs: Self, rhs: Self) -> Self
    static func * (lhs: Self, rhs: Scalar) -> Self
    static func * (lhs: Scalar, rhs: Self) -> Self
    
    func inv() -> Self
    static func / (lhs: Self, rhs: Self) -> Self
    static func / (lhs: Self, rhs: Scalar) -> Self
    static func / (lhs: Scalar, rhs: Self) -> Self
    
    static func <*> (lhs: Self, rhs: Self) -> Self
    
    func exp() -> Self
    func log() -> Self
    func sin() -> Self
    func cos() -> Self
    func pow(_ a: Scalar) -> Self
    func sqrt() -> Self
    func max() -> Scalar
    func max() -> (Scalar, Int)
    func sum() -> Scalar
    func sumDiag() -> Scalar
    func sum(axis: Int, keepDim: Bool) -> Self
    func zscore() -> (norm: Self, mean: Self, std: Self)
    func zscore_image() -> (norm: Self, mean: Self, std: Self)
    
    func sigmoid() -> Self
    func relu() -> Self
    func drelu() -> Self
    func lrelu() -> Self
    func dlrelu() -> Self
    
    var count: Int {get}
    var type: TensorType {get}
}
// MARK: Floatable
public protocol Floatable: FloatingPoint, ExpressibleByFloatLiteral {
    func powd(by val: Self) -> Self
}
extension Float: Floatable {
    public func powd(by val: Float) -> Float { return pow(self, val) }
}

extension Double: Floatable {
    public func powd(by val: Double) -> Double { return pow(self, val) }
}
// MARK: - Enums
public enum VectorType {
    case row
    case column
}
public enum TensorType: String {
    case scalar = "scalar"
    case column = "row vector"
    case row = "column vector"
    case matrix = "matrix"
    case tensor3D = "tensor3D"
    case tensor4D = "tensor4D"
    case tensorND = "tensorND"
}
public enum TensorConvType {
    case valid
    case same
    case full
}
// MARK: Int Array Ext
extension Array where Element == Int {
    public enum IntArrayConvType {
        case valid
        case same
        case full
    }
    public func conv2D_shape(with kernel_shape: [Int], type: IntArrayConvType) -> [Int] {
        precondition(self.count == 2 && self[1] != 1 && kernel_shape.count == 2 && kernel_shape[1] != 1, "Image and kernel must be matrices")
        switch type {
        case .valid:
            return conv2D_valid_shape(with: kernel_shape)
        case .same:
            return conv2D_same_shape(with: kernel_shape)
        case .full:
            return conv2D_full_shape(with: kernel_shape)
        }
    }
    public func conv2D_shape(with kernel_shape: [Int]) -> [Int] {
        precondition(self.count == 2 && self[1] != 1 && kernel_shape.count == 2 && kernel_shape[1] != 1, "Image and kernel must be matrices")
        // Padding by vDSP
        let vertPad = (kernel_shape[0] - 1) / 2
        let horzPad = (kernel_shape[1] - 1) / 2
        // Dimension of output sans padding
        let tempShape = [self[0] - kernel_shape[0] + 1, self[1] - kernel_shape[1] + 1]
        // Output shape
        let resShape = [tempShape[0] + vertPad * 2, tempShape[1] + horzPad * 2]
        return resShape
    }
    public func conv2D_valid_shape(with kernel_shape: [Int]) -> [Int] {
        return self.conv2D_shape(with: kernel_shape).trim_shape((kernel_shape[0] - 1) / 2, (kernel_shape[1] - 1) / 2)
    }
    public func conv2D_same_shape(with kernel_shape: [Int]) -> [Int] {
        return self.pad_shape((kernel_shape[0] - 1) / 2, (kernel_shape[1] - 1) / 2).conv2D_valid_shape(with: kernel_shape)
    }
    public func conv2D_full_shape(with kernel_shape: [Int]) -> [Int] {
        return self.pad_shape(kernel_shape[0] - 1, kernel_shape[1] - 1).conv2D_valid_shape(with: kernel_shape)
    }
    public func pool2D_max_shape(size: Int, strd: Int = 1) -> [Int] {
        precondition(self.count == 2 && self[1] != 1, "Must be a matrix")
        return [((self[0] - size) / strd) + 1, ((self[1] - size) / strd) + 1]
    }
    public func pad_shape(_ le: Int, _ ri: Int, _ to: Int, _ bo: Int) -> [Int] {
        precondition(self.count == 2 && self[1] != 1, "Must be a matrix")
        let resShape = [self[0] + le + ri, self[1] + to + bo]
        return resShape
    }
    public func pad_shape(_ w: Int, _ h: Int) -> [Int] {
        return pad_shape(w, w, h, h)
    }
    public func trim_shape(_ le: Int, _ ri: Int, _ to: Int, _ bo: Int) -> [Int] {
        precondition(self.count == 2 && self[1] != 1, "Must be a matrix")
        let resShape = [self[0] - le - ri, self[1] - to - bo]
        return resShape
    }
    public func trim_shape(_ w: Int, _ h: Int) -> [Int] {
        return trim_shape(w, w, h, h)
    }
}
extension Array where Element == Int {
    // basically seperating leading ones
    public func seperate() -> (leftover: ArrayView<Int>, main: ArrayView<Int>) {
        var t = 0
        // Should rarely ever even run so shouldnt really hinder performance
        while t < self.count && self[t] == 1 {
            t += 1
        }
        // ArraySlice so we arent making unnecessary copies for performance
        return (ArrayView<Int>(self.prefix(upTo: t)), ArrayView<Int>(self.suffix(from: t)))
    }
}
// MARK: Shape
public struct Shape: Equatable, CustomStringConvertible, Swift.Sequence {
    
    public var description: String {
        return arr.description
    }
    
    public static func == (lhs: Shape, rhs: Shape) -> Bool {
        return lhs.arr == rhs.arr
    }
    
    public static func == (lhs: Shape, rhs: [Int]) -> Bool {
        return lhs.arr == rhs
    }
    
    public static func == (lhs: [Int], rhs: Shape) -> Bool {
        return lhs == rhs.arr
    }
    
    public typealias Iterator = IndexingIterator<[Int]>
    
    public func makeIterator() -> IndexingIterator<[Int]> {
        return arr.makeIterator()
    }
    
    public private(set) var leftover: ArrayView<Int>
    public private(set) var main: ArrayView<Int>
    
    private var arr: [Int]
    
    public var count: Int { arr.count }
    
    public init(_ arr: [Int]) {
        (self.leftover, self.main) = arr.seperate()
        // Store actual
        self.arr = arr
    }
    
    public func strip() -> [Int] {
        return arr
    }
    
    public func reduce() -> Int {
        arr.reduce(1) { $0 * $1 }
    }
    
    public mutating func insert(_ newElement: Int, at i: Int) {
        arr.insert(newElement, at: i)
        // Reset leftover and main
        (self.leftover, self.main) = arr.seperate()
    }
    
    public mutating func insert<C>(contentsOf newElements: C, at i: Int) where C : Collection, C.Element == Int {
        arr.insert(contentsOf: newElements, at: i)
        // Reset leftover and main
        (self.leftover, self.main) = arr.seperate()
    }
    
    @discardableResult
    public mutating func remove(at index: Int) -> Int {
        let removed = arr.remove(at: index)
        // Reset leftover and main
        (self.leftover, self.main) = arr.seperate()
        return removed
    }
    
    public subscript(i: Int) -> Int {
        get {
            return arr[i]
        }
        set {
            arr[i] = newValue
            // Reset leftover and main
            (self.leftover, self.main) = arr.seperate()
        }
    }
    
    public subscript(r: Range<Int>) -> ArraySlice<Int> {
        get {
            return arr[r]
        }
        set {
            arr[r] = newValue
            // Reset leftover and main
            (self.leftover, self.main) = arr.seperate()
        }
    }
    
    public subscript(r: ClosedRange<Int>) -> ArraySlice<Int> {
        get {
            return arr[r]
        }
        set {
            arr[r] = newValue
            // Reset leftover and main
            (self.leftover, self.main) = arr.seperate()
        }
    }
}
// MARK: ArrayView
public struct ArrayView<T: Equatable>: Equatable {
    public static func == (lhs: ArrayView<T>, rhs: ArrayView<T>) -> Bool {
        return lhs.view == rhs.view
    }
    
    public var view: ArraySlice<T>
    public var count: Int {
        return view.count
    }
    
    public subscript(i: Int) -> T {
        precondition(view.startIndex + i < view.endIndex, "Index out of ArrayView bounds")
        return view[view.startIndex + i]
    }
    
    public subscript(r: Range<Int>) -> ArraySlice<T> {
        precondition(view.startIndex + r.endIndex - 1 < view.endIndex, "Index out of ArrayView bounds")
        return view[view.startIndex + r.startIndex..<view.startIndex + r.endIndex]
    }
    
    public subscript(r: ClosedRange<Int>) -> ArraySlice<T> {
        return self[Range(r)]
    }
    
    public init(_ slice: ArraySlice<T>) {
        self.view = slice
    }
}

// line 683
