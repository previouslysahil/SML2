//
//  SML2FTensorTests.swift
//  
//
//  Created by Sahil Srivastava on 1/10/22.
//

import XCTest
@testable import SML2

final class SML2FTensorTests: XCTestCase {
    
    func testInits() throws {
        var tensor: FTensor

        tensor = FTensor(shape: [2, 4], grid: Array(repeating: 1.0, count: 2 * 4))
        XCTAssert(tensor.shape == [2, 4] && tensor.grid.count == 8 && tensor.grid.allSatisfy({ $0 == 1 }), "init")
        tensor = FTensor(shape: [2, 4], repeating: 10)
        XCTAssert(tensor.shape == [2, 4] && tensor.grid.count == 8 && tensor.grid.allSatisfy({ $0 == 10 }), "init repeating")

        let val: Float = 12.0
        tensor = FTensor(val)
        XCTAssert(tensor.shape == [] && tensor.grid == [12.0], "init float")
        let arr_one: [Float] = [1, 2, 3, 4, 5]
        tensor = FTensor(arr_one, type: .row)
        XCTAssert(tensor.shape == [1, 5] && tensor.grid == arr_one, "init 1D arr to row vec")
        tensor = FTensor(arr_one, type: .column)
        XCTAssert(tensor.shape == [5, 1] && tensor.grid == arr_one, "init 1D arr to column vec")
        let arr_two: [[Float]] = [arr_one, arr_one, arr_one]
        tensor = FTensor(arr_two)
        XCTAssert(tensor.shape == [3, 5] && tensor.grid == arr_two.flatMap({ $0 }), "init 2D arr to mat")
        let arr_three: [[[Float]]] = [arr_two, arr_two]
        tensor = FTensor(arr_three)
        XCTAssert(tensor.shape == [2, 3, 5] && tensor.grid == arr_three.flatMap({ $0.flatMap({ $0 })}), "init 3D arr to mats")
    }

    func testQuery() throws {
        var tensor: FTensor

        let arr_one: [Float] = [1, 2, 3, 4, 5]
        tensor = FTensor(arr_one)
        XCTAssert(tensor[1] == 2.0, "query get row vec")
        tensor[4] = -1
        XCTAssert(tensor[4] == -1, "query set row vec")
        tensor = FTensor(arr_one, type: .row)
        XCTAssert(tensor[0, 1] == 2.0, "query get row vec")
        tensor[0, 4] = -1
        XCTAssert(tensor[0, 4] == -1, "query set row vec")
        tensor = FTensor(arr_one, type: .column)
        XCTAssert(tensor[1, 0] == 2.0, "query get col vec")
        tensor[4, 0] = -1
        XCTAssert(tensor[4, 0] == -1, "query set col vec")

        tensor = FTensor(arr_one, type: .row)
        XCTAssert(tensor[val: 1] == 2.0, "query get row vec")
        tensor[val: 4] = -1
        XCTAssert(tensor[val: 4] == -1, "query set row vec")
        tensor = FTensor(arr_one, type: .column)
        XCTAssert(tensor[val: 1] == 2.0, "query get col vec")
        tensor[val: 4] = -1
        XCTAssert(tensor[val: 4] == -1, "query set col vec")

        let arr_two: [[Float]] = [arr_one, [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
        tensor = FTensor(arr_two)
        XCTAssert(tensor[1, 2] == 8.0, "query get mat")
        tensor[2, 3] = -1
        XCTAssert(tensor[2, 3] == -1, "query set mats")

        XCTAssert(tensor[row: 0][val: 0] == 1.0, "query get val from vec from mat")
        XCTAssert(tensor[col: 0][val: 0] == 1.0, "query get val from vec from mat")
        XCTAssert(tensor[row: 2][val: 4] == 15.0, "query get val from vec from mat")
        XCTAssert(tensor[col: 4][val: 2] == 15.0, "query get val from vec from mat")

        tensor[row: 0][val: 0] = 17.0
        XCTAssert(tensor[row: 0][val: 0] == 17.0, "query get val from vec from mat")
        tensor[col: 0][val: 0] = 18.0
        XCTAssert(tensor[col: 0][val: 0] == 18.0, "query get val from vec from mat")
        tensor[row: 2][val: 4] = 19.0
        XCTAssert(tensor[row: 2][val: 4] == 19.0, "query get val from vec from mat")
        tensor[col: 4][val: 2] = 20.0
        XCTAssert(tensor[col: 4][val: 2] == 20.0, "query get val from vec from mat")

        let arr_three: [[[Float]]] = [arr_two, arr_two.map { $0.map { 15.0 + $0 } }]
        tensor = FTensor(arr_three)
        XCTAssert(tensor[1, 0, 3] == 19.0 && tensor[1, 2, 4] == 30.0, "query get mats")
        tensor[1, 1, 3] = -1
        XCTAssert(tensor[1, 1, 3] == -1, "query set mats")

        tensor = FTensor(arr_two)
        XCTAssert(tensor[col: 1].grid == [2, 7, 12], "query col of matrix")
        tensor[col: 1] = FTensor([1, 1, 1], type: .column)
        XCTAssert(tensor[col: 1].grid == [1, 1, 1], "query set col of matrix")

        tensor = FTensor(arr_two)
        XCTAssert(tensor[row: 1].grid == [6, 7, 8, 9, 10], "query row of matrix")
        tensor[row: 1] = FTensor([1, 1, 1, 1, 1], type: .row)
        XCTAssert(tensor[row: 1].grid == [1, 1, 1, 1, 1], "query set row of matrix")

        tensor = FTensor(arr_three)
        XCTAssert(tensor[mat: 0] == FTensor(arr_two), "query matrix of 3D-Tensor")
        tensor[mat: 1] = FTensor(arr_two)
        XCTAssert(tensor[mat: 1] == FTensor(arr_two), "query set matrix of 3D-Tensor")

        let arr_three_1 = arr_three.map { $0.map { $0.map { -1.35 + $0 } } }
        let arr_four = [arr_three, arr_three_1]
        tensor = FTensor(arr_four)
        XCTAssert(tensor[t3D: 1] == FTensor(arr_three_1), "query 3D-Tensor of 4D-Tensor")
        tensor[t3D: 1] = FTensor(arr_three.map { $0.map { $0.map { -1.95 + $0 } } })
        XCTAssert(tensor[t3D: 1] == FTensor(arr_three.map { $0.map { $0.map { -1.95 + $0 } } }), "query set 3D-Tensor of 4D-Tensor")
    }

    func testQueryRange() throws {
        var tensor: FTensor
        tensor = FTensor(shape: [4, 3], grid: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13])

        // Column ranges
        XCTAssert(tensor[cols: 1..<3] == FTensor(shape: [4, 2], grid: [2, 3, 5, 6, 8, 9, 12, 13]), "query columns")
        XCTAssert(tensor[cols: 0..<2] == FTensor(shape: [4, 2], grid: [1, 2, 4, 5, 7, 8, 11, 12]), "query columns")
        XCTAssert(tensor[cols: 1..<2] == FTensor(shape: [4, 1], grid: [2, 5, 8, 12]), "query columns")

        // Row ranges
        XCTAssert(tensor[rows: 1..<3] == FTensor(shape: [2, 3], grid: [4, 5, 6, 7, 8, 9]), "query rows")
        XCTAssert(tensor[rows: 0..<2] == FTensor(shape: [2, 3], grid: [1, 2, 3, 4, 5, 6]), "query rows")
        XCTAssert(tensor[rows: 1..<2] == FTensor(shape: [1, 3], grid: [4, 5, 6]), "query rows")

        // Mat ranges
        var t3D = [Float]()
        for i in 1...60 {
            t3D.append(Float(i))
        }
        tensor = FTensor(shape: [5, 4, 3], grid: t3D)
        XCTAssert(tensor[mats: 0..<2] == FTensor(shape: [2, 4, 3], grid: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]), "query mats")
        XCTAssert(tensor[mats: 2..<3] == FTensor(shape: [1, 4, 3], grid: [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0]), "query mats")
        XCTAssert(tensor[mats: 3..<5] == FTensor(shape: [2, 4, 3], grid: [37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0]), "query mats")

        // t3D ranges
        var t4D = [Float]()
        for i in 1...600 {
            t4D.append(Float(i))
        }
        tensor = FTensor(shape: [10, 5, 4, 3], grid: t4D)
        XCTAssert(tensor[t3Ds: 0..<2] == FTensor(shape: [2, 5, 4, 3], grid: Array(t4D[60 * 0..<60 * 2])), "query t3D")
        XCTAssert(tensor[t3Ds: 2..<5] == FTensor(shape: [3, 5, 4, 3], grid: Array(t4D[60 * 2..<60 * 5])), "query t3D")
        XCTAssert(tensor[t3Ds: 9..<10] == FTensor(shape: [1, 5, 4, 3], grid: Array(t4D[60 * 9..<60 * 10])), "query t3D")
    }

    func testRandomInit() throws {
        var tensor: FTensor

        tensor = FTensor.random(shape: [2, 3], min: -10, max: 25)
        XCTAssert(tensor.grid.count == 6, "random init")
        tensor = FTensor.random_xavier(shape: [2, 3], ni: -4, no: 10)
        XCTAssert(tensor.grid.count == 6, "random init")
    }

    // HAVE NOT TESTED INVERT AND TRANSPOSE

    func testAdd() throws {
        var tensor1: FTensor
        var tensor2: FTensor

        tensor1 = FTensor(3)
        tensor2 = FTensor(4)
        XCTAssert((tensor1 + tensor2) == FTensor(7), "add float + float")
        XCTAssert((tensor1 + tensor2).shape == tensor2.shape, "add float + float check shape")
        XCTAssert((tensor2 + tensor1).shape == tensor2.shape, "add float + float check shape")

        var val: Float = 1
        tensor1 = FTensor(val)
        let arr_one: [Float] = [1, 2, 3, 4, 5]
        tensor2 = FTensor(arr_one, type: .row)
        XCTAssert((tensor1 + tensor2) == FTensor(arr_one.map { $0 + val }, type: .row), "add float + row vector")
        XCTAssert((tensor2 + tensor1) == FTensor(arr_one.map { $0 + val }, type: .row), "add row vector + float")
        XCTAssert((tensor1 + tensor2).shape == tensor2.shape, "add float + row vector check shape")
        XCTAssert((tensor2 + tensor1).shape == tensor2.shape, "add row vector + float check shape")

        tensor1 = FTensor(val)
        tensor2 = FTensor(arr_one, type: .column)
        XCTAssert((tensor1 + tensor2) == FTensor(arr_one.map { $0 + val }, type: .column), "add float + column vector")
        XCTAssert((tensor2 + tensor1) == FTensor(arr_one.map { $0 + val }, type: .column), "add column vector + float")
        XCTAssert((tensor1 + tensor2).shape == tensor2.shape, "add float + column vector check shape")
        XCTAssert((tensor2 + tensor1).shape == tensor2.shape, "add column vector + float check shape")

        val = 3
        tensor1 = FTensor(val)
        let arr_two: [[Float]] = [arr_one, arr_one.map {$0 + 5}]
        tensor2 = FTensor(arr_two)
        XCTAssert((tensor1 + tensor2) == FTensor(arr_two.map { $0.map { $0 + val } }), "add float + matrix")
        XCTAssert((tensor2 + tensor1) == FTensor(arr_two.map { $0.map { $0 + val } }), "add matrix + float")
        XCTAssert((tensor1 + tensor2).shape == tensor2.shape, "add float + matrix check shape")
        XCTAssert((tensor2 + tensor1).shape == tensor2.shape, "add matrix + float check shape")

        tensor1 = FTensor(arr_one, type: .row)
        tensor2 = FTensor(arr_one, type: .row)
        XCTAssert((tensor1 + tensor2) == FTensor(arr_one.map { $0 + $0 }, type: .row), "add vec of same size + vec of same size")
        XCTAssert((tensor2 + tensor1) == FTensor(arr_one.map { $0 + $0 }, type: .row), "add vec of same size + vec of same size")
        XCTAssert((tensor1 + tensor2).shape == tensor2.shape, "add vec + vec check shape")
        XCTAssert((tensor2 + tensor1).shape == tensor2.shape, "add vec + vec check shape")

        tensor1 = FTensor(arr_two)
        tensor2 = FTensor(arr_two)
        XCTAssert((tensor1 + tensor2) == FTensor(arr_two.map { $0.map { $0 + $0 } }), "add matrix of same size + matrix of same size")
        XCTAssert((tensor2 + tensor1) == FTensor(arr_two.map { $0.map { $0 + $0 } }), "add matrix of same size + matrix of same size")
        XCTAssert((tensor1 + tensor2).shape == tensor2.shape, "add matrix + matrix check shape")
        XCTAssert((tensor2 + tensor1).shape == tensor2.shape, "add matrix + matrix check shape")

        tensor1 = FTensor([arr_one, arr_one, arr_one, arr_one])
        tensor2 = FTensor([10, 11, 12, 13], type: .column)
        let arr_test1 = [arr_one.map { $0 + 10 }, arr_one.map { $0 + 11 }, arr_one.map { $0 + 12 }, arr_one.map { $0 + 13 }]
        XCTAssert((tensor1 + tensor2) == FTensor(arr_test1), "add matrix with column vector")
        XCTAssert((tensor2 + tensor1) == FTensor(arr_test1), "add column vector with matrix")

        tensor1 = FTensor([arr_one, arr_one, arr_one, arr_one])
        tensor2 = FTensor([5, 4, 3, 2, 1], type: .row)
        let arr_test2: [Float] = [6, 6, 6, 6, 6]
        XCTAssert((tensor1 + tensor2) == FTensor([arr_test2, arr_test2, arr_test2, arr_test2]), "add a matrix with a row vector")
        XCTAssert((tensor2 + tensor1) == FTensor([arr_test2, arr_test2, arr_test2, arr_test2]), "add a row vector with a matrix")

        tensor1 = FTensor([arr_one, arr_one, arr_one, arr_one])
        tensor2 = FTensor([[10], [11], [12], [13]])
        XCTAssert((tensor1 + tensor2) == FTensor(arr_test1), "add matrix with column vector dims 2")
        XCTAssert((tensor2 + tensor1) == FTensor(arr_test1), "add column vector dims 2 with matrix")

        tensor1 = FTensor([arr_one, arr_one, arr_one, arr_one])
        tensor2 = FTensor([[5, 4, 3, 2, 1]])
        XCTAssert((tensor1 + tensor2) == FTensor([arr_test2, arr_test2, arr_test2, arr_test2]), "add a matrix with a row vector dims 2")
        XCTAssert((tensor2 + tensor1) == FTensor([arr_test2, arr_test2, arr_test2, arr_test2]), "add a row vector dims 2 with a matrix")

        tensor1 = FTensor([[arr_one, arr_one, arr_one, arr_one], [arr_one, arr_one, arr_one, arr_one]])
        tensor2 = FTensor([10, 11, 12, 13], type: .column)
        let arr_test13 = [arr_test1, arr_test1]
        XCTAssert((tensor1 + tensor2) == FTensor(arr_test13), "add a 3D-Tensor with a column vector")
        XCTAssert((tensor2 + tensor1) == FTensor(arr_test13), "add a column vector with a 3D-Tensor")

        tensor1 = FTensor([[arr_one, arr_one, arr_one, arr_one], [arr_one, arr_one, arr_one, arr_one]])
        tensor2 = FTensor([5, 4, 3, 2, 1], type: .row)
        let arr_test23 = [arr_test2, arr_test2, arr_test2, arr_test2]
        XCTAssert((tensor1 + tensor2) == FTensor([arr_test23, arr_test23]), "add a 3D-Tensor with a row vector")
        XCTAssert((tensor2 + tensor1) == FTensor([arr_test23, arr_test23]), "add a row vector with a 3D-Tensor")

        tensor1 = FTensor([[arr_one, arr_one, arr_one, arr_one], [arr_one, arr_one, arr_one, arr_one]])
        tensor2 = FTensor([[10], [11], [12], [13]])
        XCTAssert((tensor1 + tensor2) == FTensor(arr_test13), "add a 3D-Tensor with a column vector dims 2")
        XCTAssert((tensor2 + tensor1) == FTensor(arr_test13), "add a column vector dims 2 with a 3D-Tensor")

        tensor1 = FTensor([[arr_one, arr_one, arr_one, arr_one], [arr_one, arr_one, arr_one, arr_one]])
        tensor2 = FTensor([[5, 4, 3, 2, 1]])
        XCTAssert((tensor1 + tensor2) == FTensor([arr_test23, arr_test23]), "add a 3D-Tensor with a row vector dims 2")
        XCTAssert((tensor2 + tensor1) == FTensor([arr_test23, arr_test23]), "add a row vector dims 2 with a 3D-Tensor")

        let arr_test4: [Float] = [1, 3, 5]
        tensor1 = FTensor([[arr_test4, arr_test4], [arr_test4, arr_test4]])
        tensor2 = FTensor([[5, 3, 1], [5, 3, 1]])
        let arr_test41: [Float] = [6, 6, 6]
        XCTAssert((tensor1 + tensor2) == FTensor([[arr_test41, arr_test41], [arr_test41, arr_test41]]), "add a 3D-Tensor with a matrix")
        XCTAssert((tensor2 + tensor1) == FTensor([[arr_test41, arr_test41], [arr_test41, arr_test41]]), "add a matrix with a 3D-Tensor")

        // Extra shape tensors
        tensor1 = FTensor(shape: [1, 1, 2], grid: [1, 2])
        tensor2 = FTensor(shape: [1, 2], grid: [1, 2])
        XCTAssert((tensor1 + tensor1) == FTensor(shape: [1, 1, 2], grid: [2, 4]), "add a extra shape row vector with itself")
        XCTAssert((tensor1 + tensor2) == FTensor(shape: [1, 1, 2], grid: [2, 4]), "add a extra shape row vector with a row vector")

        tensor1 = FTensor(shape: [1, 2, 1], grid: [1, 2])
        tensor2 = FTensor(shape: [2, 1], grid: [1, 2])
        XCTAssert((tensor1 + tensor1) == FTensor(shape: [1, 2, 1], grid: [2, 4]), "add a extra shape row vector with itself")
        XCTAssert((tensor1 + tensor2) == FTensor(shape: [1, 2, 1], grid: [2, 4]), "add a extra shape row vector with a column vector")

        tensor1 = FTensor(shape: [1, 1, 2, 1], grid: [1, 2])
        tensor2 = FTensor(shape: [1, 2, 1], grid: [1, 2])
        XCTAssert((tensor1 + tensor1) == FTensor(shape: [1, 1, 2, 1], grid: [2, 4]), "add a extra shape row vector with itself")
        XCTAssert((tensor1 + tensor2) == FTensor(shape: [1, 1, 2, 1], grid: [2, 4]), "add a extra shape row vector with a column vector")

        tensor1 = FTensor(shape: [1, 3, 2, 1], grid: [1, 2, 3, 4, 5, 6])
        tensor2 = FTensor(shape: [3, 2, 1], grid: [1, 2, 3, 4, 5, 6])
        XCTAssert((tensor1 + tensor1) == FTensor(shape: [1, 3, 2, 1], grid: [2, 4, 6, 8, 10, 12]), "add a extra shape 3D tensor with itself")
        XCTAssert((tensor1 + tensor2) == FTensor(shape: [1, 3, 2, 1], grid: [2, 4, 6, 8, 10, 12]), "add a extra shape 3D tensor with a 3D tensor")
    }

    func testSub() throws {
        var tensor1: FTensor
        var tensor2: FTensor

        tensor1 = FTensor(shape: [3, 5, 6], repeating: 25)
        tensor2 = FTensor(shape: [3, 5, 6], repeating: 10)
        XCTAssert((tensor1 - tensor2) == FTensor(shape: [3, 5, 6], repeating: 15), "sub 3D-Tensor with a 3D-Tensor")

        tensor1 = FTensor(shape: [3, 5, 6], repeating: 25)
        tensor2 = FTensor(shape: [5, 6], repeating: 10)
        XCTAssert((tensor1 - tensor2) == FTensor(shape: [3, 5, 6], repeating: 15), "sub 3D-Tensor with a matrix")
        XCTAssert((tensor2 - tensor1) == FTensor(shape: [3, 5, 6], repeating: -15), "sub a matrix with a 3D-Tensor")
    }

    func testMult() throws {
        var tensor1: FTensor
        var tensor2: FTensor

        tensor1 = FTensor(shape: [3, 5, 6], repeating: 25)
        tensor2 = FTensor(shape: [3, 5, 6], repeating: 10)
        XCTAssert((tensor1 * tensor2) == FTensor(shape: [3, 5, 6], repeating: 250), "mult 3D-Tensor with a 3D-Tensor")
        XCTAssert((tensor2 * tensor1) == FTensor(shape: [3, 5, 6], repeating: 250), "mult 3D-Tensor with a 3D-Tensor")

        tensor1 = FTensor(shape: [3, 5, 6], repeating: 25)
        tensor2 = FTensor(shape: [5, 6], repeating: 10)
        XCTAssert((tensor1 * tensor2) == FTensor(shape: [3, 5, 6], repeating: 250), "mult 3D-Tensor with a matrix")
        XCTAssert((tensor2 * tensor1) == FTensor(shape: [3, 5, 6], repeating: 250), "mult a matrix with a 3D-Tensor")
    }

    func testDiv() throws {
        var tensor1: FTensor
        var tensor2: FTensor
        var tensor3: FTensor

        tensor1 = FTensor(shape: [3, 2, 4], repeating: 10)
        tensor2 = FTensor(shape: [3, 2, 4], repeating: 2)
        XCTAssert(((tensor1 / tensor2) - FTensor(shape: [3, 2, 4], repeating: 5)).grid.allSatisfy({ abs($0) < 0.0001 }), "div 3D-Tensor with a 3D-Tensor")
        XCTAssert(((tensor2 / tensor1) - FTensor(shape: [3, 2, 4], repeating: 0.2)).grid.allSatisfy({ abs($0) < 0.0001 }), "div 3D-Tensor with a 3D-Tensor")

        tensor1 = FTensor(shape: [3, 2, 4], repeating: 10)
        tensor2 = FTensor(shape: [2, 4], repeating: 2)
        XCTAssert(((tensor1 / tensor2) - FTensor(shape: [3, 2, 4], repeating: 5)).grid.allSatisfy({ abs($0) < 0.0001 }), "div 3D-Tensor with a 3D-Tensor")
        XCTAssert(((tensor2 / tensor1) - FTensor(shape: [3, 2, 4], repeating: 0.2)).grid.allSatisfy({ abs($0) < 0.0001 }), "div 3D-Tensor with a 3D-Tensor")

        tensor1 = FTensor(shape: [3, 2, 4], repeating: 100)
        tensor2 = FTensor(shape: [2, 4], repeating: 10)
        tensor3 = FTensor(shape: [2, 4], repeating: 2)
        XCTAssert(((tensor1 / tensor2 / tensor3) - FTensor(shape: [3, 2, 4], repeating: 5)).grid.allSatisfy({ abs($0) < 0.0001 }), "div 3D-Tensor with a 3D-Tensor")
        XCTAssert(((tensor3 / tensor2 / tensor1) - FTensor(shape: [3, 2, 4], repeating: 0.002)).grid.allSatisfy({ abs($0) < 0.0001 }), "div 3D-Tensor with a 3D-Tensor")
    }

    func testMatMul() throws {
        var tensor1: FTensor
        var tensor2: FTensor

        tensor1 = FTensor([[1.0, 2], [3.0, 4], [5.0, 6], [7.0, 8]])
        tensor2 = FTensor([[1.0, 2, 3.0, 4], [5.0, 6, 7.0, 8]])
        let ans1: [[Float]] = [
            [11, 14, 17, 20],
            [23, 30, 37, 44],
            [35, 46, 57, 68],
            [47, 62, 77, 92]
        ]
        XCTAssert((tensor1 <*> tensor2) == FTensor(ans1), "mat mul 4x2 with 2x4")

        tensor1 = FTensor([[1.0, 2], [3.0, 4], [5.0, 6], [7.0, 8]])
        tensor2 = FTensor([[1], [2]])
        let ans2: [[Float]] = [
            [5],
            [11],
            [17],
            [23]
        ]
        XCTAssert((tensor1 <*> tensor2) == FTensor(ans2), "mat mul 4x2 with 2x1")
        tensor2 = FTensor([1,2], type: .column)
        XCTAssert((tensor1 <*> tensor2) == FTensor(ans2), "mat mul 4x2 with 2x1")

        tensor1 = FTensor([1, 2, 3, 4], type: .column)
        tensor2 = FTensor([1, 2, 3, 4], type: .row)
        let ans3: [[Float]] = [
            [1, 2, 3, 4],
            [2, 4, 6, 8],
            [3, 6, 9, 12],
            [4, 8, 12, 16]
        ]
        let ans4: Float = 30
        XCTAssert((tensor1 <*> tensor2) == FTensor(ans3), "mat mul 4x1 with 1x4")
        XCTAssert((tensor2 <*> tensor1) == FTensor(shape: [1, 1], grid: [ans4]), "mat mul 1x4 with 4x1")

        tensor1 = FTensor(shape: [1, 4, 1], grid: [1, 2, 3, 4])
        tensor2 = FTensor([1, 2, 3, 4], type: .row)
        XCTAssert((tensor1 <*> tensor2) == FTensor(shape: [1, 4, 4], grid: ans3.flatMap({ $0 })), "mat mul 4x1 with 1x4")
        XCTAssert((tensor2 <*> tensor1) == FTensor(shape: [1, 1, 1], grid: [ans4]), "mat mul 1x4 with 4x1")
    }

    func testSumDiag() throws {
        var tensor1: FTensor

        let grid1: [[Float]] = [
            [1, 2, 3, 4],
            [2, 4, 6, 8],
            [3, 6, 9, 12],
            [4, 8, 12, 16]
        ]
        tensor1 = FTensor(grid1)
        XCTAssert(tensor1.sumDiag() == 30, "sum mat diag")
        XCTAssert(tensor1.diag() == FTensor(shape: tensor1.shape, grid: [1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 9, 0, 0, 0, 0, 16]))
        let grid2: [[Float]] = [
            [1, 2, 3, 4],
            [2, 4, 6, 8],
            [3, 6, 9, 12],
            [4, 8, 12, 16],
            [5, 10, 15, 20]
        ]
        tensor1 = FTensor(grid2)
        XCTAssert(tensor1.sumDiag() == 30, "sum mat diag")
        XCTAssert(tensor1.diag() == FTensor(shape: tensor1.shape, grid: [1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 9, 0, 0, 0, 0, 16, 0, 0, 0, 0]))
        let grid3: [[Float]] = [
            [1, 2, 3, 4],
            [2, 4, 6, 8],
            [3, 6, 9, 12]
        ]
        tensor1 = FTensor(grid3)
        XCTAssert(tensor1.sumDiag() == 14, "sum mat diag")
        XCTAssert(tensor1.diag() == FTensor(shape: tensor1.shape, grid: [1.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0]))
    }

    func testSumAxis() throws {
        var tensor1: FTensor

        tensor1 = FTensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        XCTAssert(tensor1.sum(axis: 0, keepDim: true) == FTensor([[2, 3, 4, 5]]), "sum axis 0 matrix")
        XCTAssert(tensor1.sum(axis: 0) == FTensor(shape: [4], grid: [2, 3, 4, 5]), "sum axis 0 matrix")

        XCTAssert(tensor1.sum(axis: 1, keepDim: true) == FTensor([[10], [4]]), "sum axis 1 matrix")
        XCTAssert(tensor1.sum(axis: 1) == FTensor(shape: [2], grid: [10, 4]), "sum axis 1 matrix")
        
        tensor1 = FTensor([[[1, 1, 1], [-1, -1, -1], [1, 1, 1]], [[2, 2, 2], [-2, -2, -2], [2, 2, 2]]])
        XCTAssert(tensor1.sum(axis: 0, keepDim: true) == FTensor(shape: [1, 3, 3], grid: [3.0, 3.0, 3.0, -3.0, -3.0, -3.0, 3.0, 3.0, 3.0]), "sum axis 0 3D-Tensor")
        XCTAssert(tensor1.sum(axis: 0) == FTensor(shape: [3, 3], grid: [3.0, 3.0, 3.0, -3.0, -3.0, -3.0, 3.0, 3.0, 3.0]), "sum axis 0 3D-Tensor")
        
        XCTAssert(tensor1.sum(axis: 1, keepDim: true) == FTensor(shape: [2, 1, 3], grid: [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]), "sum axis 1 3D-Tensor")
        XCTAssert(tensor1.sum(axis: 1) == FTensor(shape: [2, 3], grid: [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]), "sum axis 1 3D-Tensor")
        
        XCTAssert(tensor1.sum(axis: 2, keepDim: true) == FTensor(shape: [2, 3, 1], grid: [3.0, -3.0, 3.0, 6.0, -6.0, 6.0]), "sum axis 2 3D-Tensor")
        XCTAssert(tensor1.sum(axis: 2) == FTensor(shape: [2, 3], grid: [3.0, -3.0, 3.0, 6.0, -6.0, 6.0]), "sum axis 2 3D-Tensor")
        
        XCTAssert(tensor1.sum(axes: 1...2, keepDim: true) == FTensor(shape: [2, 1, 1], grid: [3.0, 6.0]), "sum axes 1 & 2 3D-Tensor")
        XCTAssert(tensor1.sum(axes: 1...2) == FTensor(shape: [2], grid: [3.0, 6.0]), "sum axes 1 & 2 3D-Tensor")
        XCTAssert(tensor1.sum(axes: 1...2, keepDim: true) == tensor1.sum(axis: 1, keepDim: true).sum(axis: 2, keepDim: true), "sum axes 1 & 2 3D-Tensor")
        
        XCTAssert(tensor1.sum(axes: 0...1, keepDim: true) == FTensor(shape: [1, 1, 3], grid: [3.0, 3.0, 3.0]), "sum axes 0 & 1 3D-Tensor")
        XCTAssert(tensor1.sum(axes: 0...1) == FTensor(shape: [3], grid: [3.0, 3.0, 3.0]), "sum axes 0 & 1 3D-Tensor")
        XCTAssert(tensor1.sum(axes: 0...1, keepDim: true) == tensor1.sum(axis: 0, keepDim: true).sum(axis: 1, keepDim: true), "sum axes 0 & 1 3D-Tensor")
        
        let cube: [[[Float]]] = [[[1, 1, 1], [-1, -1, -1], [1, 1, 1]], [[2, 2, 2], [-2, -2, -2], [2, 2, 2]]]
        tensor1 = FTensor([cube, cube.map { $0.map { $0.map { $0 * -1.0 } } }])
        XCTAssert(tensor1.sum(axis: 0, keepDim: true) == FTensor(shape: [1, 2, 3, 3], grid: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "sum axis 0 4D-Tensor")
        XCTAssert(tensor1.sum(axis: 0) == FTensor(shape: [2, 3, 3], grid: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "sum axis 0 4D-Tensor")
        
        XCTAssert(tensor1.sum(axis: 1, keepDim: true) == FTensor(shape: [2, 1, 3, 3], grid: [3.0, 3.0, 3.0, -3.0, -3.0, -3.0, 3.0, 3.0, 3.0, -3.0, -3.0, -3.0, 3.0, 3.0, 3.0, -3.0, -3.0, -3.0]), "sum axis 1 4D-Tensor")
        XCTAssert(tensor1.sum(axis: 1) == FTensor(shape: [2, 3, 3], grid: [3.0, 3.0, 3.0, -3.0, -3.0, -3.0, 3.0, 3.0, 3.0, -3.0, -3.0, -3.0, 3.0, 3.0, 3.0, -3.0, -3.0, -3.0]), "sum axis 1 4D-Tensor")
        
        XCTAssert(tensor1.sum(axis: 2, keepDim: true) == FTensor(shape: [2, 2, 1, 3], grid: [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, -1.0, -1.0, -1.0, -2.0, -2.0, -2.0]), "sum axis 2 4D-Tensor")
        XCTAssert(tensor1.sum(axis: 2) == FTensor(shape: [2, 2, 3], grid: [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, -1.0, -1.0, -1.0, -2.0, -2.0, -2.0]), "sum axis 2 4D-Tensor")
        
        XCTAssert(tensor1.sum(axis: 3, keepDim: true) == FTensor(shape: [2, 2, 3, 1], grid: [3.0, -3.0, 3.0, 6.0, -6.0, 6.0, -3.0, 3.0, -3.0, -6.0, 6.0, -6.0]), "sum axis 3 4D-Tensor")
        XCTAssert(tensor1.sum(axis: 3) == FTensor(shape: [2, 2, 3], grid: [3.0, -3.0, 3.0, 6.0, -6.0, 6.0, -3.0, 3.0, -3.0, -6.0, 6.0, -6.0]), "sum axis 3 4D-Tensor")
    }

    func testTranspose() throws {
        var tensor1: FTensor

        tensor1 = FTensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        XCTAssert(tensor1.transpose() == FTensor([[1, 1], [2, 1], [3, 1], [4, 1]]), "transpose matrix")

        tensor1 = FTensor([[1, 2, 3, 4]])
        XCTAssert(tensor1.transpose() == FTensor([[1], [2], [3], [4]]), "transpose matrix")
    }

    func testConvolve2DAndPad() throws {
        var image: FTensor
        var kernel: FTensor

        image = FTensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        kernel = FTensor([[1, 2, 1], [2, 1, 2], [1, 2, 1]])

        var t = image.conv2D(with: kernel)
        XCTAssert(t == FTensor(shape: [5, 5], grid: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "convolve2D plain vDSP")

        t = image.conv2D_valid(with: kernel)
        XCTAssert(t == FTensor(shape: [3, 3], grid: [26.0, 39.0, 52.0, 26.0, 39.0, 52.0, 26.0, 39.0, 52.0]), "convolve2D valid")

        image = FTensor(shape: [4, 4], repeating: 4)
        t = image.conv2D_same(with: kernel)
        XCTAssert(t == FTensor([[24.0, 36.0, 36.0, 24.0], [36.0, 52.0, 52.0, 36.0], [36.0, 52.0, 52.0, 36.0], [24.0, 36.0, 36.0, 24.0]]), "convolve2D full")

        t = image.conv2D_full(with: kernel)
        XCTAssert(t == FTensor([[4.0, 12.0, 16.0, 16.0, 12.0, 4.0], [12.0, 24.0, 36.0, 36.0, 24.0, 12.0], [16.0, 36.0, 52.0, 52.0, 36.0, 16.0], [16.0, 36.0, 52.0, 52.0, 36.0, 16.0], [12.0, 24.0, 36.0, 36.0, 24.0, 12.0], [4.0, 12.0, 16.0, 16.0, 12.0, 4.0]]))

        // Reset image
        image = FTensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        t = image.conv2D(with: kernel)
        XCTAssert(t == FTensor(shape: [5, 5], grid: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "convolve2D  plain vDSP with extra shape")

        t = t.pad(1, 1)
        XCTAssert(t == FTensor(shape: [7, 7], grid: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "increase padding of image")
        t = t.trim(2, 2)
        XCTAssert(t == FTensor(shape: [3, 3], grid: [26.0, 39.0, 52.0, 26.0, 39.0, 52.0, 26.0, 39.0, 52.0]), "decrease padding of image")

        t = t.pad(1, 2)
        XCTAssert(t == FTensor(shape: [5, 7], grid: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "increase padding to non symmetrical image")
        t = t.trim(1, 2)
        XCTAssert(t == FTensor(shape: [3, 3], grid: [26.0, 39.0, 52.0, 26.0, 39.0, 52.0, 26.0, 39.0, 52.0]), "decrease padding of non symmetrical image")
    }

    func testMaxPool() throws {
        var tensor: FTensor
        tensor = FTensor(shape: [4, 3], grid: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13])

        var (res, positions) = tensor.pool2D_max(size: 1)
        XCTAssert(res == FTensor(shape: [4, 3], grid: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]), "Max pools correctly")
        for (idx, pos) in positions.enumerated() {
            XCTAssert(tensor.grid[pos] == res.grid[idx], "Max pool calculates max positions wrt to main tensor correctly")
        }

        (res, positions) = tensor.pool2D_max(size: 2)
        XCTAssert(res == FTensor(shape: [3, 2], grid: [5, 6, 8, 9, 12, 13]), "Max pools correctly")
        for (idx, pos) in positions.enumerated() {
            XCTAssert(tensor.grid[pos] == res.grid[idx], "Max pool calculates max positions wrt to main tensor correctly")
        }

        (res, positions) = tensor.pool2D_max(size: 3)
        XCTAssert(res == FTensor(shape: [2, 1], grid: [9, 13]), "Max pools correctly")
        for (idx, pos) in positions.enumerated() {
            XCTAssert(tensor.grid[pos] == res.grid[idx], "Max pool calculates max positions wrt to main tensor correctly")
        }
    }

//    func testError() throws {
//        let sef = Tensor(shape: [10, 10], repeating: 1.0)
//        let kernel = Tensor(shape: [4, 4], repeating: 3.0)
//        let options = XCTMeasureOptions()
//        options.iterationCount = 1000
//        measure(options: options) {
//            sef.conv2D_mine(with: kernel)
//        }
//        print(sef.conv2D(with: kernel, type: .valid).as2D())
//    }

    func testRot180() throws {
        var tensor: FTensor

        tensor = FTensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        XCTAssert(tensor.rot180() == FTensor([[3, 3, 3], [2, 2, 2], [1, 1, 1]]), "rot 180")

        tensor = FTensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    }

    func testArrayView() throws {
        var tensor: FTensor

        tensor = FTensor(shape: [2, 4, 5, 6, 5], repeating: 0.0)
        XCTAssert(tensor.shape.main[0] == 2, "index for array view")
        XCTAssert(tensor.shape.main[1] == 4, "index for array view")
        XCTAssert(tensor.shape.main[2] == 5, "index for array view")
        XCTAssert(Array(tensor.shape.main[1..<3]) == [4, 5], "range for array view")
        XCTAssert(Array(tensor.shape.main[1...3]) == [4, 5, 6], "range for array view")
        XCTAssert(Array(tensor.shape.main[1..<5]) == [4, 5, 6, 5], "range for array view")

        tensor = FTensor(shape: [1, 2, 4, 5, 6, 5], repeating: 0.0)
        XCTAssert(tensor.shape.main[0] == 2, "index for array view with extra shape")
        XCTAssert(tensor.shape.main[1] == 4, "index for array view with extra shape")
        XCTAssert(tensor.shape.main[2] == 5, "index for array view with extra shape")
        XCTAssert(Array(tensor.shape.main[1..<3]) == [4, 5], "range for array view with extra shape")
        XCTAssert(Array(tensor.shape.main[1...3]) == [4, 5, 6], "range for array view with extra shape")
        XCTAssert(Array(tensor.shape.main[1..<5]) == [4, 5, 6, 5], "range for array view with extra shape")
    }

//    func testImgFir() throws {
//        let image = Tensor(shape: [34, 34], repeating: 3)
//        let kernel = Tensor(shape: [24, 24], repeating: 10)
//
//        var res = image.conv2D_valid(with: kernel)
//        print(res.shape)
//        for i in stride(from: 0, to: res.grid.count, by: res.shape[1]) {
//            print(res.grid[i..<i + res.shape[1]])
//        }
//        res = image.conv2D_mine(with: kernel)
//        print(res.shape)
//        for i in stride(from: 0, to: res.grid.count, by: res.shape[1]) {
//            print(res.grid[i..<i + res.shape[1]])
//        }
//    }

//    func testZscoreImage() throws {
//        var tensor: Tensor
//
//        tensor = Tensor(shape: [2, 1, 3, 3], grid: [1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6, 7, 8, 9, 10])
////        print(tensor[t3D: 0])
//        let (norm, mean, std) = tensor.zscore_image()
//        print(norm, (tensor[t3D: 0] - mean) / std, (tensor[t3D: 1] - mean) / std)
//    }

    func testType() throws {
        var tensor1: FTensor

        tensor1 = FTensor(1)
        XCTAssert(tensor1.type == .scalar)

        tensor1 = FTensor(shape: [1, 1, 1, 1], grid: [1])
        XCTAssert(tensor1.type == .scalar)

        tensor1 = FTensor([1, 2, 3, 4])
        XCTAssert(tensor1.type == .row)

        tensor1 = FTensor(shape: [1, 1, 1, 4], grid: [1, 2, 3, 4])
        XCTAssert(tensor1.type == .row)

        tensor1 = FTensor([1, 2, 3, 4], type: .column)
        XCTAssert(tensor1.type == .column)

        tensor1 = FTensor(shape: [1, 1, 4, 1], grid: [1, 2, 3, 4])
        XCTAssert(tensor1.type == .column)

        tensor1 = FTensor([1, 2, 3, 4], type: .row)
        XCTAssert(tensor1.type == .row)

        tensor1 = FTensor(shape: [1, 1, 1, 4], grid: [1, 2, 3, 4])
        XCTAssert(tensor1.type == .row)

        tensor1 = FTensor([[1, 2, 3, 4], [1, 2, 3, 4]])
        XCTAssert(tensor1.type == .matrix)

        tensor1 = FTensor(shape: [1, 1, 2, 4], grid: [1, 2, 3, 4, 1, 2, 3, 4])
        XCTAssert(tensor1.type == .matrix)

        tensor1 = FTensor([[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]])
        XCTAssert(tensor1.type == .tensor3D)

        tensor1 = FTensor(shape: [1, 1, 1, 2, 2, 4], grid: [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
        XCTAssert(tensor1.type == .tensor3D)

        tensor1 = FTensor(shape: [1, 1, 2, 2, 2, 4], grid: [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
        XCTAssert(tensor1.type == .tensor4D)

        tensor1 = FTensor(shape: [1, 1, 2, 2, 1, 4], grid: [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
        XCTAssert(tensor1.type == .tensor4D)

        tensor1 = FTensor(shape: [1, 2, 2, 2, 1, 4], grid: [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
        XCTAssert(tensor1.type == .tensorND)
    }
}
