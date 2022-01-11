//
//  SML2TensorTests.swift
//  
//
//  Created by Sahil Srivastava on 12/4/21.
//

import XCTest
@testable import SML2

final class SML2DTensorTests: XCTestCase {
    
    func testInits() throws {
        var tensor: DTensor

        tensor = DTensor(shape: [2, 4], grid: Array(repeating: 1.0, count: 2 * 4))
        XCTAssert(tensor.shape == [2, 4] && tensor.grid.count == 8 && tensor.grid.allSatisfy({ $0 == 1 }), "init")
        tensor = DTensor(shape: [2, 4], repeating: 10)
        XCTAssert(tensor.shape == [2, 4] && tensor.grid.count == 8 && tensor.grid.allSatisfy({ $0 == 10 }), "init repeating")

        let val = 12.0
        tensor = DTensor(val)
        XCTAssert(tensor.shape == [] && tensor.grid == [12.0], "init double")
        let arr_one: [Double] = [1, 2, 3, 4, 5]
        tensor = DTensor(arr_one, type: .row)
        XCTAssert(tensor.shape == [1, 5] && tensor.grid == arr_one, "init 1D arr to row vec")
        tensor = DTensor(arr_one, type: .column)
        XCTAssert(tensor.shape == [5, 1] && tensor.grid == arr_one, "init 1D arr to column vec")
        let arr_two: [[Double]] = [arr_one, arr_one, arr_one]
        tensor = DTensor(arr_two)
        XCTAssert(tensor.shape == [3, 5] && tensor.grid == arr_two.flatMap({ $0 }), "init 2D arr to mat")
        let arr_three: [[[Double]]] = [arr_two, arr_two]
        tensor = DTensor(arr_three)
        XCTAssert(tensor.shape == [2, 3, 5] && tensor.grid == arr_three.flatMap({ $0.flatMap({ $0 })}), "init 3D arr to mats")
    }

    func testQuery() throws {
        var tensor: DTensor

        let arr_one: [Double] = [1, 2, 3, 4, 5]
        tensor = DTensor(arr_one)
        XCTAssert(tensor[1] == 2.0, "query get row vec")
        tensor[4] = -1
        XCTAssert(tensor[4] == -1, "query set row vec")
        tensor = DTensor(arr_one, type: .row)
        XCTAssert(tensor[0, 1] == 2.0, "query get row vec")
        tensor[0, 4] = -1
        XCTAssert(tensor[0, 4] == -1, "query set row vec")
        tensor = DTensor(arr_one, type: .column)
        XCTAssert(tensor[1, 0] == 2.0, "query get col vec")
        tensor[4, 0] = -1
        XCTAssert(tensor[4, 0] == -1, "query set col vec")

        tensor = DTensor(arr_one, type: .row)
        XCTAssert(tensor[val: 1] == 2.0, "query get row vec")
        tensor[val: 4] = -1
        XCTAssert(tensor[val: 4] == -1, "query set row vec")
        tensor = DTensor(arr_one, type: .column)
        XCTAssert(tensor[val: 1] == 2.0, "query get col vec")
        tensor[val: 4] = -1
        XCTAssert(tensor[val: 4] == -1, "query set col vec")

        let arr_two: [[Double]] = [arr_one, [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
        tensor = DTensor(arr_two)
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

        let arr_three: [[[Double]]] = [arr_two, arr_two.map { $0.map { 15.0 + $0 } }]
        tensor = DTensor(arr_three)
        XCTAssert(tensor[1, 0, 3] == 19.0 && tensor[1, 2, 4] == 30.0, "query get mats")
        tensor[1, 1, 3] = -1
        XCTAssert(tensor[1, 1, 3] == -1, "query set mats")

        tensor = DTensor(arr_two)
        XCTAssert(tensor[col: 1].grid == [2, 7, 12], "query col of matrix")
        tensor[col: 1] = DTensor([1, 1, 1], type: .column)
        XCTAssert(tensor[col: 1].grid == [1, 1, 1], "query set col of matrix")

        tensor = DTensor(arr_two)
        XCTAssert(tensor[row: 1].grid == [6, 7, 8, 9, 10], "query row of matrix")
        tensor[row: 1] = DTensor([1, 1, 1, 1, 1], type: .row)
        XCTAssert(tensor[row: 1].grid == [1, 1, 1, 1, 1], "query set row of matrix")

        tensor = DTensor(arr_three)
        XCTAssert(tensor[mat: 0] == DTensor(arr_two), "query matrix of 3D-Tensor")
        tensor[mat: 1] = DTensor(arr_two)
        XCTAssert(tensor[mat: 1] == DTensor(arr_two), "query set matrix of 3D-Tensor")

        let arr_three_1 = arr_three.map { $0.map { $0.map { -1.35 + $0 } } }
        let arr_four = [arr_three, arr_three_1]
        tensor = DTensor(arr_four)
        XCTAssert(tensor[t3D: 1] == DTensor(arr_three_1), "query 3D-Tensor of 4D-Tensor")
        tensor[t3D: 1] = DTensor(arr_three.map { $0.map { $0.map { -1.95 + $0 } } })
        XCTAssert(tensor[t3D: 1] == DTensor(arr_three.map { $0.map { $0.map { -1.95 + $0 } } }), "query set 3D-Tensor of 4D-Tensor")
    }

    func testQueryRange() throws {
        var tensor: DTensor
        tensor = DTensor(shape: [4, 3], grid: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13])

        // Column ranges
        XCTAssert(tensor[cols: 1..<3] == DTensor(shape: [4, 2], grid: [2, 3, 5, 6, 8, 9, 12, 13]), "query columns")
        XCTAssert(tensor[cols: 0..<2] == DTensor(shape: [4, 2], grid: [1, 2, 4, 5, 7, 8, 11, 12]), "query columns")
        XCTAssert(tensor[cols: 1..<2] == DTensor(shape: [4, 1], grid: [2, 5, 8, 12]), "query columns")

        // Row ranges
        XCTAssert(tensor[rows: 1..<3] == DTensor(shape: [2, 3], grid: [4, 5, 6, 7, 8, 9]), "query rows")
        XCTAssert(tensor[rows: 0..<2] == DTensor(shape: [2, 3], grid: [1, 2, 3, 4, 5, 6]), "query rows")
        XCTAssert(tensor[rows: 1..<2] == DTensor(shape: [1, 3], grid: [4, 5, 6]), "query rows")

        // Mat ranges
        var t3D = [Double]()
        for i in 1...60 {
            t3D.append(Double(i))
        }
        tensor = DTensor(shape: [5, 4, 3], grid: t3D)
        XCTAssert(tensor[mats: 0..<2] == DTensor(shape: [2, 4, 3], grid: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]), "query mats")
        XCTAssert(tensor[mats: 2..<3] == DTensor(shape: [1, 4, 3], grid: [25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0]), "query mats")
        XCTAssert(tensor[mats: 3..<5] == DTensor(shape: [2, 4, 3], grid: [37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0]), "query mats")

        // t3D ranges
        var t4D = [Double]()
        for i in 1...600 {
            t4D.append(Double(i))
        }
        tensor = DTensor(shape: [10, 5, 4, 3], grid: t4D)
        XCTAssert(tensor[t3Ds: 0..<2] == DTensor(shape: [2, 5, 4, 3], grid: Array(t4D[60 * 0..<60 * 2])), "query t3D")
        XCTAssert(tensor[t3Ds: 2..<5] == DTensor(shape: [3, 5, 4, 3], grid: Array(t4D[60 * 2..<60 * 5])), "query t3D")
        XCTAssert(tensor[t3Ds: 9..<10] == DTensor(shape: [1, 5, 4, 3], grid: Array(t4D[60 * 9..<60 * 10])), "query t3D")
    }

    func testRandomInit() throws {
        var tensor: DTensor

        tensor = DTensor.random(shape: [2, 3], min: -10, max: 25)
        XCTAssert(tensor.grid.count == 6, "random init")
        tensor = DTensor.random_xavier(shape: [2, 3], ni: -4, no: 10)
        XCTAssert(tensor.grid.count == 6, "random init")
    }

    // HAVE NOT TESTED INVERT AND TRANSPOSE

    func testAdd() throws {
        var tensor1: DTensor
        var tensor2: DTensor

        tensor1 = DTensor(3)
        tensor2 = DTensor(4)
        XCTAssert((tensor1 + tensor2) == DTensor(7), "add double + double")
        XCTAssert((tensor1 + tensor2).shape == tensor2.shape, "add double + double check shape")
        XCTAssert((tensor2 + tensor1).shape == tensor2.shape, "add double + double check shape")

        var val: Double = 1
        tensor1 = DTensor(val)
        let arr_one: [Double] = [1, 2, 3, 4, 5]
        tensor2 = DTensor(arr_one, type: .row)
        XCTAssert((tensor1 + tensor2) == DTensor(arr_one.map { $0 + val }, type: .row), "add double + row vector")
        XCTAssert((tensor2 + tensor1) == DTensor(arr_one.map { $0 + val }, type: .row), "add row vector + double")
        XCTAssert((tensor1 + tensor2).shape == tensor2.shape, "add double + row vector check shape")
        XCTAssert((tensor2 + tensor1).shape == tensor2.shape, "add row vector + double check shape")

        tensor1 = DTensor(val)
        tensor2 = DTensor(arr_one, type: .column)
        XCTAssert((tensor1 + tensor2) == DTensor(arr_one.map { $0 + val }, type: .column), "add double + column vector")
        XCTAssert((tensor2 + tensor1) == DTensor(arr_one.map { $0 + val }, type: .column), "add column vector + double")
        XCTAssert((tensor1 + tensor2).shape == tensor2.shape, "add double + column vector check shape")
        XCTAssert((tensor2 + tensor1).shape == tensor2.shape, "add column vector + double check shape")

        val = 3
        tensor1 = DTensor(val)
        let arr_two: [[Double]] = [arr_one, arr_one.map {$0 + 5}]
        tensor2 = DTensor(arr_two)
        XCTAssert((tensor1 + tensor2) == DTensor(arr_two.map { $0.map { $0 + val } }), "add double + matrix")
        XCTAssert((tensor2 + tensor1) == DTensor(arr_two.map { $0.map { $0 + val } }), "add matrix + double")
        XCTAssert((tensor1 + tensor2).shape == tensor2.shape, "add double + matrix check shape")
        XCTAssert((tensor2 + tensor1).shape == tensor2.shape, "add matrix + double check shape")

        tensor1 = DTensor(arr_one, type: .row)
        tensor2 = DTensor(arr_one, type: .row)
        XCTAssert((tensor1 + tensor2) == DTensor(arr_one.map { $0 + $0 }, type: .row), "add vec of same size + vec of same size")
        XCTAssert((tensor2 + tensor1) == DTensor(arr_one.map { $0 + $0 }, type: .row), "add vec of same size + vec of same size")
        XCTAssert((tensor1 + tensor2).shape == tensor2.shape, "add vec + vec check shape")
        XCTAssert((tensor2 + tensor1).shape == tensor2.shape, "add vec + vec check shape")

        tensor1 = DTensor(arr_two)
        tensor2 = DTensor(arr_two)
        XCTAssert((tensor1 + tensor2) == DTensor(arr_two.map { $0.map { $0 + $0 } }), "add matrix of same size + matrix of same size")
        XCTAssert((tensor2 + tensor1) == DTensor(arr_two.map { $0.map { $0 + $0 } }), "add matrix of same size + matrix of same size")
        XCTAssert((tensor1 + tensor2).shape == tensor2.shape, "add matrix + matrix check shape")
        XCTAssert((tensor2 + tensor1).shape == tensor2.shape, "add matrix + matrix check shape")

        tensor1 = DTensor([arr_one, arr_one, arr_one, arr_one])
        tensor2 = DTensor([10, 11, 12, 13], type: .column)
        let arr_test1 = [arr_one.map { $0 + 10 }, arr_one.map { $0 + 11 }, arr_one.map { $0 + 12 }, arr_one.map { $0 + 13 }]
        XCTAssert((tensor1 + tensor2) == DTensor(arr_test1), "add matrix with column vector")
        XCTAssert((tensor2 + tensor1) == DTensor(arr_test1), "add column vector with matrix")

        tensor1 = DTensor([arr_one, arr_one, arr_one, arr_one])
        tensor2 = DTensor([5, 4, 3, 2, 1], type: .row)
        let arr_test2: [Double] = [6, 6, 6, 6, 6]
        XCTAssert((tensor1 + tensor2) == DTensor([arr_test2, arr_test2, arr_test2, arr_test2]), "add a matrix with a row vector")
        XCTAssert((tensor2 + tensor1) == DTensor([arr_test2, arr_test2, arr_test2, arr_test2]), "add a row vector with a matrix")

        tensor1 = DTensor([arr_one, arr_one, arr_one, arr_one])
        tensor2 = DTensor([[10], [11], [12], [13]])
        XCTAssert((tensor1 + tensor2) == DTensor(arr_test1), "add matrix with column vector dims 2")
        XCTAssert((tensor2 + tensor1) == DTensor(arr_test1), "add column vector dims 2 with matrix")

        tensor1 = DTensor([arr_one, arr_one, arr_one, arr_one])
        tensor2 = DTensor([[5, 4, 3, 2, 1]])
        XCTAssert((tensor1 + tensor2) == DTensor([arr_test2, arr_test2, arr_test2, arr_test2]), "add a matrix with a row vector dims 2")
        XCTAssert((tensor2 + tensor1) == DTensor([arr_test2, arr_test2, arr_test2, arr_test2]), "add a row vector dims 2 with a matrix")

        tensor1 = DTensor([[arr_one, arr_one, arr_one, arr_one], [arr_one, arr_one, arr_one, arr_one]])
        tensor2 = DTensor([10, 11, 12, 13], type: .column)
        let arr_test13 = [arr_test1, arr_test1]
        XCTAssert((tensor1 + tensor2) == DTensor(arr_test13), "add a 3D-Tensor with a column vector")
        XCTAssert((tensor2 + tensor1) == DTensor(arr_test13), "add a column vector with a 3D-Tensor")

        tensor1 = DTensor([[arr_one, arr_one, arr_one, arr_one], [arr_one, arr_one, arr_one, arr_one]])
        tensor2 = DTensor([5, 4, 3, 2, 1], type: .row)
        let arr_test23 = [arr_test2, arr_test2, arr_test2, arr_test2]
        XCTAssert((tensor1 + tensor2) == DTensor([arr_test23, arr_test23]), "add a 3D-Tensor with a row vector")
        XCTAssert((tensor2 + tensor1) == DTensor([arr_test23, arr_test23]), "add a row vector with a 3D-Tensor")

        tensor1 = DTensor([[arr_one, arr_one, arr_one, arr_one], [arr_one, arr_one, arr_one, arr_one]])
        tensor2 = DTensor([[10], [11], [12], [13]])
        XCTAssert((tensor1 + tensor2) == DTensor(arr_test13), "add a 3D-Tensor with a column vector dims 2")
        XCTAssert((tensor2 + tensor1) == DTensor(arr_test13), "add a column vector dims 2 with a 3D-Tensor")

        tensor1 = DTensor([[arr_one, arr_one, arr_one, arr_one], [arr_one, arr_one, arr_one, arr_one]])
        tensor2 = DTensor([[5, 4, 3, 2, 1]])
        XCTAssert((tensor1 + tensor2) == DTensor([arr_test23, arr_test23]), "add a 3D-Tensor with a row vector dims 2")
        XCTAssert((tensor2 + tensor1) == DTensor([arr_test23, arr_test23]), "add a row vector dims 2 with a 3D-Tensor")

        let arr_test4: [Double] = [1, 3, 5]
        tensor1 = DTensor([[arr_test4, arr_test4], [arr_test4, arr_test4]])
        tensor2 = DTensor([[5, 3, 1], [5, 3, 1]])
        let arr_test41: [Double] = [6, 6, 6]
        XCTAssert((tensor1 + tensor2) == DTensor([[arr_test41, arr_test41], [arr_test41, arr_test41]]), "add a 3D-Tensor with a matrix")
        XCTAssert((tensor2 + tensor1) == DTensor([[arr_test41, arr_test41], [arr_test41, arr_test41]]), "add a matrix with a 3D-Tensor")

        // Extra shape tensors
        tensor1 = DTensor(shape: [1, 1, 2], grid: [1, 2])
        tensor2 = DTensor(shape: [1, 2], grid: [1, 2])
        XCTAssert((tensor1 + tensor1) == DTensor(shape: [1, 1, 2], grid: [2, 4]), "add a extra shape row vector with itself")
        XCTAssert((tensor1 + tensor2) == DTensor(shape: [1, 1, 2], grid: [2, 4]), "add a extra shape row vector with a row vector")

        tensor1 = DTensor(shape: [1, 2, 1], grid: [1, 2])
        tensor2 = DTensor(shape: [2, 1], grid: [1, 2])
        XCTAssert((tensor1 + tensor1) == DTensor(shape: [1, 2, 1], grid: [2, 4]), "add a extra shape row vector with itself")
        XCTAssert((tensor1 + tensor2) == DTensor(shape: [1, 2, 1], grid: [2, 4]), "add a extra shape row vector with a column vector")

        tensor1 = DTensor(shape: [1, 1, 2, 1], grid: [1, 2])
        tensor2 = DTensor(shape: [1, 2, 1], grid: [1, 2])
        XCTAssert((tensor1 + tensor1) == DTensor(shape: [1, 1, 2, 1], grid: [2, 4]), "add a extra shape row vector with itself")
        XCTAssert((tensor1 + tensor2) == DTensor(shape: [1, 1, 2, 1], grid: [2, 4]), "add a extra shape row vector with a column vector")

        tensor1 = DTensor(shape: [1, 3, 2, 1], grid: [1, 2, 3, 4, 5, 6])
        tensor2 = DTensor(shape: [3, 2, 1], grid: [1, 2, 3, 4, 5, 6])
        XCTAssert((tensor1 + tensor1) == DTensor(shape: [1, 3, 2, 1], grid: [2, 4, 6, 8, 10, 12]), "add a extra shape 3D tensor with itself")
        XCTAssert((tensor1 + tensor2) == DTensor(shape: [1, 3, 2, 1], grid: [2, 4, 6, 8, 10, 12]), "add a extra shape 3D tensor with a 3D tensor")
    }

    func testSub() throws {
        var tensor1: DTensor
        var tensor2: DTensor

        tensor1 = DTensor(shape: [3, 5, 6], repeating: 25)
        tensor2 = DTensor(shape: [3, 5, 6], repeating: 10)
        XCTAssert((tensor1 - tensor2) == DTensor(shape: [3, 5, 6], repeating: 15), "sub 3D-Tensor with a 3D-Tensor")

        tensor1 = DTensor(shape: [3, 5, 6], repeating: 25)
        tensor2 = DTensor(shape: [5, 6], repeating: 10)
        XCTAssert((tensor1 - tensor2) == DTensor(shape: [3, 5, 6], repeating: 15), "sub 3D-Tensor with a matrix")
        XCTAssert((tensor2 - tensor1) == DTensor(shape: [3, 5, 6], repeating: -15), "sub a matrix with a 3D-Tensor")
    }

    func testMult() throws {
        var tensor1: DTensor
        var tensor2: DTensor

        tensor1 = DTensor(shape: [3, 5, 6], repeating: 25)
        tensor2 = DTensor(shape: [3, 5, 6], repeating: 10)
        XCTAssert((tensor1 * tensor2) == DTensor(shape: [3, 5, 6], repeating: 250), "mult 3D-Tensor with a 3D-Tensor")
        XCTAssert((tensor2 * tensor1) == DTensor(shape: [3, 5, 6], repeating: 250), "mult 3D-Tensor with a 3D-Tensor")

        tensor1 = DTensor(shape: [3, 5, 6], repeating: 25)
        tensor2 = DTensor(shape: [5, 6], repeating: 10)
        XCTAssert((tensor1 * tensor2) == DTensor(shape: [3, 5, 6], repeating: 250), "mult 3D-Tensor with a matrix")
        XCTAssert((tensor2 * tensor1) == DTensor(shape: [3, 5, 6], repeating: 250), "mult a matrix with a 3D-Tensor")
    }

    func testDiv() throws {
        var tensor1: DTensor
        var tensor2: DTensor
        var tensor3: DTensor

        tensor1 = DTensor(shape: [3, 2, 4], repeating: 10)
        tensor2 = DTensor(shape: [3, 2, 4], repeating: 2)
        XCTAssert((tensor1 / tensor2) == DTensor(shape: [3, 2, 4], repeating: 5), "div 3D-Tensor with a 3D-Tensor")
        XCTAssert((tensor2 / tensor1) == DTensor(shape: [3, 2, 4], repeating: 0.2), "div 3D-Tensor with a 3D-Tensor")

        tensor1 = DTensor(shape: [3, 2, 4], repeating: 10)
        tensor2 = DTensor(shape: [2, 4], repeating: 2)
        XCTAssert((tensor1 / tensor2) == DTensor(shape: [3, 2, 4], repeating: 5), "div 3D-Tensor with a 3D-Tensor")
        XCTAssert((tensor2 / tensor1) == DTensor(shape: [3, 2, 4], repeating: 0.2), "div 3D-Tensor with a 3D-Tensor")

        tensor1 = DTensor(shape: [3, 2, 4], repeating: 100)
        tensor2 = DTensor(shape: [2, 4], repeating: 10)
        tensor3 = DTensor(shape: [2, 4], repeating: 2)
        XCTAssert((tensor1 / tensor2 / tensor3) == DTensor(shape: [3, 2, 4], repeating: 5), "div 3D-Tensor with a 3D-Tensor")
        XCTAssert((tensor3 / tensor2 / tensor1) == DTensor(shape: [3, 2, 4], repeating: 0.002), "div 3D-Tensor with a 3D-Tensor")
    }

    func testMatMul() throws {
        var tensor1: DTensor
        var tensor2: DTensor

        tensor1 = DTensor([[1.0, 2], [3.0, 4], [5.0, 6], [7.0, 8]])
        tensor2 = DTensor([[1.0, 2, 3.0, 4], [5.0, 6, 7.0, 8]])
        let ans1: [[Double]] = [
            [11, 14, 17, 20],
            [23, 30, 37, 44],
            [35, 46, 57, 68],
            [47, 62, 77, 92]
        ]
        XCTAssert((tensor1 <*> tensor2) == DTensor(ans1), "mat mul 4x2 with 2x4")

        tensor1 = DTensor([[1.0, 2], [3.0, 4], [5.0, 6], [7.0, 8]])
        tensor2 = DTensor([[1], [2]])
        let ans2: [[Double]] = [
            [5],
            [11],
            [17],
            [23]
        ]
        XCTAssert((tensor1 <*> tensor2) == DTensor(ans2), "mat mul 4x2 with 2x1")
        tensor2 = DTensor([1,2], type: .column)
        XCTAssert((tensor1 <*> tensor2) == DTensor(ans2), "mat mul 4x2 with 2x1")

        tensor1 = DTensor([1, 2, 3, 4], type: .column)
        tensor2 = DTensor([1, 2, 3, 4], type: .row)
        let ans3: [[Double]] = [
            [1, 2, 3, 4],
            [2, 4, 6, 8],
            [3, 6, 9, 12],
            [4, 8, 12, 16]
        ]
        let ans4: Double = 30
        XCTAssert((tensor1 <*> tensor2) == DTensor(ans3), "mat mul 4x1 with 1x4")
        XCTAssert((tensor2 <*> tensor1) == DTensor(shape: [1, 1], grid: [ans4]), "mat mul 1x4 with 4x1")

        tensor1 = DTensor(shape: [1, 4, 1], grid: [1, 2, 3, 4])
        tensor2 = DTensor([1, 2, 3, 4], type: .row)
        XCTAssert((tensor1 <*> tensor2) == DTensor(shape: [1, 4, 4], grid: ans3.flatMap({ $0 })), "mat mul 4x1 with 1x4")
        XCTAssert((tensor2 <*> tensor1) == DTensor(shape: [1, 1, 1], grid: [ans4]), "mat mul 1x4 with 4x1")
    }

    func testSumDiag() throws {
        var tensor1: DTensor

        let grid1: [[Double]] = [
            [1, 2, 3, 4],
            [2, 4, 6, 8],
            [3, 6, 9, 12],
            [4, 8, 12, 16]
        ]
        tensor1 = DTensor(grid1)
        XCTAssert(tensor1.sumDiag() == 30, "sum mat diag")
        XCTAssert(tensor1.diag() == DTensor(shape: tensor1.shape, grid: [1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 9, 0, 0, 0, 0, 16]))
        let grid2: [[Double]] = [
            [1, 2, 3, 4],
            [2, 4, 6, 8],
            [3, 6, 9, 12],
            [4, 8, 12, 16],
            [5, 10, 15, 20]
        ]
        tensor1 = DTensor(grid2)
        XCTAssert(tensor1.sumDiag() == 30, "sum mat diag")
        XCTAssert(tensor1.diag() == DTensor(shape: tensor1.shape, grid: [1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 9, 0, 0, 0, 0, 16, 0, 0, 0, 0]))
        let grid3: [[Double]] = [
            [1, 2, 3, 4],
            [2, 4, 6, 8],
            [3, 6, 9, 12]
        ]
        tensor1 = DTensor(grid3)
        XCTAssert(tensor1.sumDiag() == 14, "sum mat diag")
        XCTAssert(tensor1.diag() == DTensor(shape: tensor1.shape, grid: [1.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0]))
    }

    func testSumAxis() throws {
        var tensor1: DTensor

        tensor1 = DTensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        XCTAssert(tensor1.sum(axis: 0, keepDim: true) == DTensor([[2, 3, 4, 5]]), "sum axis 0")
        XCTAssert(tensor1.sum(axis: 0) == DTensor(shape: [4], grid: [2, 3, 4, 5]), "sum axis 0")

        XCTAssert(tensor1.sum(axis: 1, keepDim: true) == DTensor([[10], [4]]), "sum axis 0")
        XCTAssert(tensor1.sum(axis: 1) == DTensor(shape: [2], grid: [10, 4]), "sum axis 0")
    }

    func testTranspose() throws {
        var tensor1: DTensor

        tensor1 = DTensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        XCTAssert(tensor1.transpose() == DTensor([[1, 1], [2, 1], [3, 1], [4, 1]]), "transpose matrix")

        tensor1 = DTensor([[1, 2, 3, 4]])
        XCTAssert(tensor1.transpose() == DTensor([[1], [2], [3], [4]]), "transpose matrix")
    }

    func testConvolve2DAndPad() throws {
        var image: DTensor
        var kernel: DTensor

        image = DTensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        kernel = DTensor([[1, 2, 1], [2, 1, 2], [1, 2, 1]])

        var t = image.conv2D(with: kernel)
        XCTAssert(t == DTensor(shape: [5, 5], grid: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "convolve2D plain vDSP")

        t = image.conv2D_valid(with: kernel)
        XCTAssert(t == DTensor(shape: [3, 3], grid: [26.0, 39.0, 52.0, 26.0, 39.0, 52.0, 26.0, 39.0, 52.0]), "convolve2D valid")

        image = DTensor(shape: [4, 4], repeating: 4)
        t = image.conv2D_same(with: kernel)
        XCTAssert(t == DTensor([[24.0, 36.0, 36.0, 24.0], [36.0, 52.0, 52.0, 36.0], [36.0, 52.0, 52.0, 36.0], [24.0, 36.0, 36.0, 24.0]]), "convolve2D full")

        t = image.conv2D_full(with: kernel)
        XCTAssert(t == DTensor([[4.0, 12.0, 16.0, 16.0, 12.0, 4.0], [12.0, 24.0, 36.0, 36.0, 24.0, 12.0], [16.0, 36.0, 52.0, 52.0, 36.0, 16.0], [16.0, 36.0, 52.0, 52.0, 36.0, 16.0], [12.0, 24.0, 36.0, 36.0, 24.0, 12.0], [4.0, 12.0, 16.0, 16.0, 12.0, 4.0]]))

        // Reset image
        image = DTensor([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
        t = image.conv2D(with: kernel)
        XCTAssert(t == DTensor(shape: [5, 5], grid: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "convolve2D  plain vDSP with extra shape")

        t = t.pad(1, 1)
        XCTAssert(t == DTensor(shape: [7, 7], grid: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "increase padding of image")
        t = t.trim(2, 2)
        XCTAssert(t == DTensor(shape: [3, 3], grid: [26.0, 39.0, 52.0, 26.0, 39.0, 52.0, 26.0, 39.0, 52.0]), "decrease padding of image")

        t = t.pad(1, 2)
        XCTAssert(t == DTensor(shape: [5, 7], grid: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 0.0, 0.0, 26.0, 39.0, 52.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), "increase padding to non symmetrical image")
        t = t.trim(1, 2)
        XCTAssert(t == DTensor(shape: [3, 3], grid: [26.0, 39.0, 52.0, 26.0, 39.0, 52.0, 26.0, 39.0, 52.0]), "decrease padding of non symmetrical image")
    }

    func testMaxPool() throws {
        var tensor: DTensor
        tensor = DTensor(shape: [4, 3], grid: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13])

        var (res, positions) = tensor.pool2D_max(size: 1)
        XCTAssert(res == DTensor(shape: [4, 3], grid: [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13]), "Max pools correctly")
        for (idx, pos) in positions.enumerated() {
            XCTAssert(tensor.grid[pos] == res.grid[idx], "Max pool calculates max positions wrt to main tensor correctly")
        }

        (res, positions) = tensor.pool2D_max(size: 2)
        XCTAssert(res == DTensor(shape: [3, 2], grid: [5, 6, 8, 9, 12, 13]), "Max pools correctly")
        for (idx, pos) in positions.enumerated() {
            XCTAssert(tensor.grid[pos] == res.grid[idx], "Max pool calculates max positions wrt to main tensor correctly")
        }

        (res, positions) = tensor.pool2D_max(size: 3)
        XCTAssert(res == DTensor(shape: [2, 1], grid: [9, 13]), "Max pools correctly")
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
        var tensor: DTensor

        tensor = DTensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        XCTAssert(tensor.rot180() == DTensor([[3, 3, 3], [2, 2, 2], [1, 1, 1]]), "rot 180")

        tensor = DTensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    }

    func testArrayView() throws {
        var tensor: DTensor

        tensor = DTensor(shape: [2, 4, 5, 6, 5], repeating: 0.0)
        XCTAssert(tensor.shape.main[0] == 2, "index for array view")
        XCTAssert(tensor.shape.main[1] == 4, "index for array view")
        XCTAssert(tensor.shape.main[2] == 5, "index for array view")
        XCTAssert(Array(tensor.shape.main[1..<3]) == [4, 5], "range for array view")
        XCTAssert(Array(tensor.shape.main[1...3]) == [4, 5, 6], "range for array view")
        XCTAssert(Array(tensor.shape.main[1..<5]) == [4, 5, 6, 5], "range for array view")

        tensor = DTensor(shape: [1, 2, 4, 5, 6, 5], repeating: 0.0)
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
        var tensor1: DTensor

        tensor1 = DTensor(1)
        XCTAssert(tensor1.type == .scalar)

        tensor1 = DTensor(shape: [1, 1, 1, 1], grid: [1])
        XCTAssert(tensor1.type == .scalar)

        tensor1 = DTensor([1, 2, 3, 4])
        XCTAssert(tensor1.type == .row)

        tensor1 = DTensor(shape: [1, 1, 1, 4], grid: [1, 2, 3, 4])
        XCTAssert(tensor1.type == .row)

        tensor1 = DTensor([1, 2, 3, 4], type: .column)
        XCTAssert(tensor1.type == .column)

        tensor1 = DTensor(shape: [1, 1, 4, 1], grid: [1, 2, 3, 4])
        XCTAssert(tensor1.type == .column)

        tensor1 = DTensor([1, 2, 3, 4], type: .row)
        XCTAssert(tensor1.type == .row)

        tensor1 = DTensor(shape: [1, 1, 1, 4], grid: [1, 2, 3, 4])
        XCTAssert(tensor1.type == .row)

        tensor1 = DTensor([[1, 2, 3, 4], [1, 2, 3, 4]])
        XCTAssert(tensor1.type == .matrix)

        tensor1 = DTensor(shape: [1, 1, 2, 4], grid: [1, 2, 3, 4, 1, 2, 3, 4])
        XCTAssert(tensor1.type == .matrix)

        tensor1 = DTensor([[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]])
        XCTAssert(tensor1.type == .tensor3D)

        tensor1 = DTensor(shape: [1, 1, 1, 2, 2, 4], grid: [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
        XCTAssert(tensor1.type == .tensor3D)

        tensor1 = DTensor(shape: [1, 1, 2, 2, 2, 4], grid: [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
        XCTAssert(tensor1.type == .tensor4D)

        tensor1 = DTensor(shape: [1, 1, 2, 2, 1, 4], grid: [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
        XCTAssert(tensor1.type == .tensor4D)

        tensor1 = DTensor(shape: [1, 2, 2, 2, 1, 4], grid: [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
        XCTAssert(tensor1.type == .tensorND)
    }
}
