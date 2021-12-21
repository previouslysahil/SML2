//
//  SML2TensorTests.swift
//  
//
//  Created by Sahil Srivastava on 12/4/21.
//

import XCTest
@testable import SML2

final class SML2TensorTests: XCTestCase {
    
    func testInits() throws {
        var tensor: Tensor
        
        tensor = Tensor(shape: [2, 4], grid: Array(repeating: 1.0, count: 2 * 4))
        XCTAssert(tensor.shape == [2, 4] && tensor.grid.count == 8 && tensor.grid.allSatisfy({ $0 == 1 }), "init")
        tensor = Tensor(shape: [2, 4], repeating: 10)
        XCTAssert(tensor.shape == [2, 4] && tensor.grid.count == 8 && tensor.grid.allSatisfy({ $0 == 10 }), "init repeating")
        
        let val = 12.0
        tensor = Tensor(val)
        XCTAssert(tensor.shape == [] && tensor.grid == [12.0], "init double")
        let arr_one: [Double] = [1, 2, 3, 4, 5]
        tensor = Tensor(arr_one, type: .row)
        XCTAssert(tensor.shape == [1, 5] && tensor.grid == arr_one, "init 1D arr to row vec")
        tensor = Tensor(arr_one, type: .column)
        XCTAssert(tensor.shape == [5, 1] && tensor.grid == arr_one, "init 1D arr to column vec")
        let arr_two: [[Double]] = [arr_one, arr_one, arr_one]
        tensor = Tensor(arr_two)
        XCTAssert(tensor.shape == [3, 5] && tensor.grid == arr_two.flatMap({ $0 }), "init 2D arr to mat")
        let arr_three: [[[Double]]] = [arr_two, arr_two]
        tensor = Tensor(arr_three)
        XCTAssert(tensor.shape == [2, 3, 5] && tensor.grid == arr_three.flatMap({ $0.flatMap({ $0 })}), "init 3D arr to mats")
    }
    
    func testQuery() throws {
        var tensor: Tensor
        
        let arr_one: [Double] = [1, 2, 3, 4, 5]
        tensor = Tensor(arr_one)
        XCTAssert(tensor[1] == 2.0, "query get row vec")
        tensor[4] = -1
        XCTAssert(tensor[4] == -1, "query set row vec")
        tensor = Tensor(arr_one, type: .row)
        XCTAssert(tensor[0, 1] == 2.0, "query get row vec")
        tensor[0, 4] = -1
        XCTAssert(tensor[0, 4] == -1, "query set row vec")
        tensor = Tensor(arr_one, type: .column)
        XCTAssert(tensor[1, 0] == 2.0, "query get col vec")
        tensor[4, 0] = -1
        XCTAssert(tensor[4, 0] == -1, "query set col vec")
        
        tensor = Tensor(arr_one, type: .row)
        XCTAssert(tensor[val: 1] == 2.0, "query get row vec")
        tensor[val: 4] = -1
        XCTAssert(tensor[val: 4] == -1, "query set row vec")
        tensor = Tensor(arr_one, type: .column)
        XCTAssert(tensor[val: 1] == 2.0, "query get col vec")
        tensor[val: 4] = -1
        XCTAssert(tensor[val: 4] == -1, "query set col vec")
        
        let arr_two: [[Double]] = [arr_one, [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]
        tensor = Tensor(arr_two)
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
        tensor = Tensor(arr_three)
        XCTAssert(tensor[1, 0, 3] == 19.0 && tensor[1, 2, 4] == 30.0, "query get mats")
        tensor[1, 1, 3] = -1
        XCTAssert(tensor[1, 1, 3] == -1, "query set mats")
        
        tensor = Tensor(arr_two)
        XCTAssert(tensor[col: 1].grid == [2, 7, 12], "query col of matrix")
        tensor[col: 1] = Tensor([1, 1, 1], type: .column)
        XCTAssert(tensor[col: 1].grid == [1, 1, 1], "query set col of matrix")
        
        tensor = Tensor(arr_two)
        XCTAssert(tensor[row: 1].grid == [6, 7, 8, 9, 10], "query row of matrix")
        tensor[row: 1] = Tensor([1, 1, 1, 1, 1], type: .row)
        XCTAssert(tensor[row: 1].grid == [1, 1, 1, 1, 1], "query set row of matrix")
        
        tensor = Tensor(arr_three)
        XCTAssert(tensor[mat: 0] == Tensor(arr_two), "query matrix of 3D-Tensor")
        tensor[mat: 1] = Tensor(arr_two)
        XCTAssert(tensor[mat: 1] == Tensor(arr_two), "query set matrix of 3D-Tensor")
    }
    
    func testRandomInit() throws {
        var tensor: Tensor
        
        tensor = Tensor.random(shape: [2, 3], min: -10, max: 25)
        XCTAssert(tensor.grid.count == 6, "random init")
        tensor = Tensor.random_xavier(shape: [2, 3], ni: -4, no: 10)
        XCTAssert(tensor.grid.count == 6, "random init")
    }
    
    // HAVE NOT TESTED INVERT AND TRANSPOSE
    
    func testAdd() throws {
        var tensor1: Tensor
        var tensor2: Tensor

        tensor1 = Tensor(3)
        tensor2 = Tensor(4)
        XCTAssert((tensor1 + tensor2) == Tensor(7), "add double + double")
        XCTAssert((tensor1 + tensor2).shape == tensor2.shape, "add double + double check shape")
        XCTAssert((tensor2 + tensor1).shape == tensor2.shape, "add double + double check shape")

        var val: Double = 1
        tensor1 = Tensor(val)
        let arr_one: [Double] = [1, 2, 3, 4, 5]
        tensor2 = Tensor(arr_one, type: .row)
        XCTAssert((tensor1 + tensor2) == Tensor(arr_one.map { $0 + val }, type: .row), "add double + row vector")
        XCTAssert((tensor2 + tensor1) == Tensor(arr_one.map { $0 + val }, type: .row), "add row vector + double")
        XCTAssert((tensor1 + tensor2).shape == tensor2.shape, "add double + row vector check shape")
        XCTAssert((tensor2 + tensor1).shape == tensor2.shape, "add row vector + double check shape")
        
        tensor1 = Tensor(val)
        tensor2 = Tensor(arr_one, type: .column)
        XCTAssert((tensor1 + tensor2) == Tensor(arr_one.map { $0 + val }, type: .column), "add double + column vector")
        XCTAssert((tensor2 + tensor1) == Tensor(arr_one.map { $0 + val }, type: .column), "add column vector + double")
        XCTAssert((tensor1 + tensor2).shape == tensor2.shape, "add double + column vector check shape")
        XCTAssert((tensor2 + tensor1).shape == tensor2.shape, "add column vector + double check shape")

        val = 3
        tensor1 = Tensor(val)
        let arr_two: [[Double]] = [arr_one, arr_one.map {$0 + 5}]
        tensor2 = Tensor(arr_two)
        XCTAssert((tensor1 + tensor2) == Tensor(arr_two.map { $0.map { $0 + val } }), "add double + matrix")
        XCTAssert((tensor2 + tensor1) == Tensor(arr_two.map { $0.map { $0 + val } }), "add matrix + double")
        XCTAssert((tensor1 + tensor2).shape == tensor2.shape, "add double + matrix check shape")
        XCTAssert((tensor2 + tensor1).shape == tensor2.shape, "add matrix + double check shape")

        tensor1 = Tensor(arr_one, type: .row)
        tensor2 = Tensor(arr_one, type: .row)
        XCTAssert((tensor1 + tensor2) == Tensor(arr_one.map { $0 + $0 }, type: .row), "add vec of same size + vec of same size")
        XCTAssert((tensor2 + tensor1) == Tensor(arr_one.map { $0 + $0 }, type: .row), "add vec of same size + vec of same size")
        XCTAssert((tensor1 + tensor2).shape == tensor2.shape, "add vec + vec check shape")
        XCTAssert((tensor2 + tensor1).shape == tensor2.shape, "add vec + vec check shape")

        tensor1 = Tensor(arr_two)
        tensor2 = Tensor(arr_two)
        XCTAssert((tensor1 + tensor2) == Tensor(arr_two.map { $0.map { $0 + $0 } }), "add matrix of same size + matrix of same size")
        XCTAssert((tensor2 + tensor1) == Tensor(arr_two.map { $0.map { $0 + $0 } }), "add matrix of same size + matrix of same size")
        XCTAssert((tensor1 + tensor2).shape == tensor2.shape, "add matrix + matrix check shape")
        XCTAssert((tensor2 + tensor1).shape == tensor2.shape, "add matrix + matrix check shape")

        tensor1 = Tensor([arr_one, arr_one, arr_one, arr_one])
        tensor2 = Tensor([10, 11, 12, 13], type: .column)
        let arr_test1 = [arr_one.map { $0 + 10 }, arr_one.map { $0 + 11 }, arr_one.map { $0 + 12 }, arr_one.map { $0 + 13 }]
        XCTAssert((tensor1 + tensor2) == Tensor(arr_test1), "add matrix with column vector")
        XCTAssert((tensor2 + tensor1) == Tensor(arr_test1), "add column vector with matrix")

        tensor1 = Tensor([arr_one, arr_one, arr_one, arr_one])
        tensor2 = Tensor([5, 4, 3, 2, 1], type: .row)
        let arr_test2: [Double] = [6, 6, 6, 6, 6]
        XCTAssert((tensor1 + tensor2) == Tensor([arr_test2, arr_test2, arr_test2, arr_test2]), "add a matrix with a row vector")
        XCTAssert((tensor2 + tensor1) == Tensor([arr_test2, arr_test2, arr_test2, arr_test2]), "add a row vector with a matrix")

        tensor1 = Tensor([arr_one, arr_one, arr_one, arr_one])
        tensor2 = Tensor([[10], [11], [12], [13]])
        XCTAssert((tensor1 + tensor2) == Tensor(arr_test1), "add matrix with column vector dims 2")
        XCTAssert((tensor2 + tensor1) == Tensor(arr_test1), "add column vector dims 2 with matrix")

        tensor1 = Tensor([arr_one, arr_one, arr_one, arr_one])
        tensor2 = Tensor([[5, 4, 3, 2, 1]])
        XCTAssert((tensor1 + tensor2) == Tensor([arr_test2, arr_test2, arr_test2, arr_test2]), "add a matrix with a row vector dims 2")
        XCTAssert((tensor2 + tensor1) == Tensor([arr_test2, arr_test2, arr_test2, arr_test2]), "add a row vector dims 2 with a matrix")

        tensor1 = Tensor([[arr_one, arr_one, arr_one, arr_one], [arr_one, arr_one, arr_one, arr_one]])
        tensor2 = Tensor([10, 11, 12, 13], type: .column)
        let arr_test13 = [arr_test1, arr_test1]
        XCTAssert((tensor1 + tensor2) == Tensor(arr_test13), "add a 3D-Tensor with a column vector")
        XCTAssert((tensor2 + tensor1) == Tensor(arr_test13), "add a column vector with a 3D-Tensor")

        tensor1 = Tensor([[arr_one, arr_one, arr_one, arr_one], [arr_one, arr_one, arr_one, arr_one]])
        tensor2 = Tensor([5, 4, 3, 2, 1], type: .row)
        let arr_test23 = [arr_test2, arr_test2, arr_test2, arr_test2]
        XCTAssert((tensor1 + tensor2) == Tensor([arr_test23, arr_test23]), "add a 3D-Tensor with a row vector")
        XCTAssert((tensor2 + tensor1) == Tensor([arr_test23, arr_test23]), "add a row vector with a 3D-Tensor")

        tensor1 = Tensor([[arr_one, arr_one, arr_one, arr_one], [arr_one, arr_one, arr_one, arr_one]])
        tensor2 = Tensor([[10], [11], [12], [13]])
        XCTAssert((tensor1 + tensor2) == Tensor(arr_test13), "add a 3D-Tensor with a column vector dims 2")
        XCTAssert((tensor2 + tensor1) == Tensor(arr_test13), "add a column vector dims 2 with a 3D-Tensor")

        tensor1 = Tensor([[arr_one, arr_one, arr_one, arr_one], [arr_one, arr_one, arr_one, arr_one]])
        tensor2 = Tensor([[5, 4, 3, 2, 1]])
        XCTAssert((tensor1 + tensor2) == Tensor([arr_test23, arr_test23]), "add a 3D-Tensor with a row vector dims 2")
        XCTAssert((tensor2 + tensor1) == Tensor([arr_test23, arr_test23]), "add a row vector dims 2 with a 3D-Tensor")

        let arr_test4: [Double] = [1, 3, 5]
        tensor1 = Tensor([[arr_test4, arr_test4], [arr_test4, arr_test4]])
        tensor2 = Tensor([[5, 3, 1], [5, 3, 1]])
        let arr_test41: [Double] = [6, 6, 6]
        XCTAssert((tensor1 + tensor2) == Tensor([[arr_test41, arr_test41], [arr_test41, arr_test41]]), "add a 3D-Tensor with a matrix")
        XCTAssert((tensor2 + tensor1) == Tensor([[arr_test41, arr_test41], [arr_test41, arr_test41]]), "add a matrix with a 3D-Tensor")
        
        // Extra shape tensors
        tensor1 = Tensor(shape: [1, 1, 2], grid: [1, 2])
        tensor2 = Tensor(shape: [1, 2], grid: [1, 2])
        XCTAssert((tensor1 + tensor1) == Tensor(shape: [1, 1, 2], grid: [2, 4]), "add a extra shape row vector with itself")
        XCTAssert((tensor1 + tensor2) == Tensor(shape: [1, 1, 2], grid: [2, 4]), "add a extra shape row vector with a row vector")

        tensor1 = Tensor(shape: [1, 2, 1], grid: [1, 2])
        tensor2 = Tensor(shape: [2, 1], grid: [1, 2])
        XCTAssert((tensor1 + tensor1) == Tensor(shape: [1, 2, 1], grid: [2, 4]), "add a extra shape row vector with itself")
        XCTAssert((tensor1 + tensor2) == Tensor(shape: [1, 2, 1], grid: [2, 4]), "add a extra shape row vector with a column vector")
        
        tensor1 = Tensor(shape: [1, 1, 2, 1], grid: [1, 2])
        tensor2 = Tensor(shape: [1, 2, 1], grid: [1, 2])
        XCTAssert((tensor1 + tensor1) == Tensor(shape: [1, 1, 2, 1], grid: [2, 4]), "add a extra shape row vector with itself")
        XCTAssert((tensor1 + tensor2) == Tensor(shape: [1, 1, 2, 1], grid: [2, 4]), "add a extra shape row vector with a column vector")
        
        tensor1 = Tensor(shape: [1, 3, 2, 1], grid: [1, 2, 3, 4, 5, 6])
        tensor2 = Tensor(shape: [3, 2, 1], grid: [1, 2, 3, 4, 5, 6])
        XCTAssert((tensor1 + tensor1) == Tensor(shape: [1, 3, 2, 1], grid: [2, 4, 6, 8, 10, 12]), "add a extra shape 3D tensor with itself")
        XCTAssert((tensor1 + tensor2) == Tensor(shape: [1, 3, 2, 1], grid: [2, 4, 6, 8, 10, 12]), "add a extra shape 3D tensor with a 3D tensor")
    }
    
    func testSub() throws {
        var tensor1: Tensor
        var tensor2: Tensor
        
        tensor1 = Tensor(shape: [3, 5, 6], repeating: 25)
        tensor2 = Tensor(shape: [3, 5, 6], repeating: 10)
        XCTAssert((tensor1 - tensor2) == Tensor(shape: [3, 5, 6], repeating: 15), "sub 3D-Tensor with a 3D-Tensor")
        
        tensor1 = Tensor(shape: [3, 5, 6], repeating: 25)
        tensor2 = Tensor(shape: [5, 6], repeating: 10)
        XCTAssert((tensor1 - tensor2) == Tensor(shape: [3, 5, 6], repeating: 15), "sub 3D-Tensor with a matrix")
        XCTAssert((tensor2 - tensor1) == Tensor(shape: [3, 5, 6], repeating: -15), "sub a matrix with a 3D-Tensor")
    }
    
    func testMult() throws {
        var tensor1: Tensor
        var tensor2: Tensor
        
        tensor1 = Tensor(shape: [3, 5, 6], repeating: 25)
        tensor2 = Tensor(shape: [3, 5, 6], repeating: 10)
        XCTAssert((tensor1 * tensor2) == Tensor(shape: [3, 5, 6], repeating: 250), "mult 3D-Tensor with a 3D-Tensor")
        XCTAssert((tensor2 * tensor1) == Tensor(shape: [3, 5, 6], repeating: 250), "mult 3D-Tensor with a 3D-Tensor")
        
        tensor1 = Tensor(shape: [3, 5, 6], repeating: 25)
        tensor2 = Tensor(shape: [5, 6], repeating: 10)
        XCTAssert((tensor1 * tensor2) == Tensor(shape: [3, 5, 6], repeating: 250), "mult 3D-Tensor with a matrix")
        XCTAssert((tensor2 * tensor1) == Tensor(shape: [3, 5, 6], repeating: 250), "mult a matrix with a 3D-Tensor")
    }
    
    func testDiv() throws {
        var tensor1: Tensor
        var tensor2: Tensor
        var tensor3: Tensor
        
        tensor1 = Tensor(shape: [3, 2, 4], repeating: 10)
        tensor2 = Tensor(shape: [3, 2, 4], repeating: 2)
        XCTAssert((tensor1 / tensor2) == Tensor(shape: [3, 2, 4], repeating: 5), "div 3D-Tensor with a 3D-Tensor")
        XCTAssert((tensor2 / tensor1) == Tensor(shape: [3, 2, 4], repeating: 0.2), "div 3D-Tensor with a 3D-Tensor")
        
        tensor1 = Tensor(shape: [3, 2, 4], repeating: 10)
        tensor2 = Tensor(shape: [2, 4], repeating: 2)
        XCTAssert((tensor1 / tensor2) == Tensor(shape: [3, 2, 4], repeating: 5), "div 3D-Tensor with a 3D-Tensor")
        XCTAssert((tensor2 / tensor1) == Tensor(shape: [3, 2, 4], repeating: 0.2), "div 3D-Tensor with a 3D-Tensor")
        
        tensor1 = Tensor(shape: [3, 2, 4], repeating: 100)
        tensor2 = Tensor(shape: [2, 4], repeating: 10)
        tensor3 = Tensor(shape: [2, 4], repeating: 2)
        XCTAssert((tensor1 / tensor2 / tensor3) == Tensor(shape: [3, 2, 4], repeating: 5), "div 3D-Tensor with a 3D-Tensor")
        XCTAssert((tensor3 / tensor2 / tensor1) == Tensor(shape: [3, 2, 4], repeating: 0.002), "div 3D-Tensor with a 3D-Tensor")
    }
    
    func testMatMul() throws {
        var tensor1: Tensor
        var tensor2: Tensor
        
        tensor1 = Tensor([[1.0, 2], [3.0, 4], [5.0, 6], [7.0, 8]])
        tensor2 = Tensor([[1.0, 2, 3.0, 4], [5.0, 6, 7.0, 8]])
        let ans1: [[Double]] = [
            [11, 14, 17, 20],
            [23, 30, 37, 44],
            [35, 46, 57, 68],
            [47, 62, 77, 92]
        ]
        XCTAssert((tensor1 <*> tensor2) == Tensor(ans1), "mat mul 4x2 with 2x4")
        
        tensor1 = Tensor([[1.0, 2], [3.0, 4], [5.0, 6], [7.0, 8]])
        tensor2 = Tensor([[1], [2]])
        let ans2: [[Double]] = [
            [5],
            [11],
            [17],
            [23]
        ]
        XCTAssert((tensor1 <*> tensor2) == Tensor(ans2), "mat mul 4x2 with 2x1")
        tensor2 = Tensor([1,2], type: .column)
        XCTAssert((tensor1 <*> tensor2) == Tensor(ans2), "mat mul 4x2 with 2x1")
        
        tensor1 = Tensor([1, 2, 3, 4], type: .column)
        tensor2 = Tensor([1, 2, 3, 4], type: .row)
        let ans3: [[Double]] = [
            [1, 2, 3, 4],
            [2, 4, 6, 8],
            [3, 6, 9, 12],
            [4, 8, 12, 16]
        ]
        let ans4: Double = 30
        XCTAssert((tensor1 <*> tensor2) == Tensor(ans3), "mat mul 4x1 with 1x4")
        XCTAssert((tensor2 <*> tensor1) == Tensor(shape: [1, 1], grid: [ans4]), "mat mul 1x4 with 4x1")
        
        tensor1 = Tensor(shape: [1, 4, 1], grid: [1, 2, 3, 4])
        tensor2 = Tensor([1, 2, 3, 4], type: .row)
        XCTAssert((tensor1 <*> tensor2) == Tensor(shape: [1, 4, 4], grid: ans3.flatMap({ $0 })), "mat mul 4x1 with 1x4")
        XCTAssert((tensor2 <*> tensor1) == Tensor(shape: [1, 1, 1], grid: [ans4]), "mat mul 1x4 with 4x1")
    }
    
    func testSumDiag() throws {
        var tensor1: Tensor
        
        let grid1: [[Double]] = [
            [1, 2, 3, 4],
            [2, 4, 6, 8],
            [3, 6, 9, 12],
            [4, 8, 12, 16]
        ]
        tensor1 = Tensor(grid1)
        XCTAssert(tensor1.sumDiag() == 30, "sum mat diag")
        XCTAssert(tensor1.diag() == Tensor(shape: tensor1.shape, grid: [1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 9, 0, 0, 0, 0, 16]))
        let grid2: [[Double]] = [
            [1, 2, 3, 4],
            [2, 4, 6, 8],
            [3, 6, 9, 12],
            [4, 8, 12, 16],
            [5, 10, 15, 20]
        ]
        tensor1 = Tensor(grid2)
        XCTAssert(tensor1.sumDiag() == 30, "sum mat diag")
        XCTAssert(tensor1.diag() == Tensor(shape: tensor1.shape, grid: [1, 0, 0, 0, 0, 4, 0, 0, 0, 0, 9, 0, 0, 0, 0, 16, 0, 0, 0, 0]))
        let grid3: [[Double]] = [
            [1, 2, 3, 4],
            [2, 4, 6, 8],
            [3, 6, 9, 12]
        ]
        tensor1 = Tensor(grid3)
        XCTAssert(tensor1.sumDiag() == 14, "sum mat diag")
        XCTAssert(tensor1.diag() == Tensor(shape: tensor1.shape, grid: [1.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 9.0, 0.0]))
    }
    
    func testSumAxis() throws {
        var tensor1: Tensor
        
        tensor1 = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        XCTAssert(tensor1.sum(axis: 0, keepDim: true) == Tensor([[2, 3, 4, 5]]), "sum axis 0")
        XCTAssert(tensor1.sum(axis: 0) == Tensor(shape: [4], grid: [2, 3, 4, 5]), "sum axis 0")
        
        XCTAssert(tensor1.sum(axis: 1, keepDim: true) == Tensor([[10], [4]]), "sum axis 0")
        XCTAssert(tensor1.sum(axis: 1) == Tensor(shape: [2], grid: [10, 4]), "sum axis 0")
    }
    
    func testTranspose() throws {
        var tensor1: Tensor
        
        tensor1 = Tensor([[1, 2, 3, 4], [1, 1, 1, 1]])
        XCTAssert(tensor1.transpose() == Tensor([[1, 1], [2, 1], [3, 1], [4, 1]]), "transpose matrix")
        
        tensor1 = Tensor([[1, 2, 3, 4]])
        XCTAssert(tensor1.transpose() == Tensor([[1], [2], [3], [4]]), "transpose matrix")
    }
    
    func testType() throws {
        var tensor1: Tensor
        
        tensor1 = Tensor(1)
        XCTAssert(tensor1.type == .scalar)
        
        tensor1 = Tensor(shape: [1, 1, 1, 1], grid: [1])
        XCTAssert(tensor1.type == .scalar)
        
        tensor1 = Tensor([1, 2, 3, 4])
        XCTAssert(tensor1.type == .row)
        
        tensor1 = Tensor(shape: [1, 1, 1, 4], grid: [1, 2, 3, 4])
        XCTAssert(tensor1.type == .row)
        
        tensor1 = Tensor([1, 2, 3, 4], type: .column)
        XCTAssert(tensor1.type == .column)
        
        tensor1 = Tensor(shape: [1, 1, 4, 1], grid: [1, 2, 3, 4])
        XCTAssert(tensor1.type == .column)
        
        tensor1 = Tensor([1, 2, 3, 4], type: .row)
        XCTAssert(tensor1.type == .row)
        
        tensor1 = Tensor(shape: [1, 1, 1, 4], grid: [1, 2, 3, 4])
        XCTAssert(tensor1.type == .row)
        
        tensor1 = Tensor([[1, 2, 3, 4], [1, 2, 3, 4]])
        XCTAssert(tensor1.type == .matrix)
        
        tensor1 = Tensor(shape: [1, 1, 2, 4], grid: [1, 2, 3, 4, 1, 2, 3, 4])
        XCTAssert(tensor1.type == .matrix)
        
        tensor1 = Tensor([[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [1, 2, 3, 4]]])
        XCTAssert(tensor1.type == .tensor3D)
        
        tensor1 = Tensor(shape: [1, 1, 1, 2, 2, 4], grid: [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
        XCTAssert(tensor1.type == .tensor3D)
        
        tensor1 = Tensor(shape: [1, 1, 2, 2, 2, 4], grid: [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
        XCTAssert(tensor1.type == .tensorND)
        
        tensor1 = Tensor(shape: [1, 1, 2, 2, 1, 4], grid: [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
        XCTAssert(tensor1.type == .tensorND)
    }
}
