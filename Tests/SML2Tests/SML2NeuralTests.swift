//
//  SML2NeuralTests.swift
//  
//
//  Created by Sahil Srivastava on 12/13/21.
//

import XCTest
@testable import SML2

final class SML2NeuralTests: XCTestCase {
    
    let parallel = false // usually non-parallel works better, why?
    let eps = 0.0000001
    let bound = 0.0001
    
    func testLinear() throws {
        let dt = DTensor([[1, 2, 3, 4, 5, 6], [12, 23, 34, 45, 56, 67], [13, 24, 35, 46, 57, 68], [19, 28, 37, 46, 55, 64]]).transpose()
//        let d = Variable(dt)
//        let layer = Linear(d, to: 6, out: 4)
        let layer = Linear(to: 6, out: 4)
        let J = layer.square().sum()
        let session = Session(parallel: parallel)
        session.build(J)
        session.pass([layer.input: dt])
        let (out, grads) = session.run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[layer.weight] ?? "NIL", grads[layer.input] ?? "NIL", grads[layer.bias] ?? "NIL")
        
        // NOW LETS TEST IF THIS LAYER IS CORRECT BY DOING IT BY 'HAND'
        // Weights init'd randomly so this makes sure they are the same random vals for both weight matrices
        let wtTest = layer.weight.out!
        let wTest = Variable(wtTest)
        let dtTest = DTensor([[1, 2, 3, 4, 5, 6], [12, 23, 34, 45, 56, 67], [13, 24, 35, 46, 57, 68], [19, 28, 37, 46, 55, 64]]).transpose()
        let dTest = Variable(dtTest)
        let bt = DTensor(shape: [4, 1], repeating: 0.01)
        let bTest = Variable(bt)
        func loss(_ a: Variable, _ b: Variable, _ c: Variable) -> Variable {
            return (a <*> b + c).square().sum()
        }
        let JTest = loss(wTest, dTest, bTest)
        // Forward and backward prop wrt J through session
        let sessionTest = Session(parallel: parallel)
        sessionTest.build(JTest)
        let (outTest, gradsTest) = sessionTest.run(JTest)
//        print("----------\n", outTest)
//        print(gradsTest[JTest] ?? "NIL", gradsTest[wTest] ?? "NIL", gradsTest[dTest] ?? "NIL", gradsTest[bTest] ?? "NIL")
        // Test for correctness
        let epsilon = Variable(eps)
        let grad_wTest_numerical = (loss(wTest + epsilon, dTest, bTest) - loss(wTest - epsilon, dTest, bTest)) / (Variable(2.0) * epsilon)
        let graph2 = CGraph(grad_wTest_numerical, seed: DTensor(1))
        graph2.fwd()
//        print(grad_wTest_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        func get_grad_wTest_numericals(_ pos_wTest: Variable, _ neg_wTest: Variable, idx: Int) {
            let epsilon = Variable(eps)
            let grad_wTest_numerical = (loss(pos_wTest, dTest, bTest) - loss(neg_wTest, dTest, bTest)) / (Variable(2.0) * epsilon)
            let graph2 = CGraph(grad_wTest_numerical, seed: DTensor(1))
            graph2.fwd()
//            print(grad_wTest_numerical.out ?? "NIL")
            let _ = graph2.bwd()

            let diff_wTest = abs(gradsTest[wTest]!.grid[idx] - grad_wTest_numerical.out!.grid.first!) / max(abs(gradsTest[wTest]!.grid[idx]), abs(grad_wTest_numerical.out!.grid.first!))
            XCTAssert(diff_wTest < bound, "analytical vs numerical gradient check for wTest")
        }
        for i in 0..<wtTest.grid.count {
            var pos_grid = wtTest.grid
            pos_grid[i] = pos_grid[i] + eps
            var neg_grid = wtTest.grid
            neg_grid[i] = neg_grid[i] - eps
            let pos_wtTest = Variable(DTensor(shape: wtTest.shape, grid: pos_grid))
            let neg_wtTest = Variable(DTensor(shape: wtTest.shape, grid: neg_grid))
            get_grad_wTest_numericals(pos_wtTest, neg_wtTest, idx: i)
        }
//        print("----------")
        let grad_dTest_numerical = (loss(wTest, dTest + epsilon, bTest) - loss(wTest, dTest - epsilon, bTest)) / (Variable(2.0) * epsilon)
        let graph3 = CGraph(grad_dTest_numerical, seed: DTensor(1))
        graph3.fwd()
//        print(grad_dTest_numerical.out ?? "NIL")
        let _ = graph3.bwd()
        func get_grad_dTest_numericals(_ pos_dTest: Variable, _ neg_dTest: Variable, idx: Int) {
            let epsilon = Variable(eps)
            let grad_dTest_numerical = (loss(wTest, pos_dTest, bTest) - loss(wTest, neg_dTest, bTest)) / (Variable(2.0) * epsilon)
            let graph3 = CGraph(grad_dTest_numerical, seed: DTensor(1))
            graph3.fwd()
//            print(grad_dTest_numerical.out ?? "NIL")
            let _ = graph3.bwd()

            let diff_dTest = abs(gradsTest[dTest]!.grid[idx] - grad_dTest_numerical.out!.grid.first!) / max(abs(gradsTest[dTest]!.grid[idx]), abs(grad_dTest_numerical.out!.grid.first!))
            XCTAssert(diff_dTest < bound, "analytical vs numerical gradient check for dTest")
        }
        for i in 0..<dtTest.grid.count {
            var pos_grid = dtTest.grid
            pos_grid[i] = pos_grid[i] + eps
            var neg_grid = dtTest.grid
            neg_grid[i] = neg_grid[i] - eps
            let pos_dtTest = Variable(DTensor(shape: dtTest.shape, grid: pos_grid))
            let neg_dtTest = Variable(DTensor(shape: dtTest.shape, grid: neg_grid))
            get_grad_dTest_numericals(pos_dtTest, neg_dtTest, idx: i)
        }
//        print("----------")
        let grad_bTest_numerical = (loss(wTest, dTest, bTest + epsilon) - loss(wTest, dTest, bTest - epsilon)) / (Variable(2.0) * epsilon)
        let graph4 = CGraph(grad_bTest_numerical, seed: DTensor(1))
        graph4.fwd()
//        print(grad_bTest_numerical.out ?? "NIL")
        let _ = graph4.bwd()
        func get_grad_bTest_numericals(_ pos_bTest: Variable, _ neg_bTest: Variable, idx: Int) {
            let epsilon = Variable(eps)
            let grad_bTest_numerical = (loss(wTest, dTest, pos_bTest) - loss(wTest, dTest, neg_bTest)) / (Variable(2.0) * epsilon)
            let graph4 = CGraph(grad_bTest_numerical, seed: DTensor(1))
            graph4.fwd()
//            print(grad_bTest_numerical.out ?? "NIL")
            let _ = graph4.bwd()

            let diff_bTest = abs(gradsTest[bTest]!.grid[idx] - grad_bTest_numerical.out!.grid.first!) / max(abs(gradsTest[bTest]!.grid[idx]), abs(grad_bTest_numerical.out!.grid.first!))
            XCTAssert(diff_bTest < bound, "analytical vs numerical gradient check for bTest")
        }
        for i in 0..<bt.grid.count {
            var pos_grid = bt.grid
            pos_grid[i] = pos_grid[i] + eps
            var neg_grid = bt.grid
            neg_grid[i] = neg_grid[i] - eps
            let pos_btTest = Variable(DTensor(shape: bt.shape, grid: pos_grid))
            let neg_btTest = Variable(DTensor(shape: bt.shape, grid: neg_grid))
            get_grad_bTest_numericals(pos_btTest, neg_btTest, idx: i)
        }
        // Makes sure that our hand computed linear results and our linear layers results are identical
        XCTAssert(outTest == out, "hand written linear out vs linear layer out")
        XCTAssert(gradsTest[JTest]! == grads[J]! && gradsTest[wTest]! == grads[layer.weight]! && gradsTest[dTest]! == grads[layer.input]! && gradsTest[bTest]! == grads[layer.bias]!, "hand written linear grads vs linear layer grads")
    }
    
    func testFakeNetLinear() throws {
        // Make our layers
        let sequence: Sequence = Sequence([
            Linear(to: 3, out: 5),
            Linear(to: 5, out: 10)
        ])
        let J = sequence.predicted.sum()
        let session = Session(parallel: parallel)
        session.build(J)
        let dt = DTensor([[1, 2, 3], [12, 23, 34], [13, 24, 35], [19, 28, 37]]).transpose()
        session.pass([sequence.input: dt])
        let (out, grads) = session.run(J)
        print(out)
        session.descend(grads: grads, optim: Adam(), lr: 0.3)
    }
    
    func testSigmoid() throws {
        let dt = DTensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        let d = Variable(dt)
        let layer = Sigmoid(d)
        let J = (layer <*> Variable(DTensor([[1, 2, 3, 4], [5, 6, 7, 8]]).transpose())).sum()
        let session = Session(parallel: parallel)
        session.build(J)
        session.pass([layer.input: dt])
        let (out, grads) = session.run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[layer.input] ?? "Nil")
        
        // NOW LETS TEST IF THIS LAYER IS CORRECT BY DOING IT BY 'HAND'
        let dtTest = DTensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        let dTest = Variable(dtTest)
        func loss(_ a: Variable) -> Variable {
            return ((Variable(1.0) / (Variable(1.0) + (Negate(a)).exp())) <*> Variable(DTensor([[1, 2, 3, 4], [5, 6, 7, 8]]).transpose())).sum()
        }
        let JTest = loss(dTest)
        // Forward and backward prop wrt J through session
        let sessionTest = Session(parallel: parallel)
        sessionTest.build(JTest)
        let (outTest, gradsTest) = sessionTest.run(JTest)
//        print("----------\n", outTest)
//        print(gradsTest[JTest] ?? "NIL", gradsTest[dTest] ?? "NIL")
        // Test for correctness
        let epsilon = Variable(eps)
        let grad_dTest_numerical = (loss(dTest + epsilon) - loss(dTest - epsilon)) / (Variable(2.0) * epsilon)
        let graph3 = CGraph(grad_dTest_numerical, seed: DTensor(1))
        graph3.fwd()
    //        print(grad_dTest_numerical.out ?? "NIL")
        let _ = graph3.bwd()
        func get_grad_dTest_numericals(_ pos_dTest: Variable, _ neg_dTest: Variable, idx: Int) {
            let epsilon = Variable(eps)
            let grad_dTest_numerical = (loss(pos_dTest) - loss(neg_dTest)) / (Variable(2.0) * epsilon)
            let graph3 = CGraph(grad_dTest_numerical, seed: DTensor(1))
            graph3.fwd()
//            print(grad_dTest_numerical.out ?? "NIL")
            let _ = graph3.bwd()

            let diff_dTest = abs(gradsTest[dTest]!.grid[idx] - grad_dTest_numerical.out!.grid.first!) / max(abs(gradsTest[dTest]!.grid[idx]), abs(grad_dTest_numerical.out!.grid.first!))
            XCTAssert(diff_dTest < bound, "analytical vs numerical gradient check for dTest")
        }
        for i in 0..<dtTest.grid.count {
            var pos_grid = dtTest.grid
            pos_grid[i] = pos_grid[i] + eps
            var neg_grid = dtTest.grid
            neg_grid[i] = neg_grid[i] - eps
            let pos_dtTest = Variable(DTensor(shape: dtTest.shape, grid: pos_grid))
            let neg_dtTest = Variable(DTensor(shape: dtTest.shape, grid: neg_grid))
            get_grad_dTest_numericals(pos_dtTest, neg_dtTest, idx: i)
        }
        // Makes sure that our hand computed linear results and our linear layers results are identical
        XCTAssert(outTest == out, "hand written linear out vs linear layer out")
        XCTAssert((gradsTest[JTest]! - grads[J]!).grid.allSatisfy { $0 < eps } && (gradsTest[dTest]! - grads[layer.input]!).grid.allSatisfy { $0 < eps }, "hand written linear grads vs linear layer grads")
    }
    
    func testFakeNetSigmoid() throws {
        // Make our layers
        let sequence: Sequence = Sequence([
            Linear(to: 3, out: 5),
            Sigmoid(),
            Linear(to: 5, out: 10),
            Sigmoid()
        ])
        let J = sequence.predicted.sum()
        let session = Session(parallel: parallel)
        session.build(J)
        let dt = DTensor([[1, 2, 3], [12, 23, 34], [13, 24, 35], [19, 28, 37]]).transpose()
        session.pass([sequence.input: dt])
        let (out, grads) = session.run(J)
        print(out)
        session.descend(grads: grads, optim: Adam(), lr: 0.3)
    }
    
    func testMseNet() throws {
        // Data
        var data = [[Double]]()
        for _ in 0..<1000 {
            data.append([Double.random(in: 0.0...1.0), Double.random(in: 0.0...1.0), Double.random(in: 0.0...1.0)])
        }
        var labels = [[Double]]()
        for j in 0..<1000 {
            let inverted = data[j].map { 1 - $0 }
            labels.append(inverted)
        }
        let process: SML2.Process = Process()
        let (shuffledData, shuffledLabels) = process.shuffle(data: data, labels: labels)
        // Make our layers
        let sequence = Sequence([
            Linear(to: 3, out: 6),
//            BatchNorm(to: 6),
            ReLU(),
            Linear(to: 6, out: 6),
//            BatchNorm(to: 6),
            ReLU(),
            Linear(to: 6, out: 3),
//            BatchNorm(to: 3),
            ReLU()
        ])
        // Make other necessary nodes
        let expected = Placeholder()
        let avg = Placeholder()
        let avg_lm = Placeholder()
        let lm = Constant(0.0001)
        // MSE Loss Function (Vectorized)! also our computational graph lol
        let J = avg * (sequence.predicted - expected).pow(2).sum() + avg_lm * (lm * sequence.regularizer)
        let session = Session(parallel: parallel)
        session.build(J)
        let X: DTensor = process.zscore(DTensor(shuffledData), type: .data)
        let Y: DTensor = DTensor(shuffledLabels)
        let optim = Adam()
        // Minibatch
        let b = 100
        // Set up our number of batches
        let batches = Int(ceil(Double(X.shape[0]) / Double(b)))
        // Set up all of our mini batchs in advance
        var mini_batches = [(DTensor, DTensor)](repeating: (DTensor(shape: [], grid: []), DTensor(shape: [], grid: [])), count: batches)
        // Partition X and Y
        for s in 0..<batches {
            // Get the indicies for our batches
            let start = s * b
            let end = (s + 1) * b
            // Set up sth mini X batch
            let XT_mini = X[rows: start..<(end < X.shape[0] ? end : X.shape[0])].transpose()
            // Set up sth mini X batch
            let YT_mini = Y[rows: start..<(end < Y.shape[0] ? end : Y.shape[0])].transpose()
            // Now add this mini batch to our mini bathces
            mini_batches[s] = (XT_mini, YT_mini)
        }
        // Train for # of epochs
        for i in 0..<1000 {
            // Adam requires incrementing t step each epoch
            optim.inc()
            // Set up empty loss for this epoch
            var loss = 0.0
            // Run mini batch gradient descent to train network
            for (XT_mini, YT_mini) in mini_batches {
                // Get mini batch size
                let m_mini = Double(XT_mini.shape[1])
                // Reset our placeholders for our input data and avg coefficient
                session.pass([sequence.input: XT_mini, expected: YT_mini, avg: DTensor(1.0 / (2.0 * m_mini)), avg_lm: DTensor(1.0 / (2.0 * m_mini))])
                // Forward and backward
                let (out, grads) = session.run(J)
                // Add to loss
                loss += out.grid.first!
                // Run gradient descent with Adam optimizer
                session.descend(grads: grads, optim: optim, lr: 0.001)
            }
            // Average loss from each batches lsos
            loss = loss / Double(batches)
            print("Loss \(loss), Epoch \(i + 1)")
        }
//        // Set BatchNorm layers to test
//        for batch_norm in sequence.batch_norms {
//            batch_norm.training = false
//        }
        // Predict on example
        let test1: [Double] = [0.45, 0.21, 0.89]
        // Rset our placeholder for our input data
        session.pass([sequence.input: process.zscore(DTensor(test1, type: .column), type: .pred)])
        // Stop forwarding after we have our predicted (other dependencies for J may fire during forward if they have a low dependency count despite not contributing to sequence.predicted sub graph)
        let (out, _) = session.run(J, till: sequence.predicted)
        // Should be [0.55, 0.79, 0.11]
        print(out)
    }
    
    func testMseNet2() throws {
        // Data
        var data = [[Double]]()
        for _ in 0..<10000 {
            data.append([Double.random(in: -15.0...15.0)])
        }
        var labels = [[Double]]()
        for j in 0..<10000 {
            labels.append([pow(data[j][0], 2)])
        }
        let process: SML2.Process = Process()
        let (shuffledData, shuffledLabels) = process.shuffle(data: data, labels: labels)
        // Make our layers
        let sequence = Sequence([
            Linear(to: 1, out: 8),
//            BatchNorm(to: 8),
            LReLU(),
            Linear(to: 8, out: 8),
//            BatchNorm(to: 8),
            LReLU(),
            Linear(to: 8, out: 8),
//            BatchNorm(to: 8),
            LReLU(),
            Linear(to: 8, out: 8),
//            BatchNorm(to: 8),
            LReLU(),
            Linear(to: 8, out: 1),
//            BatchNorm(to: 1),
            LReLU()
        ])
        // Make other necessary nodes
        let expected = Placeholder()
        let avg = Placeholder()
        let avg_lm = Placeholder()
        let lm = Constant(0.0001)
        // MSE Loss Function (Vectorized)! also our computational graph lol
        let J = avg * (sequence.predicted - expected).pow(2).sum() + avg_lm * (lm * sequence.regularizer)
        let session = Session(parallel: parallel)
        session.build(J)
        let X: DTensor = process.zscore(DTensor(shuffledData), type: .data)
        let Y: DTensor = DTensor(shuffledLabels)
        let optim = Adam()
        // Minibatch
        let b = 100
        // Set up our number of batches
        let batches = Int(ceil(Double(X.shape[0]) / Double(b)))
        // Set up all of our mini batchs in advance
        var mini_batches = [(DTensor, DTensor)](repeating: (DTensor(shape: [], grid: []), DTensor(shape: [], grid: [])), count: batches)
        // Partition X and Y
        for s in 0..<batches {
            // Get the indicies for our batches
            let start = s * b
            let end = (s + 1) * b
            // Set up sth mini X batch
            let XT_mini = X[rows: start..<(end < X.shape[0] ? end : X.shape[0])].transpose()
            // Set up sth mini X batch
            let YT_mini = Y[rows: start..<(end < Y.shape[0] ? end : Y.shape[0])].transpose()
            // Now add this mini batch to our mini bathces
            mini_batches[s] = (XT_mini, YT_mini)
        }
        // Train for # of epochs
        for i in 0..<400 {
            // Adam requires incrementing t step each epoch
            optim.inc()
            // Set up empty loss for this epoch
            var loss = 0.0
            // Run mini batch gradient descent to train network
            for (XT_mini, YT_mini) in mini_batches {
                // Get mini batch size
                let m_mini = Double(XT_mini.shape[1])
                // Reset our placeholders for our input data and avg coefficient
                session.pass([sequence.input: XT_mini, expected: YT_mini, avg: DTensor(1.0 / (2.0 * m_mini)), avg_lm: DTensor(1.0 / (2.0 * m_mini))])
                // Forward and backward
                let (out, grads) = session.run(J)
                // Add to loss
                loss += out.grid.first!
                // Run gradient descent with Adam optimizer
                session.descend(grads: grads, optim: optim, lr: 0.003)
            }
            // Average loss from each batches lsos
            loss = loss / Double(batches)
            print("Loss \(loss), Epoch \(i + 1)")
        }
//        // Set BatchNorm layers to test
//        for batch_norm in sequence.batch_norms {
//            batch_norm.training = false
//        }
        // Predict on example
        let test1: [Double] = [1.0]
        let test2: [Double] = [4.0]
        let test3: [Double] = [-3.0]
        let test4: [Double] = [12.0]
        let test5: [Double] = [10.0]
        let test6: [Double] = [-7.0]
        let tests = [test1, test2, test3, test4, test5, test6]
        for test in tests {
            // Rset our placeholder for our input data
            session.pass([sequence.input: process.zscore(DTensor(test, type: .column), type: .pred)])
            // Stop forwarding after we have our predicted (other dependencies for J may fire during forward if they have a low dependency count despite not contributing to sequence.predicted sub graph)
            let (out, _) = session.run(J, till: sequence.predicted)
            // Should be class
            print(out.grid, test.first!)
        }
    }
    
    func testBinaryCrossEntropy() throws {
        // Data
        var data = [[Double]]()
        for i in 0..<3000 {
            let val = Double(i) / 100.0
            if val < 10.0 {
                data.append([Double.random(in: 0..<10.0)])
            } else if val >= 10.0 && val < 20.0 {
                data.append([Double.random(in: 10.0..<20.0)])
            } else {
                data.append([Double.random(in: 20.0...30.0)])
            }
        }
        var labels = [[Double]]()
        for j in 0..<3000 {
            let val = Double(j) / 100.0
            if val < 10.0 {
                labels.append([1, 0, 0]) // 0 - 10 is CLASS 1
            } else if val >= 10.0 && val < 20.0 {
                labels.append([0, 1, 0]) // 10 - 20 is CLASS 2
            } else {
                labels.append([0, 0, 1]) // 20 - 30 is CLASS 3
            }
        }
        let process: SML2.Process = Process()
        let (shuffledData, shuffledLabels) = process.shuffle(data: data, labels: labels)
        // Make our layers
        let sequence = Sequence([
            Linear(to: 1, out: 2),
//            BatchNorm(to: 2),
            LReLU(),
            Linear(to: 2, out: 2),
//            BatchNorm(to: 2),
            LReLU(),
            Linear(to: 2, out: 3),
//            BatchNorm(to: 3),
            Sigmoid()
        ])
        // Make other necessary nodes
        let expected = Placeholder()
        let avg = Placeholder()
        let avg_lm = Placeholder()
        let lm = Constant(0.0001)
        // Binary Cross Entropy Loss Function (Vectorized)! also our computational graph lol
        let J = avg * Constant(-1.0) * ((sequence.predicted.transpose() + Constant(0.00000001)).log() <*> expected + ((Constant(1.0) - sequence.predicted.transpose() + Constant(0.00000001)).log() <*> (Constant(1.0) - expected))).sumDiag() + avg_lm * (lm * sequence.regularizer)
        let session = Session(parallel: parallel)
        session.build(J)
        let X: DTensor = process.zscore(DTensor(shuffledData), type: .data)
        let Y: DTensor = DTensor(shuffledLabels)
        let optim = Adam()
        // Minibatch
        let b = 100
        // Set up our number of batches
        let batches = Int(ceil(Double(X.shape[0]) / Double(b)))
        // Set up all of our mini batchs in advance
        var mini_batches = [(DTensor, DTensor)](repeating: (DTensor(shape: [], grid: []), DTensor(shape: [], grid: [])), count: batches)
        // Partition X and Y
        for s in 0..<batches {
            // Get the indicies for our batches
            let start = s * b
            let end = (s + 1) * b
            // Set up sth mini X batch
            let XT_mini = X[rows: start..<(end < X.shape[0] ? end : X.shape[0])].transpose()
            // Set up sth mini X batch
            let YT_mini = Y[rows: start..<(end < Y.shape[0] ? end : Y.shape[0])].transpose()
            // Now add this mini batch to our mini bathces
            mini_batches[s] = (XT_mini, YT_mini)
        }
        // Train for # of epochs
        for i in 0..<1000 {
            // Adam requires incrementing t step each epoch
            optim.inc()
            // Set up empty loss for this epoch
            var loss = 0.0
            // Run mini batch gradient descent to train network
            for (XT_mini, YT_mini) in mini_batches {
                // Get mini batch size
                let m_mini = Double(XT_mini.shape[1])
                // Reset our placeholders for our input data and avg coefficient
                session.pass([sequence.input: XT_mini, expected: YT_mini, avg: DTensor(1.0 / (m_mini)), avg_lm: DTensor(1.0 / (m_mini))])
                // Forward and backward
                let (out, grads) = session.run(J)
                // Add to loss
                loss += out.grid.first!
                // Run gradient descent with Adam optimizer
                session.descend(grads: grads, optim: optim, lr: 0.01)
            }
            // Average loss from each batches lsos
            loss = loss / Double(batches)
            print("Loss \(loss), Epoch \(i + 1)")
        }
//        // Set BatchNorm layers to test
//        for batch_norm in sequence.batch_norms {
//            batch_norm.training = false
//        }
        // Predict on example
        let test1: [Double] = [1.0]
        let test2: [Double] = [5.0]
        let test3: [Double] = [9.0]
        let test4: [Double] = [9.9]
        let test5: [Double] = [10.0]
        let test6: [Double] = [10.1]
        let test7: [Double] = [11.0]
        let test8: [Double] = [12.0]
        let test9: [Double] = [17.4]
        let test10: [Double] = [28.0]
        let test11: [Double] = [29.0]
        let tests = [test1, test2, test3, test4, test5, test6, test7, test8, test9, test10, test11]
        for test in tests {
            // Rset our placeholder for our input data
            session.pass([sequence.input: process.zscore(DTensor(test, type: .column), type: .pred)])
            // Stop forwarding after we have our predicted (other dependencies for J may fire during forward if they have a low dependency count despite not contributing to sequence.predicted sub graph)
            let (out, _) = session.run(J, till: sequence.predicted)
            // Should be class
            print(out.grid, test.first!)
        }
    }
    
    func testConvLayer() throws {
        var pad = false
        pad = false
        
        let dt1 = DTensor([[[15, 24, 13, 44, 52], [12, 24, 35, 64, 85], [51, 22, 73, 94, 55], [11, 52, 36, 47, 59], [41, 62, 83, 94, 75]], [[14, 23, 23, 44, 56], [21, 23, 353, 54, 65], [61, 42, 35, 44, 25], [91, 2, 38, 64, 54], [19, 25, 53, 40, 55]]])
        let conv1 = Conv2D(to: 2, out: 1, size: 3, pad: pad)
        
        let dt2 = DTensor([[[15, 24, 13, 44, 52], [12, 24, 35, 64, 85], [51, 22, 73, 94, 55], [11, 52, 36, 47, 59], [41, 62, 83, 94, 75]], [[14, 23, 23, 44, 56], [21, 23, 353, 54, 65], [61, 42, 35, 44, 25], [91, 2, 38, 64, 54], [19, 25, 53, 40, 55]]])
        let conv2 = Conv2D(to: 2, out: 2, size: 3, pad: pad)
        
        let dt3 = DTensor([[[15, 24, 13, 44, 52], [12, 24, 35, 64, 85], [51, 22, 73, 94, 55], [11, 52, 36, 47, 59], [41, 62, 83, 94, 75]]])
        let conv3 = Conv2D(to: 1, out: 1, size: 3, pad: pad)
        
        let dt4 = DTensor([[[15, 24, 13, 44, 52], [12, 24, 35, 64, 85], [51, 22, 73, 94, 55], [11, 52, 36, 47, 59], [41, 62, 83, 94, 75]]])
        let conv4 = Conv2D(to: 1, out: 4, size: 3, pad: pad)
        
        let DT = [dt1, dt2, dt3, dt4]
        let CONV = [conv1, conv2, conv3, conv4]
        let SHAPE: [[Int]] = pad ? [[1, 5, 5], [2, 5, 5], [1, 5, 5], [4, 5, 5]] : [[1, 3, 3], [2, 3, 3], [1, 3, 3], [4, 3, 3]]
        
        var i = 0
        // grad check for each convolution type variation
        for (dt, conv) in zip(DT, CONV) {
            let J = conv.sum()
            let session = Session(parallel: parallel)
            session.build(J)
            session.pass([conv.input: dt])
            let (out, grads) = session.run(J)
//            print("----------\n", out)
//            print(grads[J] ?? "NIL", grads[conv.input] ?? "NIL", grads[conv.kernel] ?? "NIL")
            
            // NOW LETS TEST IF THIS LAYER IS CORRECT BY DOING IT BY 'HAND'
            // Kernel init'd randomly so this makes sure they are the same random vals for both weight matrices
            let kernelTest = conv.kernel.out!
            let biasTest = conv.bias.out!
            let dataTest = conv.input.out!
            func loss(_ data: DTensor, _ kernel: DTensor, _ bias: DTensor) -> Double {
                var out: DTensor?
                if kernel.shape.count == 4 && data.shape.count == 3 && kernel.shape[1] == data.shape[0] {
                    // First get the shape of our 2D Tensor after convolution
                    let mat_shape = pad ? Array(data.shape[1...2]).pad_shape((kernel.shape[2] - 1) / 2, (kernel.shape[3] - 1) / 2).conv2D_shape(with: Array(kernel.shape[2...3]), type: .valid) : Array(data.shape[1...2]).conv2D_shape(with: Array(kernel.shape[2...3]), type: .valid)
                    // Now we can make the out shape using the 2D Tensor (matrix) with a depth of the number of kernels since out must have the same depth as the number of kernels
                    out = DTensor(shape: [kernel.shape[0], mat_shape[0], mat_shape[1]], repeating: 0.0)
                    // Now for each kernel we convolve with our data to produce our dth depth for out
                    for d in 0..<kernel.shape[0] {
                        // Get the dth kernel
                        let kernelD = kernel[t3D: d]
                        // Now convolve this kernel with our data, since both kernel and data are 3D Tensors we convolve the corresponding depth of data with that of kernelD
                        for m in 0..<kernel.shape[1] {
                            out![mat: d] = pad ? out![mat: d] + data[mat: m].pad((kernel.shape[2] - 1) / 2, (kernel.shape[3] - 1) / 2).conv2D(with: kernelD[mat: m], type: .valid) : out![mat: d] + data[mat: m].conv2D(with: kernelD[mat: m], type: .valid)
                        }
                        // Add the bias for the dth depth of out which corresponds to the dth kernel
                        out![mat: d] = out![mat: d] + bias[d]
                    }
                    // For more clarity, this is essentially the following forward calculation (One kernel with depth > 1) except we have multiple kernels that contribute to out so we need to do convolutions for each kernel with data which will produce a new depth for out (making out a multi depth output)
                } else {
                    fatalError("Data and kernels are incompatible")
                }
                return out!.sum()
            }
            let outTest = loss(dataTest, kernelTest, biasTest)
            // Test data
            func get_grad_data_numericals(_ pos_data: DTensor, _ neg_data: DTensor, idx: Int) {
                let grad_data_numerical = (loss(pos_data, kernelTest, biasTest) - loss(neg_data, kernelTest, biasTest)) / (2.0 * eps)
                
                let diff_data = abs(grads[conv.input]!.grid[idx] - grad_data_numerical) / max(abs(grads[conv.input]!.grid[idx]), abs(grad_data_numerical))
//                print(diff_data)
                XCTAssert(diff_data < bound, "analytical vs numerical gradient check for dataTest")
            }
            for i in 0..<dataTest.grid.count {
                var pos_grid = dataTest.grid
                pos_grid[i] = pos_grid[i] + eps
                var neg_grid = dataTest.grid
                neg_grid[i] = neg_grid[i] - eps
                let pos_data = DTensor(shape: dataTest.shape, grid: pos_grid)
                let neg_data = DTensor(shape: dataTest.shape, grid: neg_grid)
                get_grad_data_numericals(pos_data, neg_data, idx: i)
            }
            // Test kernel
            func get_grad_kernel_numericals(_ pos_kernel: DTensor, _ neg_kernel: DTensor, idx: Int) {
                let grad_kernel_numerical = (loss(dataTest, pos_kernel, biasTest) - loss(dataTest, neg_kernel, biasTest)) / (2.0 * eps)

                let diff_kernel = abs(grads[conv.kernel]!.grid[idx] - grad_kernel_numerical) / max(abs(grads[conv.kernel]!.grid[idx]), abs(grad_kernel_numerical))
//                print(diff_kernel)
                XCTAssert(diff_kernel < bound, "analytical vs numerical gradient check for kernelTest")
            }
            for i in 0..<kernelTest.grid.count {
                var pos_grid = kernelTest.grid
                pos_grid[i] = pos_grid[i] + eps
                var neg_grid = kernelTest.grid
                neg_grid[i] = neg_grid[i] - eps
                let pos_kernel = DTensor(shape: kernelTest.shape, grid: pos_grid)
                let neg_kernel = DTensor(shape: kernelTest.shape, grid: neg_grid)
                get_grad_kernel_numericals(pos_kernel, neg_kernel, idx: i)
            }
            // Test bias
            func get_grad_bias_numericals(_ pos_bias: DTensor, _ neg_bias: DTensor, idx: Int) {
                let grad_bias_numerical = (loss(dataTest, kernelTest, pos_bias) - loss(dataTest, kernelTest, neg_bias)) / (2.0 * eps)
                
                let diff_bias = abs(grads[conv.bias]!.grid[idx] - grad_bias_numerical) / max(abs(grads[conv.bias]!.grid[idx]), abs(grad_bias_numerical))
//                print(diff_bias)
                XCTAssert(diff_bias < bound, "analytical vs numerical gradient check for biasTest")
            }
            for i in 0..<biasTest.grid.count {
                var pos_grid = biasTest.grid
                pos_grid[i] = pos_grid[i] + eps
                var neg_grid = biasTest.grid
                neg_grid[i] = neg_grid[i] - eps
                let pos_bias = DTensor(shape: biasTest.shape, grid: pos_grid)
                let neg_bias = DTensor(shape: biasTest.shape, grid: neg_grid)
                get_grad_bias_numericals(pos_bias, neg_bias, idx: i)
            }
            // Makes sure that our hand computed linear results and our linear layers results are identical
//            print(outTest, out)
            XCTAssert(conv.out!.shape == SHAPE[i], "convolution shape output")
            XCTAssert(abs(outTest - out.grid.first!) < eps, "hand written conv out vs linear conv out")
//            print(grads[conv.input]!.shape, conv.input.out!.shape, grads[conv.kernel]!.shape, conv.kernel.out!.shape, grads[conv.bias]!.shape, conv.bias.out!.shape)
            XCTAssert(grads[conv.input]!.shape == conv.input.out!.shape && grads[conv.kernel]!.shape == conv.kernel.out!.shape && grads[conv.bias]!.shape == conv.bias.out!.shape)
            i += 1
        }
    }
    
    func testConvLayerBatch() throws {
        var pad = false
        pad = false
        
        let dt11: [[[Double]]] = [[[15, 24, 13, 44, 52], [12, 24, 35, 64, 85], [51, 22, 73, 94, 55], [11, 52, 36, 47, 59], [41, 62, 83, 94, 75]], [[14, 23, 23, 44, 56], [21, 23, 353, 54, 65], [61, 42, 35, 44, 25], [91, 2, 38, 64, 54], [19, 25, 53, 40, 55]]]
        let dt12 = dt11.map { $0.map { $0.map { $0 + 394 } } }
        let dt1 = DTensor([dt11, dt12])
        let conv1 = Conv2D(to: 2, out: 1, size: 3, pad: pad)
        
        let dt21: [[[Double]]] = [[[15, 24, 13, 44, 52], [12, 24, 35, 64, 85], [51, 22, 73, 94, 55], [11, 52, 36, 47, 59], [41, 62, 83, 94, 75]], [[14, 23, 23, 44, 56], [21, 23, 353, 54, 65], [61, 42, 35, 44, 25], [91, 2, 38, 64, 54], [19, 25, 53, 40, 55]]]
        let dt22 = dt21.map { $0.map { $0.map { $0 + 58 } } }
        let dt2 = DTensor([dt21, dt22])
        let conv2 = Conv2D(to: 2, out: 2, size: 3, pad: pad)
        
        let dt31: [[[Double]]] = [[[15, 24, 13, 44, 52], [12, 24, 35, 64, 85], [51, 22, 73, 94, 55], [11, 52, 36, 47, 59], [41, 62, 83, 94, 75]]]
        let dt32 = dt31.map { $0.map { $0.map { $0 + 23 } } }
        let dt3 = DTensor([dt31, dt32])
        let conv3 = Conv2D(to: 1, out: 1, size: 3, pad: pad)
        
        let dt41: [[[Double]]] = [[[15, 24, 13, 44, 52], [12, 24, 35, 64, 85], [51, 22, 73, 94, 55], [11, 52, 36, 47, 59], [41, 62, 83, 94, 75]]]
        let dt42 = dt41.map { $0.map { $0.map { $0 + 21 } } }
        let dt4 = DTensor([dt41, dt42])
        let conv4 = Conv2D(to: 1, out: 4, size: 3, pad: pad)
        
        let DT = [dt1, dt2, dt3, dt4]
        let CONV = [conv1, conv2, conv3, conv4]
        let SHAPE: [[Int]] = pad ? [[2, 1, 5, 5], [2, 2, 5, 5], [2, 1, 5, 5], [2, 4, 5, 5]] : [[2, 1, 3, 3], [2, 2, 3, 3], [2, 1, 3, 3], [2, 4, 3, 3]]
        
        var i = 0
        // grad check for each convolution type variation
        for (dt, conv) in zip(DT, CONV) {
            let J = conv.sum()
            let session = Session(parallel: parallel)
            session.build(J)
            session.pass([conv.input: dt])
            let (out, grads) = session.run(J)
//            print("----------\n", out)
//            print(grads[J] ?? "NIL", grads[conv.input] ?? "NIL", grads[conv.kernel] ?? "NIL")
            
            // NOW LETS TEST IF THIS LAYER IS CORRECT BY DOING IT BY 'HAND'
            // Kernel init'd randomly so this makes sure they are the same random vals for both weight matrices
            let kernelTest = conv.kernel.out!
            let biasTest = conv.bias.out!
            let dataTest = conv.input.out!
            func loss(_ data: DTensor, _ kernel: DTensor, _ bias: DTensor) -> Double {
                var out: DTensor?
                if kernel.shape.count == 4 && data.shape.count == 3 && kernel.shape[1] == data.shape[0] {
                    // First get the shape of our 2D Tensor after convolution
                    let mat_shape = pad ? Array(data.shape[1...2]).pad_shape((kernel.shape[2] - 1) / 2, (kernel.shape[3] - 1) / 2).conv2D_shape(with: Array(kernel.shape[2...3]), type: .valid) : Array(data.shape[1...2]).conv2D_shape(with: Array(kernel.shape[2...3]), type: .valid)
                    // Now we can make the out shape using the 2D Tensor (matrix) with a depth of the number of kernels since out must have the same depth as the number of kernels
                    out = DTensor(shape: [kernel.shape[0], mat_shape[0], mat_shape[1]], repeating: 0.0)
                    // Now for each kernel we convolve with our data to produce our dth depth for out
                    for d in 0..<kernel.shape[0] {
                        // Get the dth kernel
                        let kernelD = kernel[t3D: d]
                        // Now convolve this kernel with our data, since both kernel and data are 3D Tensors we convolve the corresponding depth of data with that of kernelD
                        for m in 0..<kernel.shape[1] {
                            out![mat: d] = pad ? out![mat: d] + data[mat: m].pad((kernel.shape[2] - 1) / 2, (kernel.shape[3] - 1) / 2).conv2D(with: kernelD[mat: m], type: .valid) : out![mat: d] + data[mat: m].conv2D(with: kernelD[mat: m], type: .valid)
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
                    out = DTensor(shape: [data.shape[0], kernel.shape[0], mat_shape[0], mat_shape[1]], repeating: 0.0)
                    // Now for each image we do a convolution
                    for n in 0..<data.shape[0] {
                        // Now for each kernel we convolve with our data to produce our dth depth for out
                        for d in 0..<kernel.shape[0] {
                            // Get the dth kernel
                            let kernelD = kernel[t3D: d]
                            // Now convolve this kernel with our data, since both kernel and data are 3D Tensors we convolve the corresponding depth of data with that of kernelD
                            for m in 0..<kernel.shape[1] {
                                // pad to make same convolution if we have padding
                                out![t3D: n][mat: d] = pad ? out![t3D: n][mat: d] + data[t3D: n][mat: m].pad((kernel.shape[2] - 1) / 2, (kernel.shape[3] - 1) / 2).conv2D(with: kernelD[mat: m], type: .valid) : out![t3D: n][mat: d] + data[t3D: n][mat: m].conv2D(with: kernelD[mat: m], type: .valid)
                            }
                            // Add the bias for the dth depth of out which corresponds to the dth kernel
                            out![t3D: n][mat: d] = out![t3D: n][mat: d] + bias[d]
                        }
                    }
                } else {
                    fatalError("Data and kernels are incompatible")
                }
                return out!.sum()
            }
            let outTest = loss(dataTest, kernelTest, biasTest)
            // Test data
            func get_grad_data_numericals(_ pos_data: DTensor, _ neg_data: DTensor, idx: Int) {
                let grad_data_numerical = (loss(pos_data, kernelTest, biasTest) - loss(neg_data, kernelTest, biasTest)) / (2.0 * eps)
                
                let diff_data = abs(grads[conv.input]!.grid[idx] - grad_data_numerical) / max(abs(grads[conv.input]!.grid[idx]), abs(grad_data_numerical))
//                print(diff_data, idx, "data")
                XCTAssert(diff_data < bound * 10, "analytical vs numerical gradient check for dataTest")
            }
            for i in 0..<dataTest.grid.count {
                var pos_grid = dataTest.grid
                pos_grid[i] = pos_grid[i] + eps
                var neg_grid = dataTest.grid
                neg_grid[i] = neg_grid[i] - eps
                let pos_data = DTensor(shape: dataTest.shape, grid: pos_grid)
                let neg_data = DTensor(shape: dataTest.shape, grid: neg_grid)
                get_grad_data_numericals(pos_data, neg_data, idx: i)
            }
            // Test kernel
            func get_grad_kernel_numericals(_ pos_kernel: DTensor, _ neg_kernel: DTensor, idx: Int) {
                let grad_kernel_numerical = (loss(dataTest, pos_kernel, biasTest) - loss(dataTest, neg_kernel, biasTest)) / (2.0 * eps)

                let diff_kernel = abs(grads[conv.kernel]!.grid[idx] - grad_kernel_numerical) / max(abs(grads[conv.kernel]!.grid[idx]), abs(grad_kernel_numerical))
//                print(diff_kernel, idx, "kernel")
                XCTAssert(diff_kernel < bound, "analytical vs numerical gradient check for kernelTest")
            }
//            print(kernelTest.grid.count)
            for i in 0..<kernelTest.grid.count {
                var pos_grid = kernelTest.grid
                pos_grid[i] = pos_grid[i] + eps
                var neg_grid = kernelTest.grid
                neg_grid[i] = neg_grid[i] - eps
                let pos_kernel = DTensor(shape: kernelTest.shape, grid: pos_grid)
                let neg_kernel = DTensor(shape: kernelTest.shape, grid: neg_grid)
                get_grad_kernel_numericals(pos_kernel, neg_kernel, idx: i)
            }
            // Test bias
            func get_grad_bias_numericals(_ pos_bias: DTensor, _ neg_bias: DTensor, idx: Int) {
                let grad_bias_numerical = (loss(dataTest, kernelTest, pos_bias) - loss(dataTest, kernelTest, neg_bias)) / (2.0 * eps)
                
                let diff_bias = abs(grads[conv.bias]!.grid[idx] - grad_bias_numerical) / max(abs(grads[conv.bias]!.grid[idx]), abs(grad_bias_numerical))
//                print(diff_bias, idx, "bias")
                XCTAssert(diff_bias < bound, "analytical vs numerical gradient check for biasTest")
            }
            for i in 0..<biasTest.grid.count {
                var pos_grid = biasTest.grid
                pos_grid[i] = pos_grid[i] + eps
                var neg_grid = biasTest.grid
                neg_grid[i] = neg_grid[i] - eps
                let pos_bias = DTensor(shape: biasTest.shape, grid: pos_grid)
                let neg_bias = DTensor(shape: biasTest.shape, grid: neg_grid)
                get_grad_bias_numericals(pos_bias, neg_bias, idx: i)
            }
            // Makes sure that our hand computed linear results and our linear layers results are identical
//            print(outTest, out)
//            print(conv.out!.shape, SHAPE[i])
            XCTAssert(conv.out!.shape == SHAPE[i], "convolution shape output")
            XCTAssert(abs(outTest - out.grid.first!) < eps, "hand written conv out vs conv layer out")
//            print(grads[conv.input]!.shape, conv.input.out!.shape, grads[conv.kernel]!.shape, conv.kernel.out!.shape, grads[conv.bias]!.shape, conv.bias.out!.shape)
            XCTAssert(grads[conv.input]!.shape == conv.input.out!.shape && grads[conv.kernel]!.shape == conv.kernel.out!.shape && grads[conv.bias]!.shape == conv.bias.out!.shape)
            i += 1
        }
    }
    
    func testPoolLayer() throws {
        let size = 2
        // POOL2D IS NOT A STABLE MAX POOLER but does that really matter for gradients? NO, it doesnt
        // This commneted out contains values that overlap maxes which will cause improper gradient checks since max pool layer is unstable (doesn't really matter because the math still works out)
//        let dt: [[Double]] = [[15, 24, 13, 44, 52], [12, 24, 35, 64, 85], [51, 22, 73, 94, 55], [11, 52, 36, 47, 59], [41, 62, 83, 94, 75]]
        let dt1: [[Double]] = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15], [16, 17, 18, 19, 20], [21, 22, 23, 24, 25]]
        let dt2 = dt1.map { $0.map { $0 + 24 } }
        // First is just an image with depth 1, second is image with depth 2, third is 2 images with depth 2
        let ds: [DTensor] = [DTensor([dt1]), DTensor([dt1, dt2]), DTensor([[dt1, dt2], [dt1, dt2].map { $0.map { $0.map { $0 + 392 } } }])]
        let SHAPES: [Shape] = [Shape([1, 4, 4]), Shape([2, 4, 4]), Shape([2, 2, 4, 4])]
        for (d, SHAPE) in zip(ds, SHAPES) {
            let pool = Pool2DMax(size: size, stride: 1)
            let J = pool.sum()
            let session = Session(parallel: parallel)
            session.build(J)
            session.pass([pool.input: d])
            let (out, grads) = session.run(J)
//            print("----------\n", out)
//            print(grads[J] ?? "NIL", grads[conv.input] ?? "NIL", grads[conv.kernel] ?? "NIL")
            
            // Now test
            let dataTest = pool.input.out!
            func loss(_ input: DTensor) -> Double {
                var out: DTensor
                if input.shape.count == 4 {
                    // Make empty out same size as input after pool, mat_shape is the shape of a matrix in our tensor
                    let mat_shape = Array(input.shape[2...3]).pool2D_max_shape(size: size)
                    // Now we can make the out shape using the 2D Tensor (matrix) with a depth of our input depth since pooling always maintains depth and count of our input count
                    out = DTensor(shape: [input.shape[0], input.shape[1], mat_shape[0], mat_shape[1]], repeating: 0.0)
                    // Now for each nth image pool
                    for n in 0..<input.shape[0] {
                        // Now pool each depth in the nth image input
                        for d in 0..<input.shape[1] {
                            // Pool and get positions for backpropagation
                            let (pooled, _) = input[t3D: n][mat: d].pool2D_max(size: size)
                            out[t3D: n][mat: d] = pooled
                        }
                    }
                } else if input.shape.count == 3 {
                    // Make empty out same size as input after pool, mat_shape is the shape of a matrix in our tensor
                    let mat_shape = Array(input.shape[1...2]).pool2D_max_shape(size: size)
                    // Now we can make the out shape using the 2D Tensor (matrix) with a depth of our input depth since pooling always maintains depth
                    out = DTensor(shape: [input.shape[0], mat_shape[0], mat_shape[1]], repeating: 0.0)
                    // positions will only have one 2D array
                    // Now pool each depth in input
                    for d in 0..<input.shape[0] {
                        // Pool and get positions for backpropagation
                        let (pooled, _) = input[mat: d].pool2D_max(size: size)
                        out[mat: d] = pooled
                    }
                } else if input.shape.count == 2 {
                    // Pool and get positions for backpropagation
                    let (pooled, _) = input.pool2D_max(size: size)
                    // Set pooled to be our out
                    out = pooled
                } else {
                    fatalError("Incompatible dimensions for pooling")
                }
                return out.sum()
            }
            let outTest = loss(dataTest)
            // Test input
            func get_grad_data_numericals(_ pos_data: DTensor, _ neg_data: DTensor, idx: Int) {
                let grad_data_numerical = (loss(pos_data) - loss(neg_data)) / (2.0 * eps)
                
                let diff_data = grads[pool.input]!.grid[idx] - grad_data_numerical
//                print(diff_data, idx, "data")
                XCTAssert(diff_data < bound, "analytical vs numerical gradient check for dataTest")
            }
            for i in 0..<dataTest.grid.count {
                var pos_grid = dataTest.grid
                pos_grid[i] = pos_grid[i] + eps
                var neg_grid = dataTest.grid
                neg_grid[i] = neg_grid[i] - eps
                let pos_data = DTensor(shape: dataTest.shape, grid: pos_grid)
                let neg_data = DTensor(shape: dataTest.shape, grid: neg_grid)
                get_grad_data_numericals(pos_data, neg_data, idx: i)
            }
            XCTAssert(pool.out!.shape == SHAPE)
            XCTAssert(pool.input.out!.shape == grads[pool.input]!.shape, "input shape same as grad shape")
            XCTAssert(abs(outTest - out.grid.first!) < eps, "hand written pool out vs pool layer out")
        }
    }
    
    func testFlatten() throws {
        let dt1: [[[Double]]] = [[[15, 24, 13, 44, 52], [12, 24, 35, 64, 85], [51, 22, 73, 94, 55], [11, 52, 36, 47, 59], [41, 62, 83, 94, 75]], [[14, 23, 23, 44, 56], [21, 23, 353, 54, 65], [61, 42, 35, 44, 25], [91, 2, 38, 64, 54], [19, 25, 53, 40, 55]]]
        let dt2 = dt1.map { $0.map { $0.map { $0 + 58 } } }
        
        let threeD = {
            let dt = DTensor(dt1)
            let flatten = Flatten(Variable(dt))
            flatten.forward()
            let out = flatten.out!
            XCTAssert(out.transpose()[row: 0].grid == dt1.flatMap { $0 }.flatMap { $0 }, "flatten correct")
            
            flatten.backward(dOut: flatten.out!)
            let grad = flatten.grads[0]
            XCTAssert(grad == dt, "flatten backwards correct")
        }
        
        let fourD = {
            let dt = DTensor([dt1, dt2])
            let flatten = Flatten(Variable(dt))
            flatten.forward()
            let out = flatten.out!
            XCTAssert(out.transpose()[row: 0].grid == dt1.flatMap { $0 }.flatMap { $0 }, "flatten correct")
            XCTAssert(out.transpose()[row: 1].grid == dt2.flatMap { $0 }.flatMap { $0 }, "flatten correct")
            
            flatten.backward(dOut: flatten.out!)
            let grad = flatten.grads[0]
            XCTAssert(grad == dt, "flatten backwards correct")
        }
        threeD()
        fourD()
    }
    
    func mnist() -> (images: DTensor, labels: DTensor) {
        let train_file: (images: String, labels: String) = ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
        
        let train_url: (images: URL, labels: URL) = (Bundle.module.url(forResource: train_file.images, withExtension: nil)!, Bundle.module.url(forResource: train_file.labels, withExtension: nil)!)
        
        let train_data: (images: Data, labels: Data) = (try! Data(contentsOf: train_url.images), try! Data(contentsOf: train_url.labels))
        
        let train_bytes: (images: [UInt8], labels: [UInt8]) = ([UInt8](train_data.images), [UInt8](train_data.labels))
        
        let use = 200
        
        let train_images = DTensor(shape: [use, 1, 28, 28], grid: Array(train_bytes.images[16..<train_bytes.images.count - (28 * 28 * (60000 - use))]).map { Double($0) })
        let labs: [[Double]] = Array(train_bytes.labels[8..<train_bytes.labels.count - (60000 - use)]).map {
            var arr = Array(repeating: 0.0, count: 10)
            arr[Int($0)] = 1.0
            return arr
        }
        let train_labels = DTensor(shape: [use, 10], grid: labs.flatMap { $0 })
        return (train_images / 255.0, train_labels)
    }
    
    func mnistTest() -> (images: DTensor, labels: DTensor) {
        let test_file: (images: String, labels: String) = ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
        
        let test_url: (images: URL, labels: URL) = (Bundle.module.url(forResource: test_file.images, withExtension: nil)!, Bundle.module.url(forResource: test_file.labels, withExtension: nil)!)
        
        let test_data: (images: Data, labels: Data) = (try! Data(contentsOf: test_url.images), try! Data(contentsOf: test_url.labels))
        
        let test_bytes: (images: [UInt8], labels: [UInt8]) = ([UInt8](test_data.images), [UInt8](test_data.labels))
        
        let use = 10000
        
        let test_images = DTensor(shape: [use, 1, 28, 28], grid: Array(test_bytes.images[16..<test_bytes.images.count - (28 * 28 * (10000 - use))]).map { Double($0) })
        let labs: [[Double]] = Array(test_bytes.labels[8..<test_bytes.labels.count - (10000 - use)]).map {
            var arr = Array(repeating: 0.0, count: 10)
            arr[Int($0)] = 1.0
            return arr
        }
        let test_labels = DTensor(shape: [use, 10], grid: labs.flatMap { $0 })
        return (test_images / 255.0, test_labels)
    }
    
    func testCNNMnist() throws {
        // Make our layers
        let sequence = Sequence([
            Conv2D(to: 1, out: 24, size: 5),
            LReLU(),
            Pool2DMax(size: 2, stride: 2),
            Flatten(),
            Linear(to: 3456, out: 256),
            LReLU(),
            Linear(to: 256, out: 10),
            Sigmoid()
        ])
//        print("loading saved params...")
//        let url = Bundle.module.url(forResource: "data", withExtension: "json")!
//        if let data = try? Data(contentsOf: url) {
//            sequence.decode_params(data)
//            print("saved params loaded...")
//        } else {
//            print("no saved params to load...")
//        }
        // Make other necessary nodes
        let expected = Placeholder()
        let avg = Placeholder()
        let avg_lm = Placeholder()
        let lm = Constant(0.0001)
        // Binary Cross Entropy Loss Function (Vectorized)! also our computational graph lol
        let J = avg * Constant(-1.0) * ((sequence.predicted.transpose() + Constant(0.00000001)).log() <*> expected + ((Constant(1.0) - sequence.predicted.transpose() + Constant(0.00000001)).log() <*> (Constant(1.0) - expected))).sumDiag() + avg_lm * (lm * sequence.regularizer)
        // Make and build session
        let session = Session(parallel: parallel)
        session.build(J)
        // *** TRAIN ***
        print("loading mnist...")
        var (X, Y) = mnist()
        // Set up optimizer
        let optim = Adam()
        // Minibatch
        var b = 50
        // Set up our number of batches
        var batches = Int(ceil(Double(X.shape[0]) / Double(b)))
        // Set up all of our mini batchs in advance
        var mini_batches = [(DTensor, DTensor)](repeating: (DTensor(shape: [], grid: []), DTensor(shape: [], grid: [])), count: batches)
        print("partioning batches...")
        // Partition X and Y
        for s in 0..<batches {
            // Get the indicies for our batches
            let start = s * b
            let end = (s + 1) * b
            // Set up sth mini X batch
            let X_mini = X[t3Ds: start..<(end < X.shape[0] ? end : X.shape[0])]
            // Set up sth mini Y batch
            let YT_mini = Y[rows: start..<(end < Y.shape[0] ? end : Y.shape[0])].transpose()
            // Now add this mini batch to our mini bathces
            mini_batches[s] = (X_mini, YT_mini)
        }
        // Train for # of epochs
        let epochs = 10
        // Passes for time remaining
        let passes = epochs * batches
        var pass = 0
        print("training...")
        let start = CFAbsoluteTimeGetCurrent()
        for i in 0..<epochs {
            // Adam requires incrementing t step each epoch
            optim.inc()
            // Set up empty loss for this epoch
            var loss = 0.0
            // Run mini batch gradient descent to train network
            var curr_batch = 0
            for (X_mini, YT_mini) in mini_batches {
                // Get mini batch size (number of images in this batch)
                let m_mini = Double(X_mini.shape[0])
                let batch_start = CFAbsoluteTimeGetCurrent()
                // Reset our placeholders for our input data and avg coefficient
                // X not transposed since images will be transposed when we flatten, Y transposed since predicted will be transposed when doing math with expected
                session.pass([sequence.input: X_mini, expected: YT_mini, avg: DTensor(1.0 / (m_mini)), avg_lm: DTensor(1.0 / (m_mini))])
                // Forward and backward
                let (out, grads) = session.run(J)
                // Run gradient descent with Adam optimizer
                session.descend(grads: grads, optim: optim, lr: 0.01)
                // Made anotehr pass over data set
                pass += 1
                // Notify batch finished
                print("Finished Batch \(curr_batch + 1) of \(batches) in Epoch \(i + 1) of \(epochs)")
                // Calculate time left
                let batch_time = CFAbsoluteTimeGetCurrent() - batch_start
                let seconds = Int(Double(batch_time) * Double(passes - pass))
                print("--------------- Est Remaining: \(seconds / 3600) Hours, \((seconds % 3600) / 60) Minutes, \((seconds % 3600) % 60) Seconds ---------------")
                // Add to loss
                loss += out.grid.first!
                // Moving onto next batch
                curr_batch += 1
            }
            // Average loss from each batches lsos
            loss = loss / Double(batches)
            // Display loss
            print("*****************************************************************************")
            print("")
            print(":                         Loss \(String(format: "%.10f", loss)), Epoch \(i + 1)                         :")
            print("")
            print("*****************************************************************************")
            print("")
        }
        // Calculate time taken
        let time = CFAbsoluteTimeGetCurrent() - start
        let seconds = Int(Double(time))
        print("Time taken: \(seconds / 3600) Hours, \((seconds % 3600) / 60) Minutes, \((seconds % 3600) % 60) Seconds")
        // *** TEST ***
        // Test our model
        print("")
        print("")
        print("loading mnist test...")
        // Set X and Y to be test data instead of train
        (X, Y) = mnistTest()
        // Minibatch
        b = 50
        // Set up our number of batches
        batches = Int(ceil(Double(X.shape[0]) / Double(b)))
        // Set up all of our mini batchs in advance
        mini_batches = [(DTensor, DTensor)](repeating: (DTensor(shape: [], grid: []), DTensor(shape: [], grid: [])), count: batches)
        print("partioning test batches...")
        // Partition X and Y
        for s in 0..<batches {
            // Get the indicies for our batches
            let start = s * b
            let end = (s + 1) * b
            // Set up sth mini X batch
            let X_mini = X[t3Ds: start..<(end < X.shape[0] ? end : X.shape[0])]
            // Set up sth mini Y batch
            let YT_mini = Y[rows: start..<(end < Y.shape[0] ? end : Y.shape[0])].transpose()
            // Now add this mini batch to our mini bathces
            mini_batches[s] = (X_mini, YT_mini)
        }
        // Set up empty loss for this epoch
        var loss = 0.0
        var curr_batch = 0
        print("gathering test loss...")
        for (X_mini, YT_mini) in mini_batches {
            // Get mini batch size (number of images in this batch)
            let m_mini = Double(X_mini.shape[0])
            // Reset our placeholders for our input data and avg coefficient
            // X not transposed since images will be transposed when we flatten, Y transposed since predicted will be transposed when doing math with expected
            session.pass([sequence.input: X_mini, expected: YT_mini, avg: DTensor(1.0 / (m_mini)), avg_lm: DTensor(1.0 / (m_mini))])
            // Forward and backward
            let (out, _) = session.run(J, till: J)
            // Add to loss
            loss += out.grid.first!
            if batches / 10 == 0 {
                print("@", terminator: "")
            } else {
                if curr_batch % (batches / 10) == 0 { print("@", terminator: "") }
            }
            curr_batch += 1
        }
        print("")
        // Average loss from each batches lsos
        loss = loss / Double(batches)
        var correct = 0
        print("predicting...")
        for i in 0..<X.shape[0] {
                session.pass([sequence.input: X[t3D: i]])
            // Get accuracy
            let pred = session.run(J, till: sequence.predicted).out.grid
            let max_pred = pred.enumerated().max { $0.element < $1.element }!
            let expec = Y[row: i].grid
            let max_expec = expec.enumerated().max { $0.element < $1.element }!
            if max_expec.offset == max_pred.offset { correct += 1 }
            if i % (X.shape[0] / 10) == 0 { print("@", terminator: "") }
        }
        print("")
        // Get accuracy
        let accuracy = Double(correct) / Double(X.shape[0])
        print("Accuracy on test set: \(accuracy), Loss on test set: \(loss)")
        print("")
        
//        // Save
//        print("attempting to save params...")
//        if let data = sequence.encode_params() {
//            let url = Bundle.module.url(forResource: "data", withExtension: "json")!
//            if let _ = try? data.write(to: url) {
//                print("saved params...")
//            } else {
//                print("unable to write params to path...")
//            }
//        } else {
//            print("unable to encode params...")
//        }
//        print("")
    }
}

/*
 vDSP_imgfir is doing a mathematically correct convolution operation (it rotates kernel before running), WE WANT CROSS CORRELATION (maybe gradients are wrong check kernel rot 180, passing gradient check tho?)
 SumAxis needs refactoring!
 Pool is not stable, make it stable?
 Confirm pool stride works
 vDSP_imgfir with even kernel size strange behavior, investigate
 Conv net initialization (maybe done?)
 SumAxis for 3 >= tensors does not work
 Add custom print for tensor
 Test batchnorm layer is correct with 'hand' calculated gradients (already kinda did this but maybe confirm? maybe also do batchnorm the longer way?)
 */
