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
        let dt = Tensor([[1, 2, 3, 4, 5, 6], [12, 23, 34, 45, 56, 67], [13, 24, 35, 46, 57, 68], [19, 28, 37, 46, 55, 64]]).transpose()
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
        let dtTest = Tensor([[1, 2, 3, 4, 5, 6], [12, 23, 34, 45, 56, 67], [13, 24, 35, 46, 57, 68], [19, 28, 37, 46, 55, 64]]).transpose()
        let dTest = Variable(dtTest)
        let bt = Tensor(shape: [4, 1], repeating: 0.01)
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
        let graph2 = CGraph(grad_wTest_numerical, seed: Tensor(1))
        graph2.fwd()
//        print(grad_wTest_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        func get_grad_wTest_numericals(_ pos_wTest: Variable, _ neg_wTest: Variable, idx: Int) {
            let epsilon = Variable(eps)
            let grad_wTest_numerical = (loss(pos_wTest, dTest, bTest) - loss(neg_wTest, dTest, bTest)) / (Variable(2.0) * epsilon)
            let graph2 = CGraph(grad_wTest_numerical, seed: Tensor(1))
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
            let pos_wtTest = Variable(Tensor(shape: wtTest.shape, grid: pos_grid))
            let neg_wtTest = Variable(Tensor(shape: wtTest.shape, grid: neg_grid))
            get_grad_wTest_numericals(pos_wtTest, neg_wtTest, idx: i)
        }
//        print("----------")
        let grad_dTest_numerical = (loss(wTest, dTest + epsilon, bTest) - loss(wTest, dTest - epsilon, bTest)) / (Variable(2.0) * epsilon)
        let graph3 = CGraph(grad_dTest_numerical, seed: Tensor(1))
        graph3.fwd()
//        print(grad_dTest_numerical.out ?? "NIL")
        let _ = graph3.bwd()
        func get_grad_dTest_numericals(_ pos_dTest: Variable, _ neg_dTest: Variable, idx: Int) {
            let epsilon = Variable(eps)
            let grad_dTest_numerical = (loss(wTest, pos_dTest, bTest) - loss(wTest, neg_dTest, bTest)) / (Variable(2.0) * epsilon)
            let graph3 = CGraph(grad_dTest_numerical, seed: Tensor(1))
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
            let pos_dtTest = Variable(Tensor(shape: dtTest.shape, grid: pos_grid))
            let neg_dtTest = Variable(Tensor(shape: dtTest.shape, grid: neg_grid))
            get_grad_dTest_numericals(pos_dtTest, neg_dtTest, idx: i)
        }
//        print("----------")
        let grad_bTest_numerical = (loss(wTest, dTest, bTest + epsilon) - loss(wTest, dTest, bTest - epsilon)) / (Variable(2.0) * epsilon)
        let graph4 = CGraph(grad_bTest_numerical, seed: Tensor(1))
        graph4.fwd()
//        print(grad_bTest_numerical.out ?? "NIL")
        let _ = graph4.bwd()
        func get_grad_bTest_numericals(_ pos_bTest: Variable, _ neg_bTest: Variable, idx: Int) {
            let epsilon = Variable(eps)
            let grad_bTest_numerical = (loss(wTest, dTest, pos_bTest) - loss(wTest, dTest, neg_bTest)) / (Variable(2.0) * epsilon)
            let graph4 = CGraph(grad_bTest_numerical, seed: Tensor(1))
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
            let pos_btTest = Variable(Tensor(shape: bt.shape, grid: pos_grid))
            let neg_btTest = Variable(Tensor(shape: bt.shape, grid: neg_grid))
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
        let dt = Tensor([[1, 2, 3], [12, 23, 34], [13, 24, 35], [19, 28, 37]]).transpose()
        session.pass([sequence.input: dt])
        let (out, grads) = session.run(J)
        print(out)
        session.descend(grads: grads, optim: Adam(), lr: 0.3)
    }
    
    func testSigmoid() throws {
        let dt = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        let d = Variable(dt)
        let layer = Sigmoid(d)
        let J = (layer <*> Variable(Tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).transpose())).sum()
        let session = Session(parallel: parallel)
        session.build(J)
        session.pass([layer.input: dt])
        let (out, grads) = session.run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[layer.input] ?? "Nil")
        
        // NOW LETS TEST IF THIS LAYER IS CORRECT BY DOING IT BY 'HAND'
        let dtTest = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        let dTest = Variable(dtTest)
        func loss(_ a: Variable) -> Variable {
            return ((Variable(1.0) / (Variable(1.0) + (Negate(a)).exp())) <*> Variable(Tensor([[1, 2, 3, 4], [5, 6, 7, 8]]).transpose())).sum()
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
        let graph3 = CGraph(grad_dTest_numerical, seed: Tensor(1))
        graph3.fwd()
    //        print(grad_dTest_numerical.out ?? "NIL")
        let _ = graph3.bwd()
        func get_grad_dTest_numericals(_ pos_dTest: Variable, _ neg_dTest: Variable, idx: Int) {
            let epsilon = Variable(eps)
            let grad_dTest_numerical = (loss(pos_dTest) - loss(neg_dTest)) / (Variable(2.0) * epsilon)
            let graph3 = CGraph(grad_dTest_numerical, seed: Tensor(1))
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
            let pos_dtTest = Variable(Tensor(shape: dtTest.shape, grid: pos_grid))
            let neg_dtTest = Variable(Tensor(shape: dtTest.shape, grid: neg_grid))
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
        let dt = Tensor([[1, 2, 3], [12, 23, 34], [13, 24, 35], [19, 28, 37]]).transpose()
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
        let X: Tensor = process.zscore(Tensor(shuffledData), type: .data)
        let Y: Tensor = Tensor(shuffledLabels)
        let optim = Adam()
        // Minibatch
        let b = 100
        // Set up our number of batches
        let batches = Int(ceil(Double(X.shape[0]) / Double(b)))
        // Set up all of our mini batchs in advance
        var mini_batches = [(Tensor, Tensor)](repeating: (Tensor(shape: [], grid: []), Tensor(shape: [], grid: [])), count: batches)
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
                session.pass([sequence.input: XT_mini, expected: YT_mini, avg: Tensor(1.0 / (2.0 * m_mini)), avg_lm: Tensor(1.0 / (2.0 * m_mini))])
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
        session.pass([sequence.input: process.zscore(Tensor(test1, type: .column), type: .pred)])
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
        let X: Tensor = process.zscore(Tensor(shuffledData), type: .data)
        let Y: Tensor = Tensor(shuffledLabels)
        let optim = Adam()
        // Minibatch
        let b = 100
        // Set up our number of batches
        let batches = Int(ceil(Double(X.shape[0]) / Double(b)))
        // Set up all of our mini batchs in advance
        var mini_batches = [(Tensor, Tensor)](repeating: (Tensor(shape: [], grid: []), Tensor(shape: [], grid: [])), count: batches)
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
                session.pass([sequence.input: XT_mini, expected: YT_mini, avg: Tensor(1.0 / (2.0 * m_mini)), avg_lm: Tensor(1.0 / (2.0 * m_mini))])
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
            session.pass([sequence.input: process.zscore(Tensor(test, type: .column), type: .pred)])
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
            if val <= 10.0 {
                data.append([Double.random(in: 0..<10.0)])
            } else if val > 10.0 && val <= 20.0 {
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
        let X: Tensor = process.zscore(Tensor(shuffledData), type: .data)
        let Y: Tensor = Tensor(shuffledLabels)
        let optim = Adam()
        // Minibatch
        let b = 100
        // Set up our number of batches
        let batches = Int(ceil(Double(X.shape[0]) / Double(b)))
        // Set up all of our mini batchs in advance
        var mini_batches = [(Tensor, Tensor)](repeating: (Tensor(shape: [], grid: []), Tensor(shape: [], grid: [])), count: batches)
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
                session.pass([sequence.input: XT_mini, expected: YT_mini, avg: Tensor(1.0 / (m_mini)), avg_lm: Tensor(1.0 / (m_mini))])
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
            session.pass([sequence.input: process.zscore(Tensor(test, type: .column), type: .pred)])
            // Stop forwarding after we have our predicted (other dependencies for J may fire during forward if they have a low dependency count despite not contributing to sequence.predicted sub graph)
            let (out, _) = session.run(J, till: sequence.predicted)
            // Should be class
            print(out.grid, test.first!)
        }
    }
    
    func testConvLayer() throws {
        let dt1 = Tensor([[[15, 24, 13, 44, 52], [12, 24, 35, 64, 85], [51, 22, 73, 94, 55], [11, 52, 36, 47, 59], [41, 62, 83, 94, 75]], [[14, 23, 23, 44, 56], [21, 23, 353, 54, 65], [61, 42, 35, 44, 25], [91, 2, 38, 64, 54], [19, 25, 53, 40, 55]]])
        // let conv = Conv2D(size: [2, 3, 3])
        let conv1 = Conv2D(to: 2, out: 1, size: 3)
        let dt2 = Tensor([[[15, 24, 13, 44, 52], [12, 24, 35, 64, 85], [51, 22, 73, 94, 55], [11, 52, 36, 47, 59], [41, 62, 83, 94, 75]], [[14, 23, 23, 44, 56], [21, 23, 353, 54, 65], [61, 42, 35, 44, 25], [91, 2, 38, 64, 54], [19, 25, 53, 40, 55]]])
        // let conv = Conv2D(size: [2, 2, 3, 3])
        let conv2 = Conv2D(to: 2, out: 2, size: 3)
        let dt3 = Tensor([[15, 24, 13, 44, 52], [12, 24, 35, 64, 85], [51, 22, 73, 94, 55], [11, 52, 36, 47, 59], [41, 62, 83, 94, 75]])
        // let conv = Conv2D(size: [3, 3])
        let conv3 = Conv2D(to: 1, out: 1, size: 3)
        let dt4 = Tensor([[[15, 24, 13, 44, 52], [12, 24, 35, 64, 85], [51, 22, 73, 94, 55], [11, 52, 36, 47, 59], [41, 62, 83, 94, 75]]])
        // let conv = Conv2D(size: [4, 3, 3])
        let conv4 = Conv2D(to: 1, out: 4, size: 3)
        
        let DT = [dt1, dt2, dt3, dt4]
        let CONV = [conv1, conv2, conv3, conv4]
        let SHAPE = [[3, 3], [2, 3, 3], [3, 3], [4, 3, 3]]
        
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
            func loss(_ data: Tensor, _ kernel: Tensor, _ bias: Tensor) -> Double {
                var out: Tensor?
                let (_, data_reshaped) = data.extra()
                let (_, kernel_reshaped) = kernel.extra()
                // Check for multiple kernels
                if kernel_reshaped.count == 4 && data_reshaped.count == 3 && kernel_reshaped[1] == data_reshaped[0] {
                    // Multiple kernels with depth > 1
                    let kernel1 = kernel[t3D: 0]
                    var first = data[mat: 0].conv2D(with: kernel1[mat: 0], type: .valid)
                    let (_, first_reshaped) = first.extra()
                    for m in 1..<kernel_reshaped[1] {
                        first = first + data[mat: m].conv2D(with: kernel1[mat: m], type: .valid)
                    }
                    out = Tensor(shape: [kernel_reshaped[0], first_reshaped[0], first_reshaped[1]], repeating: 0.0)
                    out![mat: 0] = first + bias[0]
                    for d in 1..<kernel_reshaped[0] {
                        let kernelD = kernel[t3D: d]
                        out![mat: d] = data[mat: 0].conv2D(with: kernelD[mat: 0], type: .valid)
                        for m in 1..<kernel_reshaped[1] {
                            out![mat: d] = out![mat: d] + data[mat: m].conv2D(with: kernelD[mat: m], type: .valid)
                        }
                        out![mat: d] = out![mat: d] + bias[d]
                    }
                } else if kernel_reshaped.count == 3 && data_reshaped.count == 3 && kernel_reshaped[0] == data_reshaped[0] {
                    // One kernel with depth > 1
                    // Get the first depths convolution so we can also get the shape
                    out = data[mat: 0].conv2D(with: kernel[mat: 0], type: .valid)
                    for m in 1..<kernel_reshaped[0] {
                        out = out! + data[mat: m].conv2D(with: kernel[mat: m], type: .valid)
                    }
                    out = out! + bias[0]
                } else if kernel_reshaped.count == 3 && data_reshaped.count == 2 {
                    // Multiple kernels with depth 1
                    let first = data.conv2D(with: kernel[mat: 0], type: .valid)
                    let (_, first_reshaped) = first.extra()
                    out = Tensor(shape: [kernel_reshaped[0], first_reshaped[0], first_reshaped[1]], repeating: 0.0)
                    out![mat: 0] = first + bias[0]
                    for m in 1..<kernel_reshaped[0] {
                        out![mat: m] = data.conv2D(with: kernel[mat: m], type: .valid) + bias[m]
                    }
                } else if kernel_reshaped.count == 2 && data_reshaped.count == 2 {
                    // One kernel with depth 1
                    out = data.conv2D(with: kernel, type: .valid) + bias[0]
                } else {
                    fatalError("Data and kernels are incompatible")
                }
                return out!.sum()
            }
            let outTest = loss(dataTest, kernelTest, biasTest)
            // Test data
            func get_grad_data_numericals(_ pos_data: Tensor, _ neg_data: Tensor, idx: Int) {
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
                let pos_data = Tensor(shape: dataTest.shape, grid: pos_grid)
                let neg_data = Tensor(shape: dataTest.shape, grid: neg_grid)
                get_grad_data_numericals(pos_data, neg_data, idx: i)
            }
            // Test kernel
            func get_grad_kernel_numericals(_ pos_kernel: Tensor, _ neg_kernel: Tensor, idx: Int) {
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
                let pos_kernel = Tensor(shape: kernelTest.shape, grid: pos_grid)
                let neg_kernel = Tensor(shape: kernelTest.shape, grid: neg_grid)
                get_grad_kernel_numericals(pos_kernel, neg_kernel, idx: i)
            }
            // Test bias
            func get_grad_bias_numericals(_ pos_bias: Tensor, _ neg_bias: Tensor, idx: Int) {
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
                let pos_bias = Tensor(shape: biasTest.shape, grid: pos_grid)
                let neg_bias = Tensor(shape: biasTest.shape, grid: neg_grid)
                get_grad_bias_numericals(pos_bias, neg_bias, idx: i)
            }
            // Makes sure that our hand computed linear results and our linear layers results are identical
//            print(outTest, out)
            XCTAssert(conv.out!.shape == SHAPE[i], "convolution shape output")
            XCTAssert(abs(outTest - out.grid.first!) < eps, "hand written linear out vs linear layer out")
//            print(grads[conv.input]!.shape, conv.input.out!.shape, grads[conv.kernel]!.shape, conv.kernel.out!.shape, grads[conv.bias]!.shape, conv.bias.out!.shape)
            XCTAssert(grads[conv.input]!.shape == conv.input.out!.shape && grads[conv.kernel]!.shape == conv.kernel.out!.shape && grads[conv.bias]!.shape == conv.bias.out!.shape)
            i += 1
        }
    }
}

/*
 vDSP_imgfir is doing a mathematically correct convolution operation (it rotates kernel before running), WE WANT CROSS CORRELATION (maybe gradients are wrong check kernel rot 180, passing gradient check tho?)
 Convolution layer take entire dataset as input?
 Extra shape calcs for conv2D in tensor kinda funky (see conv_valid,same,full) we remove extra shape before true conv so true convs insert extra shape never actually would insert extra shape
 Tensor struct can now handle extra shapes, make extra() happen in tensor init so we don't constanlty have to call? shouldn't really hinder performance since we call literally everywhere
 SumAxis for 3 >= tensors does not work
 Add custom print for tensor
 Test batchnorm layer is correct with 'hand' calculated gradients (already kinda did this but maybe confirm? maybe also do batchnorm the longer way?)
 t[cols: range] not implemented
 */
