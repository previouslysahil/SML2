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
        let bt = Tensor(shape: [4, 1], repeating: 0)
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
}

/*
 Tensor struct can now handle extra shapes
 SumAxis for 3 >= tensors does not work
 Add custom print for tensor
 Test batchnorm layer is correct with 'hand' calculated gradients (already kinda did this but maybe confirm? maybe also do batchnorm the longer way?)
 t[cols: range] not implemented
 */
