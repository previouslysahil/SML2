import XCTest
@testable import SML2

final class SML2Tests: XCTestCase {
    
    let parallel = false
    let eps = 0.0000001
    let bound = 0.0001
    
    func testScalarSimple() throws {
        // CGraph currently does not check to see if whole graph is connected since we are only adding nodes by the root
        let a = Variable<DTensor>(4)
        let b = Variable<DTensor>(3)
        // Run through function
        func loss(_ a: Variable<DTensor>, _ b: Variable<DTensor>) -> Variable<DTensor> {
            return a * (a + b)
        }
        let J = loss(a, b)
        // Forward and backward prop wrt J through session
        let session = Session<Adam<DTensor>>(parallel: parallel)
        session.build(J)
        let (_, grads) = session.run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL")
        // Test for correctness
        let epsilon = Variable<DTensor>(eps)
        let grad_a_numerical = (loss(a + epsilon, b) - loss(a - epsilon, b)) / (Variable(2.0) * epsilon)
        let graph2 = CGraph(grad_a_numerical, seed: DTensor(1))
        graph2.fwd()
//        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        let grad_b_numerical = (loss(a, b + epsilon) - loss(a, b - epsilon)) / (Variable(2.0) * epsilon)
        let graph3 = CGraph(grad_b_numerical, seed: DTensor(1))
        graph3.fwd()
//        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.bwd()

        let diff_a = abs(grads[a]!.grid.first! - grad_a_numerical.out!.grid.first!) / max(abs(grads[a]!.grid.first!), abs(grad_a_numerical.out!.grid.first!))
        let diff_b = abs(grads[b]!.grid.first! - grad_b_numerical.out!.grid.first!) / max(abs(grads[b]!.grid.first!), abs(grad_b_numerical.out!.grid.first!))
        XCTAssert(diff_a < bound, "analytical vs numerical gradient check for a")
        XCTAssert(diff_b < bound, "analytical vs numerical gradient check for b")
    }

    func testScalarMedium() throws {
        // Input nodes
        let a = Variable<DTensor>(230.3)
        let b = Variable<DTensor>(33.2)
        // Run through function
        func loss(_ a: Variable<DTensor>, _ b: Variable<DTensor>) -> Variable<DTensor> {
            return (a / b - a) * (b / a + a + b) * (a - b)
        }
        let J = loss(a, b)
        // Forward and backward prop wrt J through session
        let session = Session<Adam<DTensor>>(parallel: parallel)
        session.build(J)
        let (_, grads) = session.run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL")
        // Test for correctness
        let epsilon = Variable<DTensor>(eps)
        let grad_a_numerical = (loss(a + epsilon, b) - loss(a - epsilon, b)) / (Variable(2.0) * epsilon)
        let graph2 = CGraph(grad_a_numerical, seed: DTensor(1))
        graph2.fwd()
//        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        let grad_b_numerical = (loss(a, b + epsilon) - loss(a, b - epsilon)) / (Variable(2.0) * epsilon)
        let graph3 = CGraph(grad_b_numerical, seed: DTensor(1))
        graph3.fwd()
//        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.bwd()

        let diff_a = abs(grads[a]!.grid.first! - grad_a_numerical.out!.grid.first!) / max(abs(grads[a]!.grid.first!), abs(grad_a_numerical.out!.grid.first!))
        let diff_b = abs(grads[b]!.grid.first! - grad_b_numerical.out!.grid.first!) / max(abs(grads[b]!.grid.first!), abs(grad_b_numerical.out!.grid.first!))
        XCTAssert(diff_a < bound, "analytical vs numerical gradient check for a")
        XCTAssert(diff_b < bound, "analytical vs numerical gradient check for b")
    }

    func testScalarComplex() throws {
        // Input nodes
        let a = Variable<DTensor>(43)
        let b = Variable<DTensor>(3)
        let c = Variable<DTensor>(2)
        // Run through function
        func loss(_ a: Variable<DTensor>, _ b: Variable<DTensor>, _ c: Variable<DTensor>) -> Variable<DTensor> {
            let f = (a * b).sin() + (c - (a / b)).exp()
            return (f * f).log() * c
        }
        let J = loss(a, b, c)
        // Forward and backward prop wrt J through session
        let session = Session<Adam<DTensor>>(parallel: parallel)
        session.build(J)
        let (_, grads) = session.run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL", grads[c] ?? "NIL")
        // Test for correctness
        let epsilon = Variable<DTensor>(eps)
        let grad_a_numerical = (loss(a + epsilon, b, c) - loss(a - epsilon, b, c)) / (Variable(2.0) * epsilon)
        let graph2 = CGraph(grad_a_numerical, seed: DTensor(1))
        graph2.fwd()
//        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        let grad_b_numerical = (loss(a, b + epsilon, c) - loss(a, b - epsilon, c)) / (Variable(2.0) * epsilon)
        let graph3 = CGraph(grad_b_numerical, seed: DTensor(1))
        graph3.fwd()
//        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.bwd()
        let grad_c_numerical = (loss(a, b, c + epsilon) - loss(a, b, c - epsilon)) / (Variable(2.0) * epsilon)
        let graph4 = CGraph(grad_c_numerical, seed: DTensor(1))
        graph4.fwd()
//        print(grad_c_numerical.out ?? "NIL")
        let _ = graph4.bwd()

        let diff_a = abs(grads[a]!.grid.first! - grad_a_numerical.out!.grid.first!) / max(abs(grads[a]!.grid.first!), abs(grad_a_numerical.out!.grid.first!))
        let diff_b = abs(grads[b]!.grid.first! - grad_b_numerical.out!.grid.first!) / max(abs(grads[b]!.grid.first!), abs(grad_b_numerical.out!.grid.first!))
        let diff_c = abs(grads[c]!.grid.first! - grad_c_numerical.out!.grid.first!) / max(abs(grads[c]!.grid.first!), abs(grad_c_numerical.out!.grid.first!))
        XCTAssert(diff_a < bound, "analytical vs numerical gradient check for a")
        XCTAssert(diff_b < bound, "analytical vs numerical gradient check for b")
        XCTAssert(diff_c < bound, "analytical vs numerical gradient check for c")
    }

    func testScalarPow() throws {
        // Input nodes
        let a = Variable<DTensor>(2.0)
        let b = Variable<DTensor>(11.0)
        // Run through function
        func loss(_ a: Variable<DTensor>, _ b: Variable<DTensor>) -> Variable<DTensor> {
            return a * b.pow(2.0)
        }
        let J = loss(a, b)
        // Forward and backward prop wrt J through session
        let session = Session<Adam<DTensor>>(parallel: parallel)
        session.build(J)
        let (_, grads) = session.run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL")
        // Test for correctness
        let epsilon = Variable<DTensor>(eps)
        let grad_a_numerical = (loss(a + epsilon, b) - loss(a - epsilon, b)) / (Variable(2.0) * epsilon)
        let graph2 = CGraph(grad_a_numerical, seed: DTensor(1))
        graph2.fwd()
//        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        let grad_b_numerical = (loss(a, b + epsilon) - loss(a, b - epsilon)) / (Variable(2.0) * epsilon)
        let graph3 = CGraph(grad_b_numerical, seed: DTensor(1))
        graph3.fwd()
//        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.bwd()

        let diff_a = abs(grads[a]!.grid.first! - grad_a_numerical.out!.grid.first!) / max(abs(grads[a]!.grid.first!), abs(grad_a_numerical.out!.grid.first!))
        let diff_b = abs(grads[b]!.grid.first! - grad_b_numerical.out!.grid.first!) / max(abs(grads[b]!.grid.first!), abs(grad_b_numerical.out!.grid.first!))
        XCTAssert(diff_a < bound, "analytical vs numerical gradient check for a")
        XCTAssert(diff_b < bound, "analytical vs numerical gradient check for b")
    }

    func testScalarSquare() throws {
        // Input nodes
        let a = Variable<DTensor>(230.3)
        let b = Variable<DTensor>(33.2)
        // Run through function
        func loss(_ a: Variable<DTensor>, _ b: Variable<DTensor>) -> Variable<DTensor> {
            return (a / b - a.square()).square()
        }
        let J = loss(a, b)
        // Forward and backward prop wrt J through session
        let session = Session<Adam<DTensor>>(parallel: parallel)
        session.build(J)
        let (_, grads) = session.run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL")
        // Test for correctness
        let epsilon = Variable<DTensor>(eps)
        let grad_a_numerical = (loss(a + epsilon, b) - loss(a - epsilon, b)) / (Variable(2.0) * epsilon)
        let graph2 = CGraph(grad_a_numerical, seed: DTensor(1))
        graph2.fwd()
//        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        let grad_b_numerical = (loss(a, b + epsilon) - loss(a, b - epsilon)) / (Variable(2.0) * epsilon)
        let graph3 = CGraph(grad_b_numerical, seed: DTensor(1))
        graph3.fwd()
//        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.bwd()

        let diff_a = abs(grads[a]!.grid.first! - grad_a_numerical.out!.grid.first!) / max(abs(grads[a]!.grid.first!), abs(grad_a_numerical.out!.grid.first!))
        let diff_b = abs(grads[b]!.grid.first! - grad_b_numerical.out!.grid.first!) / max(abs(grads[b]!.grid.first!), abs(grad_b_numerical.out!.grid.first!))
        XCTAssert(diff_a < bound, "analytical vs numerical gradient check for a")
        XCTAssert(diff_b < bound, "analytical vs numerical gradient check for b")
    }

    func testMatMedium() throws {
        // Input nodes
        let at = DTensor([[1, 2, 3, 4], [5, 6, 7, 8]])
//        let at = Tensor([[1, 2, 3, 4]])
        let a = Variable(at)
        let bt = DTensor([[9, 10, 11, 12]])
//        let bt = Tensor([10, 20], type: .column)
//        let bt = Tensor([[9, 10, 11, 12]])
        let b = Variable(bt)
        // Run through function
        func loss(_ a: Variable<DTensor>, _ b: Variable<DTensor>) -> Variable<DTensor> {
            return (a * b * a / b.exp()).sum()
        }
        let J = loss(a, b)
        // Forward and backward prop wrt J through session
        let session = Session<Adam<DTensor>>(parallel: parallel)
        session.build(J)
        let (_, grads) = session.run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL")
        // Test for correctness
        let epsilon = Variable<DTensor>(eps)
        let grad_a_numerical = (loss(a + epsilon, b) - loss(a - epsilon, b)) / (Variable(2.0) * epsilon)
        let graph2 = CGraph(grad_a_numerical, seed: DTensor(1))
        graph2.fwd()
//        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        func get_grad_a_numericals(_ pos_a: Variable<DTensor>, _ neg_a: Variable<DTensor>, idx: Int) {
            let epsilon = Variable<DTensor>(eps)
            let grad_a_numerical = (loss(pos_a, b) - loss(neg_a, b)) / (Variable(2.0) * epsilon)
            let graph2 = CGraph(grad_a_numerical, seed: DTensor(1))
            graph2.fwd()
//            print(grad_a_numerical.out ?? "NIL")
            let _ = graph2.bwd()

            let diff_a = abs(grads[a]!.grid[idx] - grad_a_numerical.out!.grid.first!) / max(abs(grads[a]!.grid[idx]), abs(grad_a_numerical.out!.grid.first!))
            XCTAssert(diff_a < bound, "analytical vs numerical gradient check for a")
        }
        for i in 0..<at.grid.count {
            var pos_grid = at.grid
            pos_grid[i] = pos_grid[i] + eps
            var neg_grid = at.grid
            neg_grid[i] = neg_grid[i] - eps
            let pos_at = Variable(DTensor(shape: at.shape, grid: pos_grid))
            let neg_at = Variable(DTensor(shape: at.shape, grid: neg_grid))
            get_grad_a_numericals(pos_at, neg_at, idx: i)
        }
//        print("----------")
        let grad_b_numerical = (loss(a, b + epsilon) - loss(a, b - epsilon)) / (Variable(2.0) * epsilon)
        let graph3 = CGraph(grad_b_numerical, seed: DTensor(1))
        graph3.fwd()
//        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.bwd()
        func get_grad_b_numericals(_ pos_b: Variable<DTensor>, _ neg_b: Variable<DTensor>, idx: Int) {
            let epsilon = Variable<DTensor>(eps)
            let grad_b_numerical = (loss(a, pos_b) - loss(a, neg_b)) / (Variable(2.0) * epsilon)
            let graph3 = CGraph(grad_b_numerical, seed: DTensor(1))
            graph3.fwd()
//            print(grad_b_numerical.out ?? "NIL")
            let _ = graph3.bwd()

            let diff_b = abs(grads[b]!.grid[idx] - grad_b_numerical.out!.grid.first!) / max(abs(grads[b]!.grid[idx]), abs(grad_b_numerical.out!.grid.first!))
            XCTAssert(diff_b < bound, "analytical vs numerical gradient check for b")
        }
        for i in 0..<bt.grid.count {
            var pos_grid = bt.grid
            pos_grid[i] = pos_grid[i] + eps
            var neg_grid = bt.grid
            neg_grid[i] = neg_grid[i] - eps
            let pos_bt = Variable(DTensor(shape: bt.shape, grid: pos_grid))
            let neg_bt = Variable(DTensor(shape: bt.shape, grid: neg_grid))
            get_grad_b_numericals(pos_bt, neg_bt, idx: i)
        }
    }

    func testMatComplex() throws {
        // Input nodes
        let at = DTensor([[1, 2, 3, 4], [5, 6, 7, 8]])
//        let at = Tensor([[1, 2, 3, 4]])
        let a = Variable(at)
//        let bt = Tensor([[9, 10, 11, 12]])
        let bt = DTensor([10, 20], type: .column)
//        let bt = Tensor([[9, 10, 11, 12]])
        let b = Variable(bt)
        // Run through function
        func loss(_ a: Variable<DTensor>, _ b: Variable<DTensor>) -> Variable<DTensor> {
            return (a + b - b.pow(2) + b.log()).sum()
        }
        let J = loss(a, b)
        // Forward and backward prop wrt J through session
        let session = Session<Adam<DTensor>>(parallel: parallel)
        session.build(J)
        let (_, grads) = session.run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL")
        // Test for correctness
        let epsilon = Variable<DTensor>(eps)
        let grad_a_numerical = (loss(a + epsilon, b) - loss(a - epsilon, b)) / (Variable(2.0) * epsilon)
        let graph2 = CGraph(grad_a_numerical, seed: DTensor(1))
        graph2.fwd()
//        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        func get_grad_a_numericals(_ pos_a: Variable<DTensor>, _ neg_a: Variable<DTensor>, idx: Int) {
            let epsilon = Variable<DTensor>(eps)
            let grad_a_numerical = (loss(pos_a, b) - loss(neg_a, b)) / (Variable(2.0) * epsilon)
            let graph2 = CGraph(grad_a_numerical, seed: DTensor(1))
            graph2.fwd()
//            print(grad_a_numerical.out ?? "NIL")
            let _ = graph2.bwd()

            let diff_a = abs(grads[a]!.grid[idx] - grad_a_numerical.out!.grid.first!) / max(abs(grads[a]!.grid[idx]), abs(grad_a_numerical.out!.grid.first!))
            XCTAssert(diff_a < bound, "analytical vs numerical gradient check for a")
        }
        for i in 0..<at.grid.count {
            var pos_grid = at.grid
            pos_grid[i] = pos_grid[i] + eps
            var neg_grid = at.grid
            neg_grid[i] = neg_grid[i] - eps
            let pos_at = Variable(DTensor(shape: at.shape, grid: pos_grid))
            let neg_at = Variable(DTensor(shape: at.shape, grid: neg_grid))
            get_grad_a_numericals(pos_at, neg_at, idx: i)
        }
//        print("----------")
        let grad_b_numerical = (loss(a, b + epsilon) - loss(a, b - epsilon)) / (Variable(2.0) * epsilon)
        let graph3 = CGraph(grad_b_numerical, seed: DTensor(1))
        graph3.fwd()
//        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.bwd()
        func get_grad_b_numericals(_ pos_b: Variable<DTensor>, _ neg_b: Variable<DTensor>, idx: Int) {
            let epsilon = Variable<DTensor>(eps)
            let grad_b_numerical = (loss(a, pos_b) - loss(a, neg_b)) / (Variable(2.0) * epsilon)
            let graph3 = CGraph(grad_b_numerical, seed: DTensor(1))
            graph3.fwd()
//            print(grad_b_numerical.out ?? "NIL")
            let _ = graph3.bwd()

            let diff_b = abs(grads[b]!.grid[idx] - grad_b_numerical.out!.grid.first!) / max(abs(grads[b]!.grid[idx]), abs(grad_b_numerical.out!.grid.first!))
            XCTAssert(diff_b < bound, "analytical vs numerical gradient check for b")
        }
        for i in 0..<bt.grid.count {
            var pos_grid = bt.grid
            pos_grid[i] = pos_grid[i] + eps
            var neg_grid = bt.grid
            neg_grid[i] = neg_grid[i] - eps
            let pos_bt = Variable(DTensor(shape: bt.shape, grid: pos_grid))
            let neg_bt = Variable(DTensor(shape: bt.shape, grid: neg_grid))
            get_grad_b_numericals(pos_bt, neg_bt, idx: i)
        }
    }

    func testMatMulSimple() throws {
        // Input nodes
        let at = DTensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        let a = Variable(at)
        let bt = DTensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        let b = Variable(bt)
        // Run through function
        func loss(_ a: Variable<DTensor>, _ b: Variable<DTensor>) -> Variable<DTensor> {
            return (a <*> b).sum()
        }
        let J = loss(a, b)
        // Forward and backward prop wrt J through session
        let session = Session<Adam<DTensor>>(parallel: parallel)
        session.build(J)
        let (_, grads) = session.run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL")
        // Test for correctness
        let epsilon = Variable<DTensor>(eps)
        let grad_a_numerical = (loss(a + epsilon, b) - loss(a - epsilon, b)) / (Variable(2.0) * epsilon)
        let graph2 = CGraph(grad_a_numerical, seed: DTensor(1))
        graph2.fwd()
//        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        func get_grad_a_numericals(_ pos_a: Variable<DTensor>, _ neg_a: Variable<DTensor>, idx: Int) {
            let epsilon = Variable<DTensor>(eps)
            let grad_a_numerical = (loss(pos_a, b) - loss(neg_a, b)) / (Variable(2.0) * epsilon)
            let graph2 = CGraph(grad_a_numerical, seed: DTensor(1))
            graph2.fwd()
//            print(grad_a_numerical.out ?? "NIL")
            let _ = graph2.bwd()

            let diff_a = abs(grads[a]!.grid[idx] - grad_a_numerical.out!.grid.first!) / max(abs(grads[a]!.grid[idx]), abs(grad_a_numerical.out!.grid.first!))
            XCTAssert(diff_a < bound, "analytical vs numerical gradient check for a")
        }
        for i in 0..<at.grid.count {
            var pos_grid = at.grid
            pos_grid[i] = pos_grid[i] + eps
            var neg_grid = at.grid
            neg_grid[i] = neg_grid[i] - eps
            let pos_at = Variable(DTensor(shape: at.shape, grid: pos_grid))
            let neg_at = Variable(DTensor(shape: at.shape, grid: neg_grid))
            get_grad_a_numericals(pos_at, neg_at, idx: i)
        }
//        print("----------")
        let grad_b_numerical = (loss(a, b + epsilon) - loss(a, b - epsilon)) / (Variable(2.0) * epsilon)
        let graph3 = CGraph(grad_b_numerical, seed: DTensor(1))
        graph3.fwd()
//        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.bwd()
        func get_grad_b_numericals(_ pos_b: Variable<DTensor>, _ neg_b: Variable<DTensor>, idx: Int) {
            let epsilon = Variable<DTensor>(eps)
            let grad_b_numerical = (loss(a, pos_b) - loss(a, neg_b)) / (Variable(2.0) * epsilon)
            let graph3 = CGraph(grad_b_numerical, seed: DTensor(1))
            graph3.fwd()
//            print(grad_b_numerical.out ?? "NIL")
            let _ = graph3.bwd()

            let diff_b = abs(grads[b]!.grid[idx] - grad_b_numerical.out!.grid.first!) / max(abs(grads[b]!.grid[idx]), abs(grad_b_numerical.out!.grid.first!))
            XCTAssert(diff_b < bound, "analytical vs numerical gradient check for b")
        }
        for i in 0..<bt.grid.count {
            var pos_grid = bt.grid
            pos_grid[i] = pos_grid[i] + eps
            var neg_grid = bt.grid
            neg_grid[i] = neg_grid[i] - eps
            let pos_bt = Variable(DTensor(shape: bt.shape, grid: pos_grid))
            let neg_bt = Variable(DTensor(shape: bt.shape, grid: neg_grid))
            get_grad_b_numericals(pos_bt, neg_bt, idx: i)
        }
    }

    func testMatMulSimpleSumDiag() throws {
        // Input nodes
        let at = DTensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        let a = Variable(at)
        let bt = DTensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        let b = Variable(bt)
        // Run through function
        func loss(_ a: Variable<DTensor>, _ b: Variable<DTensor>) -> Variable<DTensor> {
            return (a <*> b).sumDiag()
        }
        let J = loss(a, b)
        // Forward and backward prop wrt J through session
        let session = Session<Adam<DTensor>>(parallel: parallel)
        session.build(J)
        let (_, grads) = session.run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL")
        // Test for correctness
        let epsilon = Variable<DTensor>(eps)
        let grad_a_numerical = (loss(a + epsilon, b) - loss(a - epsilon, b)) / (Variable(2.0) * epsilon)
        let graph2 = CGraph(grad_a_numerical, seed: DTensor(1))
        graph2.fwd()
//        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        func get_grad_a_numericals(_ pos_a: Variable<DTensor>, _ neg_a: Variable<DTensor>, idx: Int) {
            let epsilon = Variable<DTensor>(eps)
            let grad_a_numerical = (loss(pos_a, b) - loss(neg_a, b)) / (Variable(2.0) * epsilon)
            let graph2 = CGraph(grad_a_numerical, seed: DTensor(1))
            graph2.fwd()
//            print(grad_a_numerical.out ?? "NIL")
            let _ = graph2.bwd()

            let diff_a = abs(grads[a]!.grid[idx] - grad_a_numerical.out!.grid.first!) / max(abs(grads[a]!.grid[idx]), abs(grad_a_numerical.out!.grid.first!))
            XCTAssert(diff_a < bound, "analytical vs numerical gradient check for a")
        }
        for i in 0..<at.grid.count {
            var pos_grid = at.grid
            pos_grid[i] = pos_grid[i] + eps
            var neg_grid = at.grid
            neg_grid[i] = neg_grid[i] - eps
            let pos_at = Variable(DTensor(shape: at.shape, grid: pos_grid))
            let neg_at = Variable(DTensor(shape: at.shape, grid: neg_grid))
            get_grad_a_numericals(pos_at, neg_at, idx: i)
        }
//        print("----------")
        let grad_b_numerical = (loss(a, b + epsilon) - loss(a, b - epsilon)) / (Variable(2.0) * epsilon)
        let graph3 = CGraph(grad_b_numerical, seed: DTensor(1))
        graph3.fwd()
//        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.bwd()
        func get_grad_b_numericals(_ pos_b: Variable<DTensor>, _ neg_b: Variable<DTensor>, idx: Int) {
            let epsilon = Variable<DTensor>(eps)
            let grad_b_numerical = (loss(a, pos_b) - loss(a, neg_b)) / (Variable(2.0) * epsilon)
            let graph3 = CGraph(grad_b_numerical, seed: DTensor(1))
            graph3.fwd()
//            print(grad_b_numerical.out ?? "NIL")
            let _ = graph3.bwd()

            let diff_b = abs(grads[b]!.grid[idx] - grad_b_numerical.out!.grid.first!) / max(abs(grads[b]!.grid[idx]), abs(grad_b_numerical.out!.grid.first!))
            XCTAssert(diff_b < bound, "analytical vs numerical gradient check for b")
        }
        for i in 0..<bt.grid.count {
            var pos_grid = bt.grid
            pos_grid[i] = pos_grid[i] + eps
            var neg_grid = bt.grid
            neg_grid[i] = neg_grid[i] - eps
            let pos_bt = Variable(DTensor(shape: bt.shape, grid: pos_grid))
            let neg_bt = Variable(DTensor(shape: bt.shape, grid: neg_grid))
            get_grad_b_numericals(pos_bt, neg_bt, idx: i)
        }
    }

    func testMatMulMedium() throws {
        // Input nodes
        let at = DTensor([[1.0, 2], [3.0, 4], [5.0, 6], [7.0, 8]])
        let a = Variable(at)
        let bt = DTensor([[1.0, 2, 3.0, 4], [5.0, 6, 7.0, 8]])
        let b = Variable(bt)
        let ct = DTensor([[1.0, 2], [3.0, 4], [5.0, 6], [7.0, 8]])
//        let c = Variable(ct)
        let c = Placeholder<DTensor>()
        let dt = DTensor([[1], [2]])
//        let d = Variable(dt)
        let d = Placeholder<DTensor>()
        // Run through function
        func loss(_ a: Variable<DTensor>, _ b: Variable<DTensor>, _ c: Variable<DTensor>, _ d: Variable<DTensor>) -> Variable<DTensor> {
            return ((a <*> b) <*> (c <*> d)).sum()
        }
        let J = loss(a, b, c, d)
        // Forward and backward prop wrt J through session
        let session = Session<Adam<DTensor>>(parallel: parallel)
        session.build(J)
        // pass the values for our placeholders declared above before running
        var data = [Placeholder<DTensor>: DTensor]()
        data[c] = ct
        data[d] = dt
        session.pass(data)
        let (_, grads) = session.run(J)
//        session.descend(grads: grads, optim: Adam(b1: 0.9, b2: 0.99), lr: 0.1)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL", grads[c] ?? "NIL", grads[d] ?? "NIL")
        // Test for correctness
        let epsilon = Variable<DTensor>(eps)
        let grad_a_numerical = (loss(a + epsilon, b, c, d) - loss(a - epsilon, b, c, d)) / (Variable(2.0) * epsilon)
        let graph2 = CGraph(grad_a_numerical, seed: DTensor(1))
        graph2.fwd()
//        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        func get_grad_a_numericals(_ pos_a: Variable<DTensor>, _ neg_a: Variable<DTensor>, idx: Int) {
            let epsilon = Variable<DTensor>(eps)
            let grad_a_numerical = (loss(pos_a, b, c, d) - loss(neg_a, b, c, d)) / (Variable(2.0) * epsilon)
            let graph2 = CGraph(grad_a_numerical, seed: DTensor(1))
            graph2.fwd()
//            print(grad_a_numerical.out ?? "NIL")
            let _ = graph2.bwd()

            let diff_a = abs(grads[a]!.grid[idx] - grad_a_numerical.out!.grid.first!) / max(abs(grads[a]!.grid[idx]), abs(grad_a_numerical.out!.grid.first!))
            XCTAssert(diff_a < bound, "analytical vs numerical gradient check for a")
        }
        for i in 0..<at.grid.count {
            var pos_grid = at.grid
            pos_grid[i] = pos_grid[i] + eps
            var neg_grid = at.grid
            neg_grid[i] = neg_grid[i] - eps
            let pos_at = Variable(DTensor(shape: at.shape, grid: pos_grid))
            let neg_at = Variable(DTensor(shape: at.shape, grid: neg_grid))
            get_grad_a_numericals(pos_at, neg_at, idx: i)
        }
//        print("----------")
        let grad_b_numerical = (loss(a, b + epsilon, c, d) - loss(a, b - epsilon, c, d)) / (Variable(2.0) * epsilon)
        let graph3 = CGraph(grad_b_numerical, seed: DTensor(1))
        graph3.fwd()
//        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.bwd()
        func get_grad_b_numericals(_ pos_b: Variable<DTensor>, _ neg_b: Variable<DTensor>, idx: Int) {
            let epsilon = Variable<DTensor>(eps)
            let grad_b_numerical = (loss(a, pos_b, c, d) - loss(a, neg_b, c, d)) / (Variable(2.0) * epsilon)
            let graph3 = CGraph(grad_b_numerical, seed: DTensor(1))
            graph3.fwd()
//            print(grad_b_numerical.out ?? "NIL")
            let _ = graph3.bwd()

            let diff_b = abs(grads[b]!.grid[idx] - grad_b_numerical.out!.grid.first!) / max(abs(grads[b]!.grid[idx]), abs(grad_b_numerical.out!.grid.first!))
            XCTAssert(diff_b < bound, "analytical vs numerical gradient check for b")
        }
        for i in 0..<bt.grid.count {
            var pos_grid = bt.grid
            pos_grid[i] = pos_grid[i] + eps
            var neg_grid = bt.grid
            neg_grid[i] = neg_grid[i] - eps
            let pos_bt = Variable(DTensor(shape: bt.shape, grid: pos_grid))
            let neg_bt = Variable(DTensor(shape: bt.shape, grid: neg_grid))
            get_grad_b_numericals(pos_bt, neg_bt, idx: i)
        }
//        print("----------")
        let grad_c_numerical = (loss(a, b, c + epsilon, d) - loss(a, b, c - epsilon, d)) / (Variable(2.0) * epsilon)
        let graph4 = CGraph(grad_c_numerical, seed: DTensor(1))
        graph4.fwd()
//        print(grad_c_numerical.out ?? "NIL")
        let _ = graph4.bwd()
        func get_grad_c_numericals(_ pos_c: Variable<DTensor>, _ neg_c: Variable<DTensor>, idx: Int) {
            let epsilon = Variable<DTensor>(eps)
            let grad_c_numerical = (loss(a, b, pos_c, d) - loss(a, b, neg_c, d)) / (Variable(2.0) * epsilon)
            let graph4 = CGraph(grad_c_numerical, seed: DTensor(1))
            graph4.fwd()
//            print(grad_c_numerical.out ?? "NIL")
            let _ = graph4.bwd()

            let diff_c = abs(grads[c]!.grid[idx] - grad_c_numerical.out!.grid.first!) / max(abs(grads[c]!.grid[idx]), abs(grad_c_numerical.out!.grid.first!))
            XCTAssert(diff_c < bound, "analytical vs numerical gradient check for c")
        }
        for i in 0..<ct.grid.count {
            var pos_grid = ct.grid
            pos_grid[i] = pos_grid[i] + eps
            var neg_grid = ct.grid
            neg_grid[i] = neg_grid[i] - eps
            let pos_ct = Variable(DTensor(shape: ct.shape, grid: pos_grid))
            let neg_ct = Variable(DTensor(shape: ct.shape, grid: neg_grid))
            get_grad_c_numericals(pos_ct, neg_ct, idx: i)
        }
        let grad_d_numerical = (loss(a, b, c, d + epsilon) - loss(a, b, c, d - epsilon)) / (Variable(2.0) * epsilon)
        let graph5 = CGraph(grad_d_numerical, seed: DTensor(1))
        graph5.fwd()
//        print(grad_d_numerical.out ?? "NIL")
        let _ = graph5.bwd()
        func get_grad_d_numericals(_ pos_d: Variable<DTensor>, _ neg_d: Variable<DTensor>, idx: Int) {
            let epsilon = Variable<DTensor>(eps)
            let grad_d_numerical = (loss(a, b, c, pos_d) - loss(a, b, c, neg_d)) / (Variable(2.0) * epsilon)
            let graph5 = CGraph(grad_d_numerical, seed: DTensor(1))
            graph5.fwd()
//            print(grad_d_numerical.out ?? "NIL")
            let _ = graph5.bwd()

            let diff_d = abs(grads[d]!.grid[idx] - grad_d_numerical.out!.grid.first!) / max(abs(grads[d]!.grid[idx]), abs(grad_d_numerical.out!.grid.first!))
            XCTAssert(diff_d < bound, "analytical vs numerical gradient check for d")
        }
        for i in 0..<dt.grid.count {
            var pos_grid = dt.grid
            pos_grid[i] = pos_grid[i] + eps
            var neg_grid = dt.grid
            neg_grid[i] = neg_grid[i] - eps
            let pos_dt = Variable(DTensor(shape: dt.shape, grid: pos_grid))
            let neg_dt = Variable(DTensor(shape: dt.shape, grid: neg_grid))
            get_grad_d_numericals(pos_dt, neg_dt, idx: i)
        }
    }

    // This is the crazy test
    func testMatMulComplex() throws {
        // Input nodes
        let at = DTensor([[1.0, 2], [3.0, 4], [5.0, 6], [7.0, 8]])
        let a = Variable(at)
        let bt = DTensor([[1.0, 2, 3.0, 4], [5.0, 6, 7.0, 8]])
        let b = Variable(bt)
        let ct = DTensor([[1.0, 2], [3.0, 4], [5.0, 6], [7.0, 8]])
        let c = Variable(ct)
        let dt = DTensor([[1], [2]])
        let d = Variable(dt)
        // Run through function
        func loss(_ a: Variable<DTensor>, _ b: Variable<DTensor>, _ c: Variable<DTensor>, _ d: Variable<DTensor>) -> Variable<DTensor> {
            var res = Variable<DTensor>(0)
            for _ in 0..<1000 { res = res + ((a <*> b) <*> (c <*> d)).sum() }
            return res
        }
        let J = loss(a, b, c, d)
        // Forward and backward prop wrt J through session
        let session = Session<Adam<DTensor>>(parallel: parallel)
        session.build(J)
        let (_, grads) = session.run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL", grads[c] ?? "NIL", grads[d] ?? "NIL")
        // Test for correctness
        let epsilon = Variable<DTensor>(eps)
        let grad_a_numerical = (loss(a + epsilon, b, c, d) - loss(a - epsilon, b, c, d)) / (Variable(2.0) * epsilon)
        let graph2 = CGraph(grad_a_numerical, seed: DTensor(1))
        graph2.fwd()
//        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        func get_grad_a_numericals(_ pos_a: Variable<DTensor>, _ neg_a: Variable<DTensor>, idx: Int) {
            let epsilon = Variable<DTensor>(eps)
            let grad_a_numerical = (loss(pos_a, b, c, d) - loss(neg_a, b, c, d)) / (Variable(2.0) * epsilon)
            let graph2 = CGraph(grad_a_numerical, seed: DTensor(1))
            graph2.fwd()
//            print(grad_a_numerical.out ?? "NIL")
            let _ = graph2.bwd()

            let diff_a = abs(grads[a]!.grid[idx] - grad_a_numerical.out!.grid.first!) / max(abs(grads[a]!.grid[idx]), abs(grad_a_numerical.out!.grid.first!))
            XCTAssert(diff_a < bound, "analytical vs numerical gradient check for a")
        }
        for i in 0..<at.grid.count {
            var pos_grid = at.grid
            pos_grid[i] = pos_grid[i] + eps
            var neg_grid = at.grid
            neg_grid[i] = neg_grid[i] - eps
            let pos_at = Variable(DTensor(shape: at.shape, grid: pos_grid))
            let neg_at = Variable(DTensor(shape: at.shape, grid: neg_grid))
            get_grad_a_numericals(pos_at, neg_at, idx: i)
        }
//        print("----------")
        let grad_b_numerical = (loss(a, b + epsilon, c, d) - loss(a, b - epsilon, c, d)) / (Variable(2.0) * epsilon)
        let graph3 = CGraph(grad_b_numerical, seed: DTensor(1))
        graph3.fwd()
//        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.bwd()
        func get_grad_b_numericals(_ pos_b: Variable<DTensor>, _ neg_b: Variable<DTensor>, idx: Int) {
            let epsilon = Variable<DTensor>(eps)
            let grad_b_numerical = (loss(a, pos_b, c, d) - loss(a, neg_b, c, d)) / (Variable(2.0) * epsilon)
            let graph3 = CGraph(grad_b_numerical, seed: DTensor(1))
            graph3.fwd()
//            print(grad_b_numerical.out ?? "NIL")
            let _ = graph3.bwd()

            let diff_b = abs(grads[b]!.grid[idx] - grad_b_numerical.out!.grid.first!) / max(abs(grads[b]!.grid[idx]), abs(grad_b_numerical.out!.grid.first!))
            XCTAssert(diff_b < bound, "analytical vs numerical gradient check for b")
        }
        for i in 0..<bt.grid.count {
            var pos_grid = bt.grid
            pos_grid[i] = pos_grid[i] + eps
            var neg_grid = bt.grid
            neg_grid[i] = neg_grid[i] - eps
            let pos_bt = Variable(DTensor(shape: bt.shape, grid: pos_grid))
            let neg_bt = Variable(DTensor(shape: bt.shape, grid: neg_grid))
            get_grad_b_numericals(pos_bt, neg_bt, idx: i)
        }
//        print("----------")
        let grad_c_numerical = (loss(a, b, c + epsilon, d) - loss(a, b, c - epsilon, d)) / (Variable(2.0) * epsilon)
        let graph4 = CGraph(grad_c_numerical, seed: DTensor(1))
        graph4.fwd()
//        print(grad_c_numerical.out ?? "NIL")
        let _ = graph4.bwd()
        func get_grad_c_numericals(_ pos_c: Variable<DTensor>, _ neg_c: Variable<DTensor>, idx: Int) {
            let epsilon = Variable<DTensor>(eps)
            let grad_c_numerical = (loss(a, b, pos_c, d) - loss(a, b, neg_c, d)) / (Variable(2.0) * epsilon)
            let graph4 = CGraph(grad_c_numerical, seed: DTensor(1))
            graph4.fwd()
//            print(grad_c_numerical.out ?? "NIL")
            let _ = graph4.bwd()

            let diff_c = abs(grads[c]!.grid[idx] - grad_c_numerical.out!.grid.first!) / max(abs(grads[c]!.grid[idx]), abs(grad_c_numerical.out!.grid.first!))
            XCTAssert(diff_c < bound, "analytical vs numerical gradient check for c")
        }
        for i in 0..<ct.grid.count {
            var pos_grid = ct.grid
            pos_grid[i] = pos_grid[i] + eps
            var neg_grid = ct.grid
            neg_grid[i] = neg_grid[i] - eps
            let pos_ct = Variable(DTensor(shape: ct.shape, grid: pos_grid))
            let neg_ct = Variable(DTensor(shape: ct.shape, grid: neg_grid))
            get_grad_c_numericals(pos_ct, neg_ct, idx: i)
        }
        let grad_d_numerical = (loss(a, b, c, d + epsilon) - loss(a, b, c, d - epsilon)) / (Variable(2.0) * epsilon)
        let graph5 = CGraph(grad_d_numerical, seed: DTensor(1))
        graph5.fwd()
//        print(grad_d_numerical.out ?? "NIL")
        let _ = graph5.bwd()
        func get_grad_d_numericals(_ pos_d: Variable<DTensor>, _ neg_d: Variable<DTensor>, idx: Int) {
            let epsilon = Variable<DTensor>(eps)
            let grad_d_numerical = (loss(a, b, c, pos_d) - loss(a, b, c, neg_d)) / (Variable(2.0) * epsilon)
            let graph5 = CGraph(grad_d_numerical, seed: DTensor(1))
            graph5.fwd()
//            print(grad_d_numerical.out ?? "NIL")
            let _ = graph5.bwd()

            let diff_d = abs(grads[d]!.grid[idx] - grad_d_numerical.out!.grid.first!) / max(abs(grads[d]!.grid[idx]), abs(grad_d_numerical.out!.grid.first!))
            XCTAssert(diff_d < bound, "analytical vs numerical gradient check for d")
        }
        for i in 0..<dt.grid.count {
            var pos_grid = dt.grid
            pos_grid[i] = pos_grid[i] + eps
            var neg_grid = dt.grid
            neg_grid[i] = neg_grid[i] - eps
            let pos_dt = Variable(DTensor(shape: dt.shape, grid: pos_grid))
            let neg_dt = Variable(DTensor(shape: dt.shape, grid: neg_grid))
            get_grad_d_numericals(pos_dt, neg_dt, idx: i)
        }
    }
}
