import XCTest
@testable import SML2

final class SML2Tests: XCTestCase {
    
    let parallel = true
    let eps = 0.0000001
    let bound = 0.0001
    
    func test1() throws {
        // SMLGraph currently does not check to see if whole graph is connected since we are only adding nodes by the root
        let a = SMLUnit(4)
        let b = SMLUnit(3)
        // Run through function
        func loss(_ a: SMLUnit, _ b: SMLUnit) -> SMLUnit {
            return a * (a + b)
        }
        let J = loss(a, b)
        // Forward and backward prop wrt J through session
        let (_, grads) = Session(parallel: parallel).run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL")
        // Test for correctness
        let epsilon = SMLUnit(eps)
        let grad_a_numerical = (loss(a + epsilon, b) - loss(a - epsilon, b)) / (SMLUnit(2.0) * epsilon)
        let graph2 = SMLGraph(grad_a_numerical, seed: Tensor(1))
        graph2.fwd()
//        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        let grad_b_numerical = (loss(a, b + epsilon) - loss(a, b - epsilon)) / (SMLUnit(2.0) * epsilon)
        let graph3 = SMLGraph(grad_b_numerical, seed: Tensor(1))
        graph3.fwd()
//        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.bwd()
        
        let diff_a = abs(grads[a]!.grid.first! - grad_a_numerical.out!.grid.first!) / max(abs(grads[a]!.grid.first!), abs(grad_a_numerical.out!.grid.first!))
        let diff_b = abs(grads[b]!.grid.first! - grad_b_numerical.out!.grid.first!) / max(abs(grads[b]!.grid.first!), abs(grad_b_numerical.out!.grid.first!))
        XCTAssert(diff_a < bound, "analytical vs numerical gradient check for a")
        XCTAssert(diff_b < bound, "analytical vs numerical gradient check for b")
    }
    
    func test2() throws {
        // Input nodes
        let a = SMLUnit(230.3)
        let b = SMLUnit(33.2)
        // Run through function
        func loss(_ a: SMLUnit, _ b: SMLUnit) -> SMLUnit {
            return (a / b - a) * (b / a + a + b) * (a - b)
        }
        let J = loss(a, b)
        // Forward and backward prop wrt J through session
        let (_, grads) = Session(parallel: parallel).run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL")
        // Test for correctness
        let epsilon = SMLUnit(eps)
        let grad_a_numerical = (loss(a + epsilon, b) - loss(a - epsilon, b)) / (SMLUnit(2.0) * epsilon)
        let graph2 = SMLGraph(grad_a_numerical, seed: Tensor(1))
        graph2.fwd()
//        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        let grad_b_numerical = (loss(a, b + epsilon) - loss(a, b - epsilon)) / (SMLUnit(2.0) * epsilon)
        let graph3 = SMLGraph(grad_b_numerical, seed: Tensor(1))
        graph3.fwd()
//        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.bwd()
        
        let diff_a = abs(grads[a]!.grid.first! - grad_a_numerical.out!.grid.first!) / max(abs(grads[a]!.grid.first!), abs(grad_a_numerical.out!.grid.first!))
        let diff_b = abs(grads[b]!.grid.first! - grad_b_numerical.out!.grid.first!) / max(abs(grads[b]!.grid.first!), abs(grad_b_numerical.out!.grid.first!))
        XCTAssert(diff_a < bound, "analytical vs numerical gradient check for a")
        XCTAssert(diff_b < bound, "analytical vs numerical gradient check for b")
    }
    
    func test3() throws {
        // Input nodes
        let a = SMLUnit(43)
        let b = SMLUnit(3)
        let c = SMLUnit(2)
        // Run through function
        func loss(_ a: SMLUnit, _ b: SMLUnit, _ c: SMLUnit) -> SMLUnit {
            let f = (a * b).smlsin() + (c - (a / b)).smlexp()
            return (f * f).smllog() * c
        }
        let J = loss(a, b, c)
        // Forward and backward prop wrt J through session
        let (_, grads) = Session(parallel: parallel).run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL", grads[c] ?? "NIL")
        // Test for correctness
        let epsilon = SMLUnit(eps)
        let grad_a_numerical = (loss(a + epsilon, b, c) - loss(a - epsilon, b, c)) / (SMLUnit(2.0) * epsilon)
        let graph2 = SMLGraph(grad_a_numerical, seed: Tensor(1))
        graph2.fwd()
//        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        let grad_b_numerical = (loss(a, b + epsilon, c) - loss(a, b - epsilon, c)) / (SMLUnit(2.0) * epsilon)
        let graph3 = SMLGraph(grad_b_numerical, seed: Tensor(1))
        graph3.fwd()
//        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.bwd()
        let grad_c_numerical = (loss(a, b, c + epsilon) - loss(a, b, c - epsilon)) / (SMLUnit(2.0) * epsilon)
        let graph4 = SMLGraph(grad_c_numerical, seed: Tensor(1))
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
    
    func test4() throws {
        // Input nodes
        let at = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
        let a = SMLUnit(at)
        let bt = Tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        let b = SMLUnit(bt)
        // Run through function
        func loss(_ a: SMLUnit, _ b: SMLUnit) -> SMLUnit {
            return (a <*> b).smlsum()
        }
        let J = loss(a, b)
        // Forward and backward prop wrt J through session
        let (_, grads) = Session(parallel: parallel).run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL")
        // Test for correctness
        let epsilon = SMLUnit(eps)
        let grad_a_numerical = (loss(a + epsilon, b) - loss(a - epsilon, b)) / (SMLUnit(2.0) * epsilon)
        let graph2 = SMLGraph(grad_a_numerical, seed: Tensor(1))
        graph2.fwd()
//        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        func get_grad_a_numericals(_ pos_a: SMLUnit, _ neg_a: SMLUnit, idx: Int) {
            let epsilon = SMLUnit(eps)
            let grad_a_numerical = (loss(pos_a, b) - loss(neg_a, b)) / (SMLUnit(2.0) * epsilon)
            let graph2 = SMLGraph(grad_a_numerical, seed: Tensor(1))
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
            let pos_at = SMLUnit(Tensor(shape: at.shape, grid: pos_grid))
            let neg_at = SMLUnit(Tensor(shape: at.shape, grid: neg_grid))
            get_grad_a_numericals(pos_at, neg_at, idx: i)
        }
//        print("----------")
        let grad_b_numerical = (loss(a, b + epsilon) - loss(a, b - epsilon)) / (SMLUnit(2.0) * epsilon)
        let graph3 = SMLGraph(grad_b_numerical, seed: Tensor(1))
        graph3.fwd()
//        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.bwd()
        func get_grad_b_numericals(_ pos_b: SMLUnit, _ neg_b: SMLUnit, idx: Int) {
            let epsilon = SMLUnit(eps)
            let grad_b_numerical = (loss(a, pos_b) - loss(a, neg_b)) / (SMLUnit(2.0) * epsilon)
            let graph3 = SMLGraph(grad_b_numerical, seed: Tensor(1))
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
            let pos_bt = SMLUnit(Tensor(shape: bt.shape, grid: pos_grid))
            let neg_bt = SMLUnit(Tensor(shape: bt.shape, grid: neg_grid))
            get_grad_b_numericals(pos_bt, neg_bt, idx: i)
        }
    }
    
    func test5() throws {
        // Input nodes
        let at = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
//        let at = Tensor([[1, 2, 3, 4]])
        let a = SMLUnit(at)
        let bt = Tensor([[9, 10, 11, 12]])
//        let bt = Tensor([[9, 10, 11, 12]])
        let b = SMLUnit(bt)
        // Run through function
        func loss(_ a: SMLUnit, _ b: SMLUnit) -> SMLUnit {
            return (a + b - b.smlpow(2) + b.smllog()).smlsum()
        }
        let J = loss(a, b)
        // Forward and backward prop wrt J through session
        let (_, grads) = Session(parallel: parallel).run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL")
        // Test for correctness
        let epsilon = SMLUnit(eps)
        let grad_a_numerical = (loss(a + epsilon, b) - loss(a - epsilon, b)) / (SMLUnit(2.0) * epsilon)
        let graph2 = SMLGraph(grad_a_numerical, seed: Tensor(1))
        graph2.fwd()
//        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        func get_grad_a_numericals(_ pos_a: SMLUnit, _ neg_a: SMLUnit, idx: Int) {
            let epsilon = SMLUnit(eps)
            let grad_a_numerical = (loss(pos_a, b) - loss(neg_a, b)) / (SMLUnit(2.0) * epsilon)
            let graph2 = SMLGraph(grad_a_numerical, seed: Tensor(1))
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
            let pos_at = SMLUnit(Tensor(shape: at.shape, grid: pos_grid))
            let neg_at = SMLUnit(Tensor(shape: at.shape, grid: neg_grid))
            get_grad_a_numericals(pos_at, neg_at, idx: i)
        }
//        print("----------")
        let grad_b_numerical = (loss(a, b + epsilon) - loss(a, b - epsilon)) / (SMLUnit(2.0) * epsilon)
        let graph3 = SMLGraph(grad_b_numerical, seed: Tensor(1))
        graph3.fwd()
//        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.bwd()
        func get_grad_b_numericals(_ pos_b: SMLUnit, _ neg_b: SMLUnit, idx: Int) {
            let epsilon = SMLUnit(eps)
            let grad_b_numerical = (loss(a, pos_b) - loss(a, neg_b)) / (SMLUnit(2.0) * epsilon)
            let graph3 = SMLGraph(grad_b_numerical, seed: Tensor(1))
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
            let pos_bt = SMLUnit(Tensor(shape: bt.shape, grid: pos_grid))
            let neg_bt = SMLUnit(Tensor(shape: bt.shape, grid: neg_grid))
            get_grad_b_numericals(pos_bt, neg_bt, idx: i)
        }
    }
    
    func test6() throws {
        // Input nodes
        let at = Tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
//        let at = Tensor([[1, 2, 3, 4]])
        let a = SMLUnit(at)
        let bt = Tensor([[9, 10, 11, 12]])
//        let bt = Tensor([[9, 10, 11, 12]])
        let b = SMLUnit(bt)
        // Run through function
        func loss(_ a: SMLUnit, _ b: SMLUnit) -> SMLUnit {
            return (a * b * a / b.smlexp()).smlsum()
        }
        let J = loss(a, b)
        // Forward and backward prop wrt J through session
        let (_, grads) = Session(parallel: parallel).run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL")
        // Test for correctness
        let epsilon = SMLUnit(eps)
        let grad_a_numerical = (loss(a + epsilon, b) - loss(a - epsilon, b)) / (SMLUnit(2.0) * epsilon)
        let graph2 = SMLGraph(grad_a_numerical, seed: Tensor(1))
        graph2.fwd()
//        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        func get_grad_a_numericals(_ pos_a: SMLUnit, _ neg_a: SMLUnit, idx: Int) {
            let epsilon = SMLUnit(eps)
            let grad_a_numerical = (loss(pos_a, b) - loss(neg_a, b)) / (SMLUnit(2.0) * epsilon)
            let graph2 = SMLGraph(grad_a_numerical, seed: Tensor(1))
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
            let pos_at = SMLUnit(Tensor(shape: at.shape, grid: pos_grid))
            let neg_at = SMLUnit(Tensor(shape: at.shape, grid: neg_grid))
            get_grad_a_numericals(pos_at, neg_at, idx: i)
        }
//        print("----------")
        let grad_b_numerical = (loss(a, b + epsilon) - loss(a, b - epsilon)) / (SMLUnit(2.0) * epsilon)
        let graph3 = SMLGraph(grad_b_numerical, seed: Tensor(1))
        graph3.fwd()
//        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.bwd()
        func get_grad_b_numericals(_ pos_b: SMLUnit, _ neg_b: SMLUnit, idx: Int) {
            let epsilon = SMLUnit(eps)
            let grad_b_numerical = (loss(a, pos_b) - loss(a, neg_b)) / (SMLUnit(2.0) * epsilon)
            let graph3 = SMLGraph(grad_b_numerical, seed: Tensor(1))
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
            let pos_bt = SMLUnit(Tensor(shape: bt.shape, grid: pos_grid))
            let neg_bt = SMLUnit(Tensor(shape: bt.shape, grid: neg_grid))
            get_grad_b_numericals(pos_bt, neg_bt, idx: i)
        }
    }
    
    func test7() throws {
        // Input nodes
        let a = SMLUnit(230.3)
        let b = SMLUnit(33.2)
        // Run through function
        func loss(_ a: SMLUnit, _ b: SMLUnit) -> SMLUnit {
            return (a / b - a.smlsquare()).smlsquare()
        }
        let J = loss(a, b)
        // Forward and backward prop wrt J through session
        let (_, grads) = Session(parallel: parallel).run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL")
        // Test for correctness
        let epsilon = SMLUnit(eps)
        let grad_a_numerical = (loss(a + epsilon, b) - loss(a - epsilon, b)) / (SMLUnit(2.0) * epsilon)
        let graph2 = SMLGraph(grad_a_numerical, seed: Tensor(1))
        graph2.fwd()
//        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        let grad_b_numerical = (loss(a, b + epsilon) - loss(a, b - epsilon)) / (SMLUnit(2.0) * epsilon)
        let graph3 = SMLGraph(grad_b_numerical, seed: Tensor(1))
        graph3.fwd()
//        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.bwd()
        
        let diff_a = abs(grads[a]!.grid.first! - grad_a_numerical.out!.grid.first!) / max(abs(grads[a]!.grid.first!), abs(grad_a_numerical.out!.grid.first!))
        let diff_b = abs(grads[b]!.grid.first! - grad_b_numerical.out!.grid.first!) / max(abs(grads[b]!.grid.first!), abs(grad_b_numerical.out!.grid.first!))
        XCTAssert(diff_a < bound, "analytical vs numerical gradient check for a")
        XCTAssert(diff_b < bound, "analytical vs numerical gradient check for b")
    }
    
    func test8() throws {
        // Input nodes
        let a = SMLUnit(2.0)
        let b = SMLUnit(11.0)
        // Run through function
        func loss(_ a: SMLUnit, _ b: SMLUnit) -> SMLUnit {
            return a * b.smlpow(2.0)
        }
        let J = loss(a, b)
        // Forward and backward prop wrt J through session
        let (_, grads) = Session(parallel: parallel).run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL")
        // Test for correctness
        let epsilon = SMLUnit(eps)
        let grad_a_numerical = (loss(a + epsilon, b) - loss(a - epsilon, b)) / (SMLUnit(2.0) * epsilon)
        let graph2 = SMLGraph(grad_a_numerical, seed: Tensor(1))
        graph2.fwd()
//        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        let grad_b_numerical = (loss(a, b + epsilon) - loss(a, b - epsilon)) / (SMLUnit(2.0) * epsilon)
        let graph3 = SMLGraph(grad_b_numerical, seed: Tensor(1))
        graph3.fwd()
//        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.bwd()
        
        let diff_a = abs(grads[a]!.grid.first! - grad_a_numerical.out!.grid.first!) / max(abs(grads[a]!.grid.first!), abs(grad_a_numerical.out!.grid.first!))
        let diff_b = abs(grads[b]!.grid.first! - grad_b_numerical.out!.grid.first!) / max(abs(grads[b]!.grid.first!), abs(grad_b_numerical.out!.grid.first!))
        XCTAssert(diff_a < bound, "analytical vs numerical gradient check for a")
        XCTAssert(diff_b < bound, "analytical vs numerical gradient check for b")
    }
    
    // This is the crazy test
    func test9() throws {
        // Input nodes
        let at = Tensor([[1.0, 2], [3.0, 4], [5.0, 6], [7.0, 8]])
        let a = SMLUnit(at)
        let bt = Tensor([[1.0, 2, 3.0, 4], [5.0, 6, 7.0, 8]])
        let b = SMLUnit(bt)
        let ct = Tensor([[1.0, 2], [3.0, 4], [5.0, 6], [7.0, 8]])
        let c = SMLUnit(ct)
        let dt = Tensor([[1], [2]])
        let d = SMLUnit(dt)
        // Run through function
        func loss(_ a: SMLUnit, _ b: SMLUnit, _ c: SMLUnit, _ d: SMLUnit) -> SMLUnit {
            var res = SMLUnit(0)
            for _ in 0..<1000 { res = res + ((a <*> b) <*> (c <*> d)).smlsum() }
            return res
        }
        let J = loss(a, b, c, d)
        // Forward and backward prop wrt J through session
        let (_, grads) = Session(parallel: parallel).run(J)
//        print("----------\n", out)
//        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL", grads[c] ?? "NIL", grads[d] ?? "NIL")
        // Test for correctness
        let epsilon = SMLUnit(eps)
        let grad_a_numerical = (loss(a + epsilon, b, c, d) - loss(a - epsilon, b, c, d)) / (SMLUnit(2.0) * epsilon)
        let graph2 = SMLGraph(grad_a_numerical, seed: Tensor(1))
        graph2.fwd()
//        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.bwd()
        func get_grad_a_numericals(_ pos_a: SMLUnit, _ neg_a: SMLUnit, idx: Int) {
            let epsilon = SMLUnit(eps)
            let grad_a_numerical = (loss(pos_a, b, c, d) - loss(neg_a, b, c, d)) / (SMLUnit(2.0) * epsilon)
            let graph2 = SMLGraph(grad_a_numerical, seed: Tensor(1))
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
            let pos_at = SMLUnit(Tensor(shape: at.shape, grid: pos_grid))
            let neg_at = SMLUnit(Tensor(shape: at.shape, grid: neg_grid))
            get_grad_a_numericals(pos_at, neg_at, idx: i)
        }
//        print("----------")
        let grad_b_numerical = (loss(a, b + epsilon, c, d) - loss(a, b - epsilon, c, d)) / (SMLUnit(2.0) * epsilon)
        let graph3 = SMLGraph(grad_b_numerical, seed: Tensor(1))
        graph3.fwd()
//        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.bwd()
        func get_grad_b_numericals(_ pos_b: SMLUnit, _ neg_b: SMLUnit, idx: Int) {
            let epsilon = SMLUnit(eps)
            let grad_b_numerical = (loss(a, pos_b, c, d) - loss(a, neg_b, c, d)) / (SMLUnit(2.0) * epsilon)
            let graph3 = SMLGraph(grad_b_numerical, seed: Tensor(1))
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
            let pos_bt = SMLUnit(Tensor(shape: bt.shape, grid: pos_grid))
            let neg_bt = SMLUnit(Tensor(shape: bt.shape, grid: neg_grid))
            get_grad_b_numericals(pos_bt, neg_bt, idx: i)
        }
//        print("----------")
        let grad_c_numerical = (loss(a, b, c + epsilon, d) - loss(a, b, c - epsilon, d)) / (SMLUnit(2.0) * epsilon)
        let graph4 = SMLGraph(grad_c_numerical, seed: Tensor(1))
        graph4.fwd()
//        print(grad_c_numerical.out ?? "NIL")
        let _ = graph4.bwd()
        func get_grad_c_numericals(_ pos_c: SMLUnit, _ neg_c: SMLUnit, idx: Int) {
            let epsilon = SMLUnit(eps)
            let grad_c_numerical = (loss(a, b, pos_c, d) - loss(a, b, neg_c, d)) / (SMLUnit(2.0) * epsilon)
            let graph4 = SMLGraph(grad_c_numerical, seed: Tensor(1))
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
            let pos_ct = SMLUnit(Tensor(shape: ct.shape, grid: pos_grid))
            let neg_ct = SMLUnit(Tensor(shape: ct.shape, grid: neg_grid))
            get_grad_c_numericals(pos_ct, neg_ct, idx: i)
        }
        let grad_d_numerical = (loss(a, b, c, d + epsilon) - loss(a, b, c, d - epsilon)) / (SMLUnit(2.0) * epsilon)
        let graph5 = SMLGraph(grad_d_numerical, seed: Tensor(1))
        graph5.fwd()
//        print(grad_d_numerical.out ?? "NIL")
        let _ = graph5.bwd()
        func get_grad_d_numericals(_ pos_d: SMLUnit, _ neg_d: SMLUnit, idx: Int) {
            let epsilon = SMLUnit(eps)
            let grad_d_numerical = (loss(a, b, c, pos_d) - loss(a, b, c, neg_d)) / (SMLUnit(2.0) * epsilon)
            let graph5 = SMLGraph(grad_d_numerical, seed: Tensor(1))
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
            let pos_dt = SMLUnit(Tensor(shape: dt.shape, grid: pos_grid))
            let neg_dt = SMLUnit(Tensor(shape: dt.shape, grid: neg_grid))
            get_grad_d_numericals(pos_dt, neg_dt, idx: i)
        }
    }
}
