import XCTest
@testable import SML2

final class SML2Tests: XCTestCase {
    
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
        let (out, grads) = Session().run(J)
        print("----------\n", out)
        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL")
        // Test for correctness
        let epsilon = SMLUnit(0.00000001)
        let grad_a_numerical = (loss(a + epsilon, b) - loss(a - epsilon, b)) / (SMLUnit(2.0) * epsilon)
        let graph2 = SMLGraph(grad_a_numerical)
        graph2.forward()
        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.backward()
        let grad_b_numerical = (loss(a, b + epsilon) - loss(a, b - epsilon)) / (SMLUnit(2.0) * epsilon)
        let graph3 = SMLGraph(grad_b_numerical)
        graph3.forward()
        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.backward()
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
        let (out, grads) = Session().run(J)
        print("----------\n", out)
        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL")
        // Test for correctness
        let epsilon = SMLUnit(0.00000001)
        let grad_a_numerical = (loss(a + epsilon, b) - loss(a - epsilon, b)) / (SMLUnit(2.0) * epsilon)
        let graph2 = SMLGraph(grad_a_numerical)
        graph2.forward()
        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.backward()
        let grad_b_numerical = (loss(a, b + epsilon) - loss(a, b - epsilon)) / (SMLUnit(2.0) * epsilon)
        let graph3 = SMLGraph(grad_b_numerical)
        graph3.forward()
        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.backward()
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
        let (out, grads) = Session().run(J)
        print("----------\n", out)
        print(grads[J] ?? "NIL", grads[a] ?? "NIL", grads[b] ?? "NIL", grads[c] ?? "NIL")
        // Test for correctness
        let epsilon = SMLUnit(0.00000001)
        let grad_a_numerical = (loss(a + epsilon, b, c) - loss(a - epsilon, b, c)) / (SMLUnit(2.0) * epsilon)
        let graph2 = SMLGraph(grad_a_numerical)
        graph2.forward()
        print(grad_a_numerical.out ?? "NIL")
        let _ = graph2.backward()
        let grad_b_numerical = (loss(a, b + epsilon, c) - loss(a, b - epsilon, c)) / (SMLUnit(2.0) * epsilon)
        let graph3 = SMLGraph(grad_b_numerical)
        graph3.forward()
        print(grad_b_numerical.out ?? "NIL")
        let _ = graph3.backward()
        let grad_c_numerical = (loss(a, b, c + epsilon) - loss(a, b, c - epsilon)) / (SMLUnit(2.0) * epsilon)
        let graph4 = SMLGraph(grad_c_numerical)
        graph4.forward()
        print(grad_c_numerical.out ?? "NIL")
        let _ = graph4.backward()
    }
}
