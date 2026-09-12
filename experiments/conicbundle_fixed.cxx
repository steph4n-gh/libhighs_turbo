// Research driver for the unmodified ConicBundle 1.a.2 library.
// Fixed, externally supplied cuts: no cut-selection or incumbent cost charged.
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "MatrixCBSolver.hxx"
#include "PSCAffineFunction.hxx"
#include "CMsymsparse.hxx"
#include "CMsingleton.hxx"
#include "PSCPrimal.hxx"
using namespace CH_Matrix_Classes;
using namespace ConicBundle;
int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) return 2;
  auto started = std::chrono::steady_clock::now();
  std::ifstream input(argv[1]);
  Integer n, m, k;
  input >> n >> m >> k;
  Indexmatrix u(m, 1, Integer(0)), v(m, 1, Integer(0));
  Matrix coefficients(m, 1, 0.);
  for (Integer i=0; i<m; ++i) {
    Real weight;
    input >> u(i) >> v(i) >> weight;
    coefficients(i) = -weight / 2.;
  }
  Sparsesym objective(n, m, u, v, coefficients);
  Indexmatrix dimensions(1, 1, n);
  SparseCoeffmatMatrix C(dimensions, 1), At(dimensions, n+k);
  C.set(0, 0, new CMsymsparse(objective));
  Matrix lb(n+k, 1, CB_minus_infinity), ub(n+k, 1, CB_plus_infinity);
  Matrix rhs(n+k, 1, 1.);
  for (Integer i=0; i<n; ++i) At.set(0, i, new CMsingleton(n, i, i, -1.));
  for (Integer i=0; i<k; ++i) {
    Integer length;
    Real cut_rhs;
    input >> length >> cut_rhs;
    Indexmatrix a(length, 1, Integer(0)), b(length, 1, Integer(0));
    Matrix values(length, 1, 0.);
    Real sum_coefficients=0.;
    for (Integer j=0; j<length; ++j) {
      Integer edge; Real value;
      input >> edge >> value;
      a(j)=u(edge); b(j)=v(edge); values(j)=value/2.;
      sum_coefficients += value;
    }
    At.set(0, n+i, new CMsymsparse(Sparsesym(n, length, a, b, values)));
    lb(n+i)=0.; rhs(n+i)=2.*cut_rhs-sum_coefficients;
  }
  if (!input) throw std::runtime_error("Malformed research input");
  PSCAffineFunction oracle(C, At, new GramSparsePSCPrimal(objective));
  oracle.set_out(&std::cout, 0);
  MatrixCBSolver solver(&std::cout, 0);
  Matrix initial(n+k, 1, 0.);
  if (argc == 5) {
    std::ifstream warm(argv[4]);
    for (Integer i=0; i<n+k; ++i) warm >> initial(i);
    if (!warm) throw std::runtime_error("Malformed initial multipliers");
  }
  if (solver.init_problem(n+k, &lb, &ub, argc == 5 ? &initial : 0, &rhs)) return 3;
  if (solver.add_function(oracle, Real(n), ObjectiveFunction)) return 4;
  solver.set_term_relprec(1e-7);
  solver.set_time_limit(std::stoi(argv[3]));
  int result=solver.solve();
  Matrix y;
  if (solver.get_center(y)) return 5;
  const auto* primal=dynamic_cast<const GramSparsePSCPrimal*>(solver.get_approximate_primal(oracle));
  Integer rank=0;
  std::string prefix(argv[2]);
  if (primal) {
    const Matrix& V=primal->get_grammatrix();
    rank=V.coldim();
    std::ofstream vectors(prefix+".vectors", std::ios::binary);
    for (Integer i=0; i<n; ++i) for (Integer j=0; j<rank; ++j) {
      double value=V(i,j);
      vectors.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
  }
  std::ofstream output(prefix);
  output << std::setprecision(17) << "{\"method\":\"conicbundle\",\"n\":" << n
         << ",\"rank\":" << rank << ",\"status\":" << solver.termination_code()
         << ",\"return_code\":" << result << ",\"numerical\":" << -solver.get_objval()
         << ",\"dual\":[";
  for (Integer i=0; i<y.dim(); ++i) output << (i ? "," : "") << y(i);
  output << "],\"seconds\":" << std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count() << "}\n";
}
