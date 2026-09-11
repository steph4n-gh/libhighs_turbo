#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "bit_graph.hpp"
#include "rational_verifier.hpp"
#include "cut_engine.hpp"
#include "solver_callback.hpp"


namespace py = pybind11;
using namespace highs_turbo;

PYBIND11_MODULE(_compiled_engine, m) {
    m.doc() = "Compiled C++ Engine for Neural-Surrogate Cutting Plane System";

    // 0. NativeTopologicalFeatures
    py::class_<BitGraph::NativeTopologicalFeatures>(m, "NativeTopologicalFeatures")
        .def_readonly("num_nodes", &BitGraph::NativeTopologicalFeatures::num_nodes)
        .def_readonly("num_edges", &BitGraph::NativeTopologicalFeatures::num_edges)
        .def_property_readonly("node_features", [](py::object& self) {
            auto& f = self.cast<BitGraph::NativeTopologicalFeatures&>();
            return py::array_t<float>(
                {f.num_nodes, static_cast<size_t>(5)},
                {5 * sizeof(float), sizeof(float)},
                f.node_features.data(),
                self
            );
        })
        .def_property_readonly("edge_features", [](py::object& self) {
            auto& f = self.cast<BitGraph::NativeTopologicalFeatures&>();
            return py::array_t<float>(
                {f.num_edges, static_cast<size_t>(8)},
                {8 * sizeof(float), sizeof(float)},
                f.edge_features.data(),
                self
            );
        });

    // 1. BitGraph
    py::class_<BitGraph>(m, "BitGraph")
        .def(py::init<size_t>(), py::arg("num_nodes"))
        .def("add_edge", &BitGraph::add_edge, py::arg("u"), py::arg("v"))
        .def("add_edges", &BitGraph::add_edges, py::arg("edges"))
        .def("has_edge", &BitGraph::has_edge, py::arg("u"), py::arg("v"))
        .def("degree", &BitGraph::degree, py::arg("u"))
        .def("count_common_neighbors", &BitGraph::count_common_neighbors, py::arg("u"), py::arg("v"))
        .def("count_triangles_on_edge", &BitGraph::count_triangles_on_edge, py::arg("u"), py::arg("v"))
        .def("count_triangles", &BitGraph::count_triangles)
        .def("find_triangles", &BitGraph::find_triangles)
        .def("find_k5_cliques", &BitGraph::find_k5_cliques)
        .def("get_edges", &BitGraph::get_edges)
        .def("compute_1wl_colors", &BitGraph::compute_1wl_colors, py::arg("max_iters") = 3)
        .def("extract_topological_features", &BitGraph::extract_topological_features,
             py::arg("edge_list"), py::arg("edge_weights") = std::vector<float>{})
        .def_property_readonly("num_nodes", &BitGraph::get_num_nodes)
        .def_property_readonly("num_edges", &BitGraph::get_num_edges)
        .def_readonly("words_per_row", &BitGraph::words_per_row);

    // 2. ExactRational
    py::class_<ExactRational>(m, "ExactRational")
        .def(py::init<>())
        .def(py::init<int64_t, int64_t>(), py::arg("n"), py::arg("d") = 1)
        .def(py::init<const std::string&, const std::string&>(), py::arg("n_str"), py::arg("d_str") = "1")
        .def_static("from_double", &ExactRational::from_double, py::arg("val"), py::arg("max_denominator") = 100000)
        .def("to_double", &ExactRational::to_double)
        .def("num_str", &ExactRational::num_str)
        .def("den_str", &ExactRational::den_str)
        .def("is_negative", &ExactRational::is_negative)
        .def("is_zero", &ExactRational::is_zero)
        .def("is_positive", &ExactRational::is_positive)
        .def("__str__", &ExactRational::str)
        .def("__repr__", [](const ExactRational& r) {
            return "ExactRational(" + r.num_str() + ", " + r.den_str() + ")";
        })
        .def("__add__", [](const ExactRational& a, const ExactRational& b) { return a + b; })
        .def("__sub__", [](const ExactRational& a, const ExactRational& b) { return a - b; })
        .def("__mul__", [](const ExactRational& a, const ExactRational& b) { return a * b; })
        .def("__lt__", &ExactRational::operator<)
        .def("__le__", &ExactRational::operator<=)
        .def("__gt__", &ExactRational::operator>)
        .def("__ge__", &ExactRational::operator>=)
        .def("__eq__", &ExactRational::operator==)
        .def("__ne__", &ExactRational::operator!=);

    // 3. NativeVerificationCertificate
    py::class_<NativeVerificationCertificate>(m, "VerificationCertificate")
        .def_readonly("is_valid", &NativeVerificationCertificate::is_valid)
        .def_readonly("status", &NativeVerificationCertificate::status)
        .def_readonly("rejection_reason", &NativeVerificationCertificate::rejection_reason)
        .def_readonly("num_active_supports", &NativeVerificationCertificate::num_active_supports)
        .def_readonly("sha256_hash", &NativeVerificationCertificate::sha256_hash)
        .def("exact_rhs_float", &NativeVerificationCertificate::exact_rhs_float)
        .def_property_readonly("exact_rhs", [](const NativeVerificationCertificate& cert) {
            return cert.exact_rhs;
        })
        .def("compute_sha256", &NativeVerificationCertificate::compute_sha256)
        .def("get_coefficients", [](const NativeVerificationCertificate& cert) {
            py::dict d;
            for (const auto& [e, coeff] : cert.exact_coefficients) {
                py::tuple edge = py::make_tuple(e.first, e.second);
                py::tuple frac = py::make_tuple(coeff.num_str(), coeff.den_str());
                d[edge] = frac;
            }
            return d;
        });

    // 4. RationalVerifier
    py::class_<RationalVerifier>(m, "RationalVerifier")
        .def(py::init<int64_t>(), py::arg("denominator_limit") = 100000)
        .def("to_rational", &RationalVerifier::to_rational, py::arg("val"))
        .def("verify_clique_conic_combination", [](
            RationalVerifier& self,
            const BitGraph& graph,
            const py::object& py_multipliers,
            const py::object& py_candidate_rhs,
            double tol
        ) {
            std::vector<std::pair<std::vector<uint32_t>, ExactRational>> mults;

            if (py::isinstance<py::dict>(py_multipliers)) {
                py::dict d = py_multipliers.cast<py::dict>();
                for (auto item : d) {
                    std::vector<uint32_t> clq;
                    if (py::isinstance<py::tuple>(item.first) || py::isinstance<py::list>(item.first)) {
                        for (auto elem : item.first) {
                            clq.push_back(elem.cast<uint32_t>());
                        }
                    }
                    ExactRational r;
                    if (py::isinstance<ExactRational>(item.second)) {
                        r = item.second.cast<ExactRational>();
                    } else if (py::hasattr(item.second, "numerator") && py::hasattr(item.second, "denominator")) {
                        std::string num_s = py::str(item.second.attr("numerator")).cast<std::string>();
                        std::string den_s = py::str(item.second.attr("denominator")).cast<std::string>();
                        r = ExactRational(num_s, den_s);
                    } else {
                        double v = item.second.cast<double>();
                        if (v < 0.0) {
                            r = self.to_rational(v);
                            if (r.is_zero()) r = ExactRational(-1, 1000000000LL);
                        } else {
                            r = self.to_rational(v);
                        }
                    }
                    mults.emplace_back(std::move(clq), std::move(r));
                }
            } else if (py::isinstance<py::list>(py_multipliers)) {
                py::list l = py_multipliers.cast<py::list>();
                for (auto item : l) {
                    py::tuple tup = item.cast<py::tuple>();
                    std::vector<uint32_t> clq;
                    for (auto elem : tup[0]) {
                        clq.push_back(elem.cast<uint32_t>());
                    }
                    ExactRational r;
                    if (py::isinstance<ExactRational>(tup[1])) {
                        r = tup[1].cast<ExactRational>();
                    } else if (py::hasattr(tup[1], "numerator") && py::hasattr(tup[1], "denominator")) {
                        std::string num_s = py::str(tup[1].attr("numerator")).cast<std::string>();
                        std::string den_s = py::str(tup[1].attr("denominator")).cast<std::string>();
                        r = ExactRational(num_s, den_s);
                    } else {
                        double v = tup[1].cast<double>();
                        if (v < 0.0) {
                            r = self.to_rational(v);
                            if (r.is_zero()) r = ExactRational(-1, 1000000000LL);
                        } else {
                            r = self.to_rational(v);
                        }
                    }
                    mults.emplace_back(std::move(clq), std::move(r));
                }
            }

            std::optional<ExactRational> cand_rhs = std::nullopt;
            if (!py_candidate_rhs.is_none()) {
                if (py::isinstance<ExactRational>(py_candidate_rhs)) {
                    cand_rhs = py_candidate_rhs.cast<ExactRational>();
                } else if (py::hasattr(py_candidate_rhs, "numerator") && py::hasattr(py_candidate_rhs, "denominator")) {
                    std::string num_s = py::str(py_candidate_rhs.attr("numerator")).cast<std::string>();
                    std::string den_s = py::str(py_candidate_rhs.attr("denominator")).cast<std::string>();
                    cand_rhs = ExactRational(num_s, den_s);
                } else {
                    cand_rhs = self.to_rational(py_candidate_rhs.cast<double>());
                }
            }

            return self.verify_clique_conic_combination(graph, mults, cand_rhs, tol);
        }, py::arg("graph"), py::arg("multipliers"), py::arg("candidate_rhs") = py::none(), py::arg("tol") = 1e-6)
        .def("verify_cycle_conic_combination", [](
            RationalVerifier& self,
            const BitGraph& graph,
            const py::list& py_multipliers,
            const py::object& py_candidate_rhs,
            double tol
        ) {
            std::vector<std::pair<std::pair<std::vector<uint32_t>, std::vector<std::pair<uint32_t, uint32_t>>>, ExactRational>> mults;
            for (auto item : py_multipliers) {
                py::tuple t = item.cast<py::tuple>();
                std::vector<uint32_t> cycle_nodes;
                std::vector<std::pair<uint32_t, uint32_t>> odd_edges;
                py::object mult_obj;
                if (t.size() == 3) {
                    cycle_nodes = t[0].cast<std::vector<uint32_t>>();
                    odd_edges = t[1].cast<std::vector<std::pair<uint32_t, uint32_t>>>();
                    mult_obj = t[2];
                } else if (t.size() == 2) {
                    py::tuple cycle_data = t[0].cast<py::tuple>();
                    cycle_nodes = cycle_data[0].cast<std::vector<uint32_t>>();
                    odd_edges = cycle_data[1].cast<std::vector<std::pair<uint32_t, uint32_t>>>();
                    mult_obj = t[1];
                } else {
                    throw std::invalid_argument("Expected tuple of (nodes, odd_edges, mult) or ((nodes, odd_edges), mult)");
                }

                ExactRational r;
                if (py::isinstance<ExactRational>(mult_obj)) {
                    r = mult_obj.cast<ExactRational>();
                } else if (py::hasattr(mult_obj, "numerator") && py::hasattr(mult_obj, "denominator")) {
                    std::string num_s = py::str(mult_obj.attr("numerator")).cast<std::string>();
                    std::string den_s = py::str(mult_obj.attr("denominator")).cast<std::string>();
                    r = ExactRational(num_s, den_s);
                } else {
                    double v = mult_obj.cast<double>();
                    if (v < 0.0) {
                        r = self.to_rational(v);
                        if (r.is_zero()) r = ExactRational(-1, 1000000000LL);
                    } else {
                        r = self.to_rational(v);
                    }
                }
                mults.push_back({{cycle_nodes, odd_edges}, r});
            }

            std::optional<ExactRational> cand_rhs = std::nullopt;
            if (!py_candidate_rhs.is_none()) {
                if (py::isinstance<ExactRational>(py_candidate_rhs)) {
                    cand_rhs = py_candidate_rhs.cast<ExactRational>();
                } else if (py::hasattr(py_candidate_rhs, "numerator") && py::hasattr(py_candidate_rhs, "denominator")) {
                    std::string num_s = py::str(py_candidate_rhs.attr("numerator")).cast<std::string>();
                    std::string den_s = py::str(py_candidate_rhs.attr("denominator")).cast<std::string>();
                    cand_rhs = ExactRational(num_s, den_s);
                } else {
                    cand_rhs = self.to_rational(py_candidate_rhs.cast<double>());
                }
            }

            return self.verify_cycle_conic_combination(graph, mults, cand_rhs, tol);
        }, py::arg("graph"), py::arg("multipliers"), py::arg("candidate_rhs") = py::none(), py::arg("tol") = 1e-6);

    // 5. ViolatedCut & SurrogateCut
    py::class_<ViolatedCut>(m, "ViolatedCut")
        .def_readonly("type", &ViolatedCut::type)
        .def_readonly("nodes", &ViolatedCut::nodes)
        .def_readonly("edge_indices", &ViolatedCut::edge_indices)
        .def_readonly("coefficients", &ViolatedCut::coefficients)
        .def_readonly("rhs", &ViolatedCut::rhs)
        .def_readonly("violation", &ViolatedCut::violation)
        .def_readonly("canonical_hash", &ViolatedCut::canonical_hash);

    py::class_<SurrogateCut>(m, "SurrogateCut")
        .def_readonly("full_coefficients", &SurrogateCut::full_coefficients)
        .def_readonly("nz_indices", &SurrogateCut::nz_indices)
        .def_readonly("nz_values", &SurrogateCut::nz_values)
        .def_readonly("rhs", &SurrogateCut::rhs)
        .def_readonly("is_valid", &SurrogateCut::is_valid)
        .def_readonly("status", &SurrogateCut::status)
        .def_readonly("certificate_hash", &SurrogateCut::certificate_hash)
        .def_readonly("num_active_supports", &SurrogateCut::num_active_supports);

    // 6. CutEngine
    py::class_<CutEngine>(m, "CutEngine")
        .def(py::init<int64_t>(), py::arg("rational_denominator_limit") = 100000)
        .def("separate_violated_k5", &CutEngine::separate_violated_k5,
             py::arg("graph"), py::arg("edge_list"), py::arg("primal_sol"),
             py::arg("threshold") = 1e-4, py::arg("max_cuts") = 0)
        .def("separate_violated_triangles", &CutEngine::separate_violated_triangles,
             py::arg("graph"), py::arg("edge_list"), py::arg("primal_sol"),
             py::arg("threshold") = 1e-4, py::arg("max_cuts") = 0)
        .def("separate_violated_4cycles", &CutEngine::separate_violated_4cycles,
             py::arg("graph"), py::arg("edge_list"), py::arg("primal_sol"),
             py::arg("threshold") = 1e-4, py::arg("max_cuts") = 0)
        .def_static("select_canonical_cuts", &CutEngine::select_canonical_cuts,
                    py::arg("cuts"), py::arg("max_cuts") = 0)
        .def("compile_surrogate_cut", &CutEngine::compile_surrogate_cut,
             py::arg("graph"), py::arg("edge_list"), py::arg("k5_multipliers"), py::arg("tol") = 1e-6)
        .def("compile_violation_surrogate", &CutEngine::compile_violation_surrogate,
             py::arg("graph"), py::arg("edge_list"), py::arg("violated_cuts"));

    // 7. SolverCallbackBridge
    py::class_<AddedRow>(m, "AddedRow")
        .def_readonly("lower", &AddedRow::lower)
        .def_readonly("upper", &AddedRow::upper)
        .def_readonly("indices", &AddedRow::indices)
        .def_readonly("values", &AddedRow::values)
        .def_readonly("label", &AddedRow::label);

    py::class_<SolverCallbackBridge>(m, "SolverCallbackBridge")
        .def(py::init<size_t>(), py::arg("num_edges"))
        .def("add_cut_row", &SolverCallbackBridge::add_cut_row,
             py::arg("upper"), py::arg("indices"), py::arg("values"), py::arg("label") = "")
        .def("add_surrogate_cut", &SolverCallbackBridge::add_surrogate_cut,
             py::arg("cut"), py::arg("label") = "")
        .def("get_num_rows", &SolverCallbackBridge::get_num_rows)
        .def("get_rows", &SolverCallbackBridge::get_rows)
        .def("get_latest_row", &SolverCallbackBridge::get_latest_row)
        .def("get_num_certified_cuts", &SolverCallbackBridge::get_num_certified_cuts)
        .def("get_certified_cuts", &SolverCallbackBridge::get_certified_cuts)
        .def("clear", &SolverCallbackBridge::clear)
        .def("get_flat_row_data", [](const SolverCallbackBridge& bridge) {
            std::vector<double> lower, upper, values;
            std::vector<int> starts, indices;
            bridge.get_flat_row_data(lower, upper, starts, indices, values);
            return py::make_tuple(lower, upper, starts, indices, values);
        });
}
