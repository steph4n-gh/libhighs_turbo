"""Portable, versioned bounds for externally supplied binary answers.

Models use integer indices and JSON lists. Coefficients are finite JSON numbers
or exact rational strings; a JSON float denotes its exact binary64 value.
Repeated terms, including reversed endpoints, add exactly. Max-Cut self-loops
contribute zero. Ising answers use -1/+1; QUBO and Max-Cut answers use 0/1.

Verification checks mathematical validity, not producer identity or provenance.
A different feasible answer can reuse the same bound and gets its own gap.
"""

from fractions import Fraction
import math
import re
import time

from highs_turbo._models import qubo_to_ising
from highs_turbo.ising import solve_ising, verify_ising_certificate
from highs_turbo.ising_cuts import IsingCertificate


# Explicit JSON-interface limits prevent a small malformed input from requesting
# an enormous dense vector. They are input limits, not solver performance claims.
MAX_VARIABLES = 100_000
MAX_TERMS = 1_000_000
MAX_COEFFICIENT_BITS = 4096


def _number(value):
    if type(value) not in (int, float, str):
        raise ValueError("Coefficients must be finite JSON numbers or rational strings")
    if isinstance(value, str):
        if len(value) > MAX_COEFFICIENT_BITS:
            raise ValueError("Coefficient exceeds the exact-arithmetic size limit")
        # Limit text before Fraction parses it: a short exponent such as
        # '1e999999999999' must not request an enormous integer allocation.
        if re.fullmatch(r"[+-]?[0-9]+(?:/[0-9]+)?", value) is None:
            raise ValueError("Rational strings must be integers or numerator/denominator")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Coefficients must be finite JSON numbers or rational strings")
    try:
        exact = Fraction(value)
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError("Coefficients must be finite JSON numbers or rational strings") from exc
    if max(exact.numerator.bit_length(), exact.denominator.bit_length()) > MAX_COEFFICIENT_BITS:
        raise ValueError("Coefficient exceeds the exact-arithmetic size limit")
    return exact


def _size(value):
    if type(value) is not int or not 0 <= value <= MAX_VARIABLES:
        raise ValueError(f"Variable count must be an integer in [0, {MAX_VARIABLES}]")
    return value


def _terms(rows, n, *, maxcut=False):
    if not isinstance(rows, list) or len(rows) > MAX_TERMS:
        raise ValueError(f"Terms must be a list with at most {MAX_TERMS} entries")
    combined = {}
    for row in rows:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("Every term must be a three-item [index, index, coefficient] list")
        u, v, raw = row
        if any(type(i) is not int or not 0 <= i < n for i in (u, v)):
            raise ValueError("Term endpoints must be integer indices within the model")
        value = _number(raw)
        if maxcut and u == v:
            continue
        pair = (min(u, v), max(u, v))
        combined[pair] = combined.get(pair, Fraction()) + value
    rows = []
    for (u, v), value in sorted(combined.items()):
        if value:
            if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_COEFFICIENT_BITS:
                raise ValueError("Combined coefficient exceeds the exact-arithmetic size limit")
            rows.append([u, v, str(value)])
    return rows


def _normalize_model(model):
    if not isinstance(model, dict):
        raise ValueError("model must be a JSON object")
    kind = model.get("type")
    if kind == "ising":
        allowed = {"type", "fields", "couplings", "offset"}
        fields = model.get("fields")
        if not isinstance(fields, list):
            raise ValueError("Ising fields must be a list")
        n = _size(len(fields))
        normalized = {"type": kind, "fields": [str(_number(v)) for v in fields],
                      "couplings": _terms(model.get("couplings", []), n),
                      "offset": str(_number(model.get("offset", 0)))}
    elif kind == "qubo":
        allowed = {"type", "num_variables", "coefficients", "offset"}
        n = _size(model.get("num_variables"))
        normalized = {"type": kind, "num_variables": n,
                      "coefficients": _terms(model.get("coefficients", []), n),
                      "offset": str(_number(model.get("offset", 0)))}
    elif kind == "maxcut":
        allowed = {"type", "num_nodes", "edges"}
        n = _size(model.get("num_nodes"))
        normalized = {"type": kind, "num_nodes": n,
                      "edges": _terms(model.get("edges", []), n, maxcut=True)}
    else:
        raise ValueError("model.type must be 'ising', 'qubo', or 'maxcut'")
    if set(model) - allowed:
        raise ValueError("Unknown model fields: " + ", ".join(sorted(map(str, set(model) - allowed))))
    return normalized, n


def _answer(answer, n, kind):
    domain = (-1, 1) if kind == "ising" else (0, 1)
    if (not isinstance(answer, list) or len(answer) != n
            or any(type(x) is not int or x not in domain for x in answer)):
        choices = "-1 or +1" if kind == "ising" else "0 or 1"
        raise ValueError(f"The external answer must contain one integer {choices} per variable")
    return list(answer)


def _ising_model(model, n):
    kind = model["type"]
    if kind == "qubo":
        return qubo_to_ising(n, ((i, j, Fraction(v)) for i, j, v in model["coefficients"]),
                             Fraction(model["offset"]))
    rows = model["couplings"] if kind == "ising" else model["edges"]
    fields = [Fraction(v) for v in model["fields"]] if kind == "ising" else [Fraction()] * n
    return fields, {(u, v): Fraction(w) for u, v, w in rows}, Fraction(model.get("offset", 0))


def verify(bundle):
    """Return exact objective, bound and gap after checking a bundle without solving.

    Raises ValueError for invalid models, assignments, versions or witnesses.
    Bounds are lower bounds for Ising/QUBO and upper bounds for Max-Cut.
    Producer metadata is informational and is not authenticated.
    """
    if not isinstance(bundle, dict) or type(bundle.get("version")) is not int or bundle["version"] != 1:
        raise ValueError("Unsupported certification bundle version; expected version 1")
    model, n = _normalize_model(bundle.get("model"))
    answer = _answer(bundle.get("answer"), n, model["type"])
    fields, couplings, offset = _ising_model(model, n)
    witness = bundle.get("certificate")
    if (not isinstance(witness, dict) or type(witness.get("version")) is not int
            or not verify_ising_certificate(fields, couplings, witness, offset=offset)):
        raise ValueError("Certificate does not verify against the supplied model")
    bound = IsingCertificate.from_dict(witness).lower_bound
    spins = answer if model["type"] == "ising" else [2 * x - 1 for x in answer]
    items = fields.items() if isinstance(fields, dict) else enumerate(fields)
    energy = offset + sum((h * spins[i] for i, h in items), Fraction())
    energy += sum((w * spins[u] * spins[v] for (u, v), w in couplings.items()), Fraction())
    gap = energy - bound
    if gap < 0:
        raise ValueError("Certified lower bound exceeds the external feasible energy")
    maximize = model["type"] == "maxcut"
    if maximize:
        total = sum(couplings.values(), Fraction())
        energy, bound, gap = (total - energy) / 2, (total - bound) / 2, gap / 2
    return {"model_type": model["type"], "objective_sense": "maximize" if maximize else "minimize",
            "candidate_objective": str(energy), "bound": str(bound), "gap": str(gap),
            "bound_verified": True, "optimality_proven": gap == 0}


def certify(model, answer, *, time_limit=2.0, seed=0):
    """Bound a supplied answer and return a complete JSON-serializable proof bundle.

    ``time_limit`` is a finite nonnegative cooperative budget, including model
    preparation. The supplied answer is checked, but does not seed solver search.
    Verification and serialization can take additional time. Use process
    supervision when the application requires a hard deadline.
    """
    began = time.perf_counter()
    try:
        valid_time = type(time_limit) in (int, float) and math.isfinite(time_limit) and time_limit >= 0
    except OverflowError:
        valid_time = False
    if not valid_time:
        raise ValueError("time_limit must be finite and nonnegative")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    normalized, n = _normalize_model(model)
    answer = _answer(answer, n, normalized["type"])
    fields, couplings, offset = _ising_model(normalized, n)
    result = solve_ising(fields, couplings, offset=offset,
                         time_limit=max(0., time_limit - (time.perf_counter() - began)),
                         seed=seed, threads=1)
    from highs_turbo import __version__

    bundle = {"version": 1, "model": normalized, "answer": answer,
              "certificate": result.certificate.to_dict(),
              "producer": {"package": "highs-turbo", "version": __version__,
                           "solver_status": result.status, "seed": seed,
                           "time_limit": time_limit, "api_seconds": time.perf_counter() - began}}
    verify(bundle)
    bundle["producer"]["api_seconds"] = time.perf_counter() - began
    return bundle
