# Research-only runners for unmodified published packages.
# julia --project=<research environment> published_baselines.jl METHOD INPUT OUTPUT SECONDS
using LinearAlgebra, SparseArrays, Random, JSON
using AugmentedMixing, TSSOS, DynamicPolynomials, JuMP, SCS
BLAS.set_num_threads(1)
method, input, output, seconds_text = ARGS
seconds = parse(Float64, seconds_text)
data = JSON.parsefile(input)
n = Int(data["n"])
edges = [Int.(edge) .+ 1 for edge in data["edges"]]
weights = Float64.(data["weights"])

if method == "mixing"
    function construct(n, edges, weights, cuts)
        C = spzeros(n,n)
        for ((u,v),w) in zip(edges, weights)
            C[u,v] += w/2; C[v,u] += w/2
        end
        As = [sparse([i],[i],[1.],n,n) for i in 1:n]
        b = ones(n)
        for cut in cuts
            A = spzeros(n,n)
            for (e,c) in zip(cut["indices"],cut["coefficients"])
                u,v = edges[Int(e)+1]
                A[u,v] += c/2; A[v,u] += c/2
            end
            push!(As,A)
            push!(b,sum(cut["coefficients"])-2*cut["rhs"])
        end
        SdpData(As,b,C,n+1)
    end
    # Compile with a separate tiny instance before starting the measurement.
    augmented_mixing(construct(3,[[1,2],[1,3],[2,3]],ones(3),[]); max_iters=1)
    Random.seed!(get(data,"seed",0))
    started = time_ns()
    sdp = construct(n,edges,weights,get(data,"cuts",[]))
    X,y,Z,status,ws = augmented_mixing(sdp; time_limit=seconds, tol=1e-7,
        mu_start=get(data,"mu_start",sqrt(Float64(n))),
        scaling=get(data,"scaling",true),tau=get(data,"tau",1.03))
    elapsed = (time_ns()-started)/1e9
    write(output*".gram",Z[1])
    open(output,"w") do stream
        JSON.print(stream,Dict("method"=>method,"seconds"=>elapsed,"status"=>string(status),
                              "dual"=>y,"n"=>n,"numerical"=>dot(sdp.b,y)))
    end
elseif method in ("tssos", "cs_tssos")
    # SCS is an explicitly supported open-source backend. No Mosek licence is
    # required; preserve TSSOS's basis and block construction without edits.
    function run_tssos(n,edges,weights,seconds; ts="MD")
        model = Model(optimizer_with_attributes(SCS.Optimizer,
                      "time_limit_secs"=>seconds,"eps_abs"=>1e-5,"eps_rel"=>1e-5,
                      "verbose"=>0))
        if method == "cs_tssos"
            # The Ising polynomial is already reduced modulo x_i^2=1.
            # Use the documented sparse-support API to avoid symbolic work.
            f = TSSOS.poly{Float64}([UInt16.(e) for e in edges],weights)
            return cs_tssos([f],n,2; nb=n,TS=ts,Gram=true,model=model,QUIET=true)
        end
        @polyvar x[1:n]
        f = sum(w*x[u]*x[v] for ((u,v),w) in zip(edges,weights))
        tssos([f],x,2; nb=n,TS=ts,Gram=true,model=model,QUIET=true)
    end
    run_tssos(3,[[1,2],[1,3],[2,3]],ones(3),.05)
    Random.seed!(get(data,"seed",0))
    started = time_ns()
    opt,sol,proof = run_tssos(n,edges,weights,seconds; ts=get(data,"ts","MD"))
    elapsed = (time_ns()-started)/1e9
    blocks = []
    cliques = method == "cs_tssos" ? eachindex(proof.blocks) : 1:1
    for clique in cliques
        cb = method == "cs_tssos" ? proof.blocks[clique][1] : proof.blocks[1]
        cg = method == "cs_tssos" ? proof.GramMat[clique][1] : proof.GramMat[1]
        basis = method == "cs_tssos" ? proof.basis[clique][1] : proof.basis[1]
        for (block,gram) in zip(cb,cg)
            push!(blocks,Dict("basis"=>[Int.(basis[i]) .- 1 for i in block],
                              "gram"=>[collect(row) for row in eachrow(gram)]))
        end
    end
    open(output,"w") do stream
        JSON.print(stream,Dict("method"=>method,"seconds"=>elapsed,
                              "status"=>string(proof.SDP_status),"numerical"=>opt,
                              "blocks"=>blocks,"n"=>n))
    end
else
    error("Unknown method: $method")
end
