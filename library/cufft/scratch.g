
opts := SpiralDefaults;
opts.includes := [];
opts.arrayBufModifier := "";

M := 1024;
N := 1024;
K := 1024;
p1 := 32;
p2 := 32;

plans := var("plans", TArray(TSym("cufftHandle"), 3));
sizes := var("sizes", TArray(TInt, 3));
grid := var("grid", TArray(TInt, 2));
cu_result := var("cuResult", TSym("cufftResult"));
mpi_result := var("mpiResult", TSym("mpiResult"));
t1 := var("T1", TPtr(TReal));
t2 := var("T2", TPtr(TReal));


c := data(sizes, Value(TArray(TInt, 3), [M, N, K]),
     data(grid, Value(TArray(TInt, 2), [p1, p2]),
     decl([plans, cu_result, mpi_result, t1, t2],
    chain(
        assign(cu_result, fcall("cufftPlanMany", plans, 1, sizes, sizes, 1, M,
    	    sizes, (N * K) / (p1 * p2), 1, "CUFFT_Z2Z", (N * K) / (p1 * p2))),
        assign(cu_result, fcall("cufftPlanMany", plans + 1, 1, sizes + 1, sizes + 1, 1, N,
    	    sizes + 1, (M * K) / (p1 * p2), 1, "CUFFT_Z2Z", (N * K) / (p1 * p2))),
        assign(cu_result, fcall("cufftPlanMany", plans + 2, 1, sizes + 2, sizes + 2, 1, K,
    	    sizes + 2, (M * N) / (p1 * p2), 1, "CUFFT_Z2Z", (M * N) / (p1 * p2))),
        skip(),
        assign(cu_result, fcall("cufftExecZ2Z", plans, tcast(TPtr(TSym("cufftDoubleComplex")), X), tcast(TPtr(TSym("cufftDoubleComplex")), t1), "CUFFT_FORWARD")),
        assign(mpi_result, fcall("mpi3DFFTStage_AllToAll", 0, sizes, grid)),
        assign(cu_result, fcall("cufftExecZ2Z", plans+1, tcast(TPtr(TSym("cufftDoubleComplex")), t1), tcast(TPtr(TSym("cufftDoubleComplex")), t2), "CUFFT_FORWARD")),
        assign(mpi_result, fcall("mpi3DFFTStage_AllToAll", 1, sizes, grid)),
        assign(cu_result, fcall("cufftExecZ2Z", plans+2, tcast(TPtr(TSym("cufftDoubleComplex")), t2), tcast(TPtr(TSym("cufftDoubleComplex")), Y), "CUFFT_FORWARD"))
    )
)));

PrintCode("", c, opts);

PrintTo("mpi_cuFFT_MDDFT_1024x1024x1024.cu", PrintCode("", c, opts));

