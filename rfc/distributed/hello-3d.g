# FFTX Hello World MPI 3D FFT distributed example
# block cyclic pencils on a 2D processor grid

Load(mpi);
ImportAll(mpi);
Load(fftx);
ImportAll(fftx);

# we have a 4 x 4 process grid
Nproc := [4, 4];
procGrid := MPICommunicator([0, 1], Nproc);

# get me the opts for that
conf := FFTXGlobals.defaulMPIConf(procGrid);
opts := FFTXGlobals.getOpts(conf);

# we do a distributed 3D DFT 1024 x 1024 x 1024
Nglobal := [1024, 1024, 1024];
k := -1;

# on each node we have a local box of 1024 x 256 x 256 
Nlocal := [Nglobal[1], Nglobal[2]/Nproc[1], Nglobal[3]/Nproc[2]];
localBrick := localMemory([0,1,2], Nlocal);

# thus our global box is 1024 x 1024 x 1024
dataLayout := globalMemory(procGrid, localBrick);
Xglobal := tcast(TPtrLayout(TComplex, dataLayout), X);
Yglobal := tcast(TPtrLayout(TComplex, dataLayout), Y);

# we do a 3D FFT on the logical 1024 x 1024 x 1024
mddft := TRC(MDDFT(Nglobal, k));
t := TFuncMPI(TDAG([
        TDAGNode(mddft, Yglobal, Xglobal)
    ])).withTags(opts.tags);

c := opts.fftxGen(t);
opts.prettyPrint(c);


