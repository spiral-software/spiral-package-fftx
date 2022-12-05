# spiral-package-fftx

SPIRAL Package for FFTX


Installation
------------

Clone this repository into the namespaces/packages subdirectory of your SPIRAL
installation tree and rename it to "fftx". For example, from the SPIRAL root
directory:

```
cd namespaces/packages
git clone https://github.com/spiral-software/spiral-package-fftx.git fftx
```

Examples
--------

This generates CUDA code for a batch of FFTs.

```
Load(fftx);
ImportAll(fftx);

n := 2;
N := 16;
xdim := n;
ydim := n;
zdim := n;
ix := Ind(xdim);
iy := Ind(ydim);
iz := Ind(zdim);

t := let(
    name := "grid_dft",
    TFCall(TRC(TMap(DFT(N, -1), [iz, iy, ix], AVec, AVec)), 
        rec(fname := name, params := []))
);

conf := LocalConfig.fftx.confGPU();
opts := conf.getOpts(t);
opts.wrapCFuncs := true;

tt := opts.tagIt(t);
c := opts.fftxGen(tt);
opts.prettyPrint(c);
```