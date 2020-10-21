# We are interested to see if
# IMDPRDFT(l)*MDPRDFT(l) = Product(l)*MatSPL(Product(l));
# because we want to see if we can recover the original REAL data.

Declare(MDPRDFT);
Declare(IMDPRDFT);

# Same as (floor(n_t/2)+1)*2.
# RClength := (n) -> (n + 1) mod 2 + n + 1;
RClength := (n) -> PRDFT1(n).dims()[1];

IJmatrix := (n) -> DirectSum(I(1), J(n-1));

tensorIJmatrix := (l) -> When(Length(l) > 0,
                         Tensor(IJmatrix(l[1]), tensorIJmatrix(Drop(l, 1))),
                         Diag([1, -1]));

#F MDPRDFT(<dims>, [<exp>]) - multi-dimensional PRDFT non-terminal
#F   dims = [ <n_1>,.., <n_t> ] list of (positive) dimensions
#F   exp = root of unity exponent scaling (see DFT for exact definition)
#F
#F Definition : multidimensional matrix of size M x N, where
#F M = n_1*...*n_{t-1}*(floor(n_t/2)+1)*2
#F N = n_1*..*n_t
#F This matrix has real components, for an operator with
#F N real inputs,
#F M real outputs for interleaved real and imaginary components.
#F
#F Example (direct)  : MDPRDFT([2,4,4])
#F Example (inverse) : MDPRDFT([2,4,4], -1)
#F
#F In last dimension, t: real-to-complex DFT;
#F then dimensions t-1, ..., 1: complex-to-complex DFT.
#F

# To verify that components of m are real:
# Im(MatSPL(m)) = MatSPL(ApplyFunc(O, m.dims()));

Class(MDPRDFT, TaggedNonTerminal, rec(
    a_lengths := self >> self.params[1],
    a_exp := self >> self.params[2],

    # Components of self.a_lengths() that are equal to 1 are removed.
    abbrevs := [
        P     -> Checked(IsList(P),
                         ForAll(P, IsPosInt),
                         Product(P) > 1,
                         [ RemoveOnes(P), 1 ]),
        (P,k) -> Checked(IsList(P),
                         ForAll(P, IsPosInt),
                         IsInt(k),
                         Product(P) > 1,
                         Gcd(Product(P), k) = 1,
                         [ RemoveOnes(P), k mod Product(P) ])
        ],

    # dims() has 2 components, counting outputs and inputs:
    # dims()[1] is product of all but last dimension, and RClength on last.
    # dims()[2] is product of all dimensions given.
    dims := self >> let(a_lengths := self.a_lengths(),
                        [Product(DropLast(a_lengths, 1)) *
                         RClength(Last(a_lengths)),
                         Product(a_lengths)
                        ]),

terminate := self >>  let(
        a_lengths := self.a_lengths(),
        Ninputs := self.dims()[2],              # number of scalar inputs
        Nlast := Last(a_lengths),        # length of last dimension
        Nrest := DropLast(a_lengths, 1), # all dimension lengths but last
        rcNlast := RClength(Nlast),
        # interleave has a 2x1 matrix [1; 0] along diagonal for each input;
        # it is a block-diagonal matrix of size (2*Ninputs) x Ninputs
        # that takes real inputs and puts a 0 between each.
        interleave := Scat(fTensor(fId(Ninputs), fBase(2,0))),
        # Here we are defining mddft as multi-dimensional DFT that takes
        # input as interleaved real and imaginary data,
        # output as interleaved real and imaginary data.
        mddft := RC(ApplyFunc(MDDFT, self.params)),
        # gath is block-diagonal matrix consisting of Product(Nrest) blocks
        # of size rcNlast x (2*Nlast), so it has
        # input as Product(Nrest) blocks of length 2*Nlast,
        # output as Product(Nrest) blocks of length rcNlast.
        f := fAdd(2*Nlast, rcNlast, 0), # pad length rcNlast to 2*Nlast
        # could also set f := HStack(I(rcNlast), O(rcNlast, 2*Nlast-rcNlast))
        ff := fTensor(List(Nrest, N -> fId(N)), f),
        # could also set ff := Tensor(List(Nrest, N->I(N)), f), no Gath needed.
        gath := Gath(ff),
        # 1. interleave: take real input, put a 0 after each element.
        # 2. mddft: multi-dimensional DFT, interleaved complex input & output.
        # 3. gath: in last dimension, keep only rcNlast of the 2*Nlast components.
        mdprdft := gath * mddft * interleave,
        mdprdft), 
    
    transpose := self >> IMDPRDFT(self.a_lengths(), -self.a_exp()),

    isReal := True,

    normalizedArithCost :=  (self) >> let(n := Product(self.a_lengths()),
                                        IntDouble(2 * n * d_log(n) / d_log(2)) )
));


# Conjugate in interleaved form:  (a, b) maps to (a, -b).
# That's multiplication by matrix [ [1, 0], [0, -1] ].
# That's fBase(2, 0) above -fBase(2, 1).
# Try adapting fId, replacing let(i := Ind(self.params[1]), Lambda(i,i)).
# Or dLin or dOmega.

#F IMDPRDFT(<dims>, [<exp>]) - multi-dimensional inverse PRDFT non-terminal
#F   dims = [ <n_1>,.., <n_t> ] list of (positive) dimensions
#F   exp = root of unity exponent scaling (see DFT for exact definition)
#F
#F Definition : multidimensional matrix of size N x M, where
#F N = n_1*..*n_t
#F M = n_1*...*n_{t-1}*(floor(n_t/2)+1)*2
#F This matrix has real components, for an operator with
#F M real inputs for interleaved real and imaginary components,
#F N real outputs.
#F
#F Example (direct)  : IMDPRDFT([2,4,4])
#F Example (inverse) : IMDPRDFT([2,4,4], -1)
#F
#F In dimensions 1, .., t-1: complex-to-complex DFT;
#F then in last dimension, t: complex-to-real DFT.
#F
Class(IMDPRDFT, TaggedNonTerminal, rec(
    a_lengths := self >> self.params[1],
    a_exp := self >> self.params[2],

    # Components of self.a_lengths() that are equal to 1 are removed.
    abbrevs := [
        P     -> Checked(IsList(P),
                         ForAll(P, IsPosInt),
                         Product(P) > 1,
                         [ RemoveOnes(P), 1 ]),
        (P,k) -> Checked(IsList(P),
                         ForAll(P, IsPosInt),
                         IsInt(k),
                         Product(P) > 1,
                         Gcd(Product(P), k) = 1,
                         [ RemoveOnes(P), k mod Product(P) ])
        ],

    # dims() has 2 components, counting outputs and inputs:
    # dims()[1] is product of all dimensions given.
    # dims()[2] is product of all but last dimension, and RClength on last.
    dims := self >> let(a_lengths := self.a_lengths(),
                        [Product(a_lengths),
                         Product(DropLast(a_lengths, 1)) *
                         RClength(Last(a_lengths))
                        ]),

    terminate := self >>  let(
        a_lengths := self.a_lengths(),
        Nlast := Last(a_lengths),        # length of last dimension
        Nrest := DropLast(a_lengths, 1), # all dimension lengths but last
        rcNlast := RClength(Nlast),
        newcomplex := Nlast - rcNlast/2,
        Nallbutlast := Product(Nrest),
        shuffle := Tensor( L(Nallbutlast*rcNlast/2, rcNlast/2), I(2)),
        unshuffle := Tensor( L(Nallbutlast*Nlast, Nallbutlast), I(2)),
        # This is the basic conjugation matrix.
        flipper := tensorIJmatrix(Nrest),
        newrows := 2*newcomplex*Nallbutlast,
        nzeroleft := When(Nlast mod 2 = 0,
                          Nallbutlast*(rcNlast - 2*newcomplex - 2),
                          Nallbutlast*(rcNlast - 2*newcomplex)),
        hpart := When(newrows > 0,
           HStack(
                  O(newrows, nzeroleft),
                  Tensor(J(newcomplex), flipper),
                  When(Nlast mod 2 = 0, O(newrows, 2*Nallbutlast), [ ])
                 ),
            [ ]),
        stacker := VStack(I(rcNlast*Nallbutlast), hpart),
        # Here we are defining mddft as multi-dimensional DFT that takes
        # input as interleaved real and imaginary data,
        # output as interleaved real and imaginary data.
        imddft := RC(ApplyFunc(MDDFT, self.params)),
        # extract real has 1x2 matrix [1, 0] along diagonal for each input;
        # it is a block-diagonal matrix of size (2*Noutputs) x Noutputs
        # that extracts every other component of the input.
        extractreal := Gath(fTensor(List(a_lengths, N->fId(N)), fBase(2, 0))),
        extend := unshuffle * stacker * shuffle,
        # 1. extend: in last dimension, extend the rcNlast interleaved complex input to 2*Nlast.
        # 2. imddft: inverse multi-dimensional DFT, interleaved complex input & output.
        # 3. extractreal: return the real parts of interleaved complex data.
        imdprdft := extractreal * imddft * extend,
        imdprdft), 
    
    transpose := self >> MDPRDFT(self.a_lengths(), -self.a_exp()),

    isReal := True,

    normalizedArithCost :=  (self) >> let(n := Product(self.a_lengths()),
                                        IntDouble(2 * n * d_log(n) / d_log(2)) )
));
