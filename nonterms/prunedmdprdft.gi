
##  Copyright (c) 2018-2021, Carnegie Mellon University
##  See LICENSE for details

Declare(PrunedMDPRDFT);
Declare(PrunedIMDPRDFT);

# Same as (floor(n_t/2)+1)*2.
# RClength := (n) -> (n + 1) mod 2 + n + 1;
RClength := (n) -> PRDFT1(n).dims()[1];

IJmatrix := (n) -> DirectSum(I(1), J(n-1));

tensorIJmatrix := (l) -> When(Length(l) > 0,
                         Tensor(IJmatrix(l[1]), tensorIJmatrix(Drop(l, 1))),
                         Diag([1, -1]));

# Distinct(list) means the elements of list are distinct.
Distinct := (l) -> (Length(l) = Length(Set(l)));

# pairup takes two lists of the same length,
# returns list of ordered pairs with elements of first and second.
pairup := (l1, l2) -> When(Length(l1)=0,
                           [],
                           Concat([ [l1[1], l2[1]] ],
                                  pairup(Drop(l1, 1),
                                         Drop(l2, 1))));


#F PrunedMDPRDFT(<dims>, <pat>, [<exp>=1])
#F Pruned multi-dimensional PRDFT (packed real DFT) non-terminal
#F   dims = [ <n_1>, ..., <n_t> ] list of (positive) dimensions
#F   pat = [ <l_1>, ..., <l_t> ] lists l_i having distinct elements in 0..n_i
#F   exp = root of unity exponent scaling (see DFT for exact definition)
#F
#F Definition : multidimensional matrix of size M x N, where
#F M = n_1*...*n_{t-1}*(floor(n_t/2)+1)*2
#F N = Length(l_1)*..*Length(l_t)
#F This matrix has real components, for an operator with
#F N real inputs,
#F M real outputs for interleaved real and imaginary components.
#F
#F Example (direct)  : MDPRDFT([4,5,6], [[0..3], [2..4], [5, 3, 1]])
#F Example (inverse) : MDPRDFT([4,5,6], [[0..3], [2..4], [5, 3, 1]], -1)
#F
#F In last dimension, t: real-to-complex DFT;
#F then dimensions t-1, ..., 1: complex-to-complex DFT.
#F

# To verify that components of m are real:
# Im(MatSPL(m)) = MatSPL(ApplyFunc(O, m.dims()));

Class(PrunedMDPRDFT, TaggedNonTerminal, rec(
    a_lengths := self >> self.params[1],
    a_pat := self >> self.params[2],
    a_exp := self >> self.params[3],

    abbrevs := [
        (L, pat)    -> Checked(IsList(L),
                               ForAll(L, IsPosInt),
                               ForAll(L, e->e > 1),
                               IsList(pat),
                               Length(pat) = Length(L),
                               ForAll(pat, IsList),
                               # Elements of pat[d] are distinct and in 0..L[d]-1.
                               ForAll(pat, Distinct),
                               Minimum(List(pat, Minimum)) >= 0,
                               ForAll(L - List(pat, Maximum), IsPosInt),
                               [ L, pat, 1 ]),
        (L, pat, k) -> Checked(IsList(L),
                               ForAll(L, IsPosInt),
                               ForAll(L, e->e > 1),
                               IsList(pat),
                               Length(pat) = Length(L),
                               ForAll(pat, IsList),
                               ForAll(pat, Distinct),
                               # Elements of pat[d] are distinct and in 0..L[d]-1.
                               Minimum(List(pat, Minimum)) >= 0,
                               ForAll(L - List(pat, Maximum), IsPosInt),
                               IsInt(k),
                               Gcd(Product(L), k) = 1,
                               [ L, pat, k mod Product(L) ])
        ],

    # dims() has 2 components, counting outputs and inputs:
    # dims()[1] is product of all but last dimension, and RClength on last.
    # dims()[2] is product of all lengths of pattern components.
    dims := self >> let(a_lengths := self.a_lengths(),
                        a_pat := self.a_pat(),
                        [Product(DropLast(a_lengths, 1)) *
                         RClength(Last(a_lengths)),
                         Product(List(a_pat, Length))
                        ]),

     # Just pad pat[d] with zeroes to fill 0..lengths[d]-1,
     # and then call MDPRDFT.  But would that be too simple?
     # It's the terminate function, not a Rule, so maybe OK?
terminate := self >>  let(
        a_lengths := self.a_lengths(),
        a_pats := self.a_pat(),
        a_exp := self.a_exp(),
        mdprdft := MDPRDFT(a_lengths, a_exp),
        lenpatpairs := pairup(a_lengths, a_pats),
        lv := List(lenpatpairs, lp->HStack(List(lp[2], e->Scat(fBase(lp[1], e))))),
        tlv := mdprdft * Tensor(lv),
        tlv),

    transpose := self >> PrunedIMDPRDFT(self.a_lengths(), self.a_pat(), -self.a_exp()),

    isReal := True,

    normalizedArithCost :=  (self) >> let(n := Product(self.a_lengths()),
                                        IntDouble(2 * n * d_log(n) / d_log(2)) )
));



#F PrunedIMDPRDFT(<dims>, <pat>, [<exp>=1])
#F Pruned multi-dimensional inverse PRDFT (packed real DFT) non-terminal
#F   dims = [ <n_1>, ..., <n_t> ] list of (positive) dimensions
#F   pat = [ <l_1>, ..., <l_t> ] lists l_i having distinct elements in 0..n_i
#F   exp = root of unity exponent scaling (see DFT for exact definition)
#F
#F Definition : multidimensional matrix of size N x M, where
#F N = Length(l_1)*..*Length(l_t)
#F M = n_1*...*n_{t-1}*(floor(n_t/2)+1)*2
#F This matrix has real components, for an operator with
#F M real inputs for interleaved real and imaginary components,
#F N real outputs.
#F
#F Example (direct)  : IMDPRDFT([4,5,6], [[0..3], [2..4], [5, 3, 1]])
#F Example (inverse) : IMDPRDFT([4,5,6], [[0..3], [2..4], [5, 3, 1]], -1)
#F
#F In dimensions 1, ..., t-1: complex-to-complex DFT;
#F then in last dimension, t: complex-to-real DFT.
#F

Class(PrunedIMDPRDFT, TaggedNonTerminal, rec(
    a_lengths := self >> self.params[1],
    a_pat := self >> self.params[2],
    a_exp := self >> self.params[3],

    abbrevs := [
        (L, pat)    -> Checked(IsList(L),
                               ForAll(L, IsPosInt),
                               ForAll(L, e->e > 1),
                               IsList(pat),
                               Length(pat) = Length(L),
                               ForAll(pat, IsList),
                               # Elements of pat[d] are distinct and in 0..L[d]-1.
                               ForAll(pat, Distinct),
                               Minimum(List(pat, Minimum)) >= 0,
                               ForAll(L - List(pat, Maximum), IsPosInt),
                               [ L, pat, 1 ]),
        (L, pat, k) -> Checked(IsList(L),
                               ForAll(L, IsPosInt),
                               ForAll(L, e->e > 1),
                               IsList(pat),
                               Length(pat) = Length(L),
                               ForAll(pat, IsList),
                               ForAll(pat, Distinct),
                               # Elements of pat[d] are distinct and in 0..L[d]-1.
                               Minimum(List(pat, Minimum)) >= 0,
                               ForAll(L - List(pat, Maximum), IsPosInt),
                               IsInt(k),
                               Gcd(Product(L), k) = 1,
                               [ L, pat, k mod Product(L) ])
        ],

    # dims() has 2 components, counting outputs and inputs:
    # dims()[1] is product of all lengths of pattern components.
    # dims()[2] is product of all but last dimension, and RClength on last.
    dims := self >> let(a_lengths := self.a_lengths(),
                        a_pat := self.a_pat(),
                        [Product(List(a_pat, Length)),
                         Product(DropLast(a_lengths, 1)) *
                         RClength(Last(a_lengths))
                        ]),

     # Just call IMDPRDFT and then prune in each dimension according to pat.
     # It's the terminate function, not a Rule, so maybe OK?
terminate := self >>  let(
        a_lengths := self.a_lengths(),
        a_pats := self.a_pat(),
        a_exp := self.a_exp(),
        imdprdft := IMDPRDFT(a_lengths, a_exp),
        lenpatpairs := pairup(a_lengths, a_pats),
        lv := List(lenpatpairs, lp->VStack(List(lp[2], e->Gath(fBase(lp[1], e))))),
        tlv := Tensor(lv) * imdprdft,
        tlv),

    transpose := self >> PrunedMDPRDFT(self.a_lengths(), self.a_pat(), -self.a_exp()),

    isReal := True,

    normalizedArithCost :=  (self) >> let(n := Product(self.a_lengths()),
                                        IntDouble(2 * n * d_log(n) / d_log(2)) )
));
