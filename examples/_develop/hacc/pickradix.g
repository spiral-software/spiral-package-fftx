knownFactors := [];


factorInto := function(N, stages)
    local stageval, fct, n, mapping, factors, m, buckets, j, sad, mn, idx, bestFct;
    stageval := exp(log(N)/stages).v;
    
    fct := Factors(N);
    n := Length(fct);
    mapping := ApplyFunc(Cartesian, Replicate(n, [1..stages]));
    
    factors := [];
    for m in mapping do
        buckets := Replicate(stages, 1);
        for j in [1..n] do
            buckets[m[j]] := buckets[m[j]] * fct[j];
           Add(factors, buckets);
        od;
    od;
    
    sad := List(factors, m -> Sum(List(m, i -> AbsFloat(i - stageval))));
    mn := Minimum(sad);
    idx := Position(sad, mn);
    bestFct := factors[idx];

    return bestFct;
end;

bestFactors := function(N, max_factor)
    local factors, i, f, bestf;
    
    if IsBound(knownFactors[N]) then return knownFactors[N]; fi;
    
    factors := List([2..4], i -> factorInto(N, i));
    
    bestf := Filtered(factors, f -> ForAll(f, i -> i < 26))[1];
    knownFactors[N] := bestf;
    return bestf;
end;




N := 30000;

factors := bestFactors(N, 16);




# -- alternative version
#f1 := peelFactor(N);
#f2 := peelFactorN(f1);
#f3 := peelFactorN(f2);

# N := 30000;
# MAX_FACTOR := 26;
# 
# peelFactor := n -> Filtered(DivisorPairs(n), e->e[1] <= MAX_FACTOR);
# peelFactorN := f -> let(lst := List(f, e -> DropLast(e, 1)::peelFactor(Last(e))), 
#     ApplyFunc(Concatenation, List(lst, a -> let(ll := Length(Filtered(a, v -> not IsList(v))), List(Drop(a, ll), v ->a{[1..ll]}::v)))));
# nthRoot := (n,r) -> exp(log(N)/r).v;
# 
# 
# factors := peelFactor(N);
# while not ForAny(factors, l -> ForAll(l, i -> i <= MAX_FACTOR)) do
#     factors := peelFactorN(factors);
# od;
# 
# stageval := nthRoot(N, Length(factors[1]));
# sad := List(factors, m -> Sum(List(m, i -> AbsFloat(i - stageval))));
# mn := Minimum(sad);
# idx := Position(sad, mn);
# factors := factors[idx];


#
N := 30000;
MAX_FACTOR := 26;

# peelFactor := n -> Filtered(DivisorPairs(n), e->e[1] <= MAX_FACTOR);
# expandFactors := f -> When(IsInt(f), When(Last(Factors(f)) > MAX_FACTOR, [], peelFactor(f)), let(lst := List(f, e -> DropLast(e, 1)::peelFactor(Last(e))), 
#     ApplyFunc(Concatenation, List(lst, a -> let(ll := Length(Filtered(a, v -> not IsList(v))), List(Drop(a, ll), v ->a{[1..ll]}::v))))));
# findFactors := lst -> When(not ForAny(lst, lst -> Last(lst) <= MAX_FACTOR), findFactors(expandFactors(lst)), lst);   
# 


peelFactor := n -> Filtered(DivisorPairs(n), e->e[1] <= MAX_FACTOR);
expandFactors := f -> let(lst := List(f, e -> DropLast(e, 1)::peelFactor(Last(e))), 
    ApplyFunc(Concatenation, List(lst, a -> let(ll := Length(Filtered(a, v -> not IsList(v))), List(Drop(a, ll), v ->a{[1..ll]}::v)))));
findFactors := lst -> When(IsInt(lst), When(Last(Factors(lst)) > MAX_FACTOR, [], findFactors(peelFactor(lst))), When(not ForAny(lst, lst -> Last(lst) <= MAX_FACTOR), findFactors(expandFactors(lst)), lst));   
factorize := n -> let(fct := When(Last(Factors(n)) > MAX_FACTOR, [[n]], findFactors(peelFactor(n))), nroot := exp(log(n)/Length(fct[1])).v, 
    sad := List(fct, m -> Sum(List(m, i -> AbsFloat(i -nroot)))),  mn := Minimum(sad), Sort(fct[Position(sad, mn)]));

factorize(N);







