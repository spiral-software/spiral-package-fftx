N := 14*15*16;

stages := 3;
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
