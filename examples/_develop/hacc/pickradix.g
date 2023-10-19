N := 14*15*16;

stages := 3;
stageval := exp(log(N)/stages).v;

fct := Factors(N);
n := Length(fct);
mapping := ApplyFunc(Cartesian, Replicate(n, [1..stages]));

factors := [];
for i in mapping do
    buckets := Replicate(stages, 1);
    for j in [1..n] do
        buckets[m[j]] := buckets[m[j]] * fct[j];
       Add(factors, buckets);
    od;
od;

List(buckets, i -> AbsFloat(i - stageval));
