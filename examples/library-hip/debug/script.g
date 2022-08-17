cl := V(32);
rw := V(256);

i := Ind(rw*cl);
shft := V(1);
shft2 := V(1);

col := imod(i, cl);
row := idiv(i, cl);
row2 := idiv(i, cl*cl);


scol := imod(col + row * shft + row2*shft2, cl);

slin := row * cl + scol;

f := Lambda(i, slin);


fl := List(f.tolist(), i->i.v);


Set(List(fl{[0..63]*32+1}, i->Mod(i, 64)));



List([1..32], k->Set(List(fl{[0..63]*32+k}, i->Mod(i, 64)))=Set([0..63]));



List([1..64], k->Set(List(fl{[0..63]*128+k}, i->Mod(i, 64)))=Set([0..63]));




cl := V(64);
rw := V(128);

i := Ind(rw*cl);
shft := V(7);
shft2 := V(1);

f := Lambda(i, idiv(i, cl) * cl + imod(imod(i, cl) + idiv(i, cl) * shft + idiv(i, cl*cl)*shft2, cl));

fl := List(f.tolist(), i->i.v);

Set([0..rw.v*cl.v-1]) = Set(fl);
List([1..64], k->Set(List(fl{[0..63]*64+k}, i->Mod(i, 64)))=Set([0..63]));
List([1..64], k->Set(List(fl{[0..63]*128+k}, i->Mod(i, 64)))=Set([0..63]));


scat := Scat(f);
gath := Gath(f);

#==============================================================

cl := V(32);
rw := V(256);

i := Ind(rw*cl);
shft := V(1);
shft2 := V(1);

col := imod(i, cl);
row := idiv(i, cl);
row2 := idiv(i, cl*cl);


scol := imod(col + row * shft + row2*shft2, cl);

slin := row * cl + scol;

f := Lambda(i, slin);
fl := List(f.tolist(), i->i.v);

Set(List(fl{[0..63]*32+1}, i->Mod(i, 64)));
Set(List(fl{[0..63]*2+1}, i->Mod(i, 64)));



List([1..32], k->Set(List(fl{[0..63]*32+k}, i->Mod(i, 64)))=Set([0..63]));
