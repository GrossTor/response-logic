#program reacting_conversion.

reacting_brave(I,N,1):-reacting(I,N).
reacting_brave(I,N,0):-not reacting(I,N),pert(I,_),node(N).