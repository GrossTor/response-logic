#program brave.
:- not reacting_brave(I,N,B) : pert(I,_), node(N), B=0..1, @is_brave_structure(N,I,B) == 0.

#show reacting_brave/3.