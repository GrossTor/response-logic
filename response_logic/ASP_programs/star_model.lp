reacted(I,OUT):-pert(I,IN),edge(IN,OUT).

reacted(I,N):-reacted(J,N),pert(J,M),J!=I,reacted(I,M).
reacted(I,N):-reacted(J,N),pert(J,M),J!=I,pert(I,M). %this line matter when multiple perturbations hit the same node, but a perturbed node does not react.
#show reacted/2.