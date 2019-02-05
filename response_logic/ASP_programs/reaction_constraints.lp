%:-reacting(I,N,0),reacted(I,N).
%:-reacting(I,N,1),-reacted(I,N).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
:-not reacting(I,N),reacted(I,N).
:-reacting(I,N),-reacted(I,N).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%#show (I,N):reacting(I,N).


%#show edge/2.
%#show reacting/2.
%#show reacted/2.
%#show -reacted/2.