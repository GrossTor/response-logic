__all__=('conform_response_pattern','brave_solving','sparsify_brave_structure',
         'prepare_ASP_program','iterate_conforming_networks')


import clingo
import time

import numpy as np
import itertools as it
import os


root_folder=os.path.dirname(__file__)
ASP_programs_folder=os.path.join(root_folder, 'ASP_programs/')


def my_logger_setup(logging_state):
    '''Control the output to STDOUT (printing) behaviour. The logging state
    decides which importance level print statement must have to be be executed.
    '''    
    if logging_state=='silent':
        def my_logger(*to_log,importance=10):
            return
    else:
        def my_logger( *to_log,importance=10):
            if importance<5 and logging_state=='calm':
                return
            else:
                print(*to_log, end='')
            return
    return my_logger

my_logger=my_logger_setup('calm')

def prepare_ASP_program(model,lp_program,heuristics={},ASP_options=[]):
    '''Returns a clingo Control Object, into which the response logic 
    ('lp_program') is loaded, using 'options' as well as the known edges and
    reaction information.
    heuristics allow to add additional constraints beyond those induced by data:
    -global_{max/min}_nbr_{in/out}edges: etc allow to put boundaries on the 
        number of nodes from/to any node
    -local_{max/min}_nbr_inedges: same as before but allows to specify a particular node
        to which this applies
    -parsimonious: discard any network which has links that can be removed without
        altering the response pattern
    Requires model['response']['rxn_on_pert'] and 
    model['response']['no_rxn_on_pert'].
    '''
    #lp_program : reaction.lp reaction_minimal_model.lp
    #options : ['--configuration=handy']) trendy
    
    N=model['N']
    #P=model['P']
    reaction_upon_perturbation=model['response']['rxn_on_pert']
    no_reaction_upon_perturbation=model['response']['no_rxn_on_pert']
    
    rxn_logic_pgr = clingo.Control(ASP_options)
    try: rxn_logic_pgr.load(os.path.join(ASP_programs_folder,lp_program)) #rxn_logic_pgr.load(ASP_programs_folder+'/'+lp_program)
    
    except RuntimeError: 
        my_logger('Clingo program loading failed! Try, once more with feelings.',model,'\n',importance=10)
        time.sleep(0.5)
        rxn_logic_pgr = clingo.Control(ASP_options)
        time.sleep(0.5)
        try: rxn_logic_pgr.load(os.path.join(ASP_programs_folder,lp_program))#rxn_logic_pgr.load(ASP_programs_folder+'/'+lp_program)
        except RuntimeError:
            raise Warning('Clingo program loading failed twice! I give up.','\n')
    
   
    add_string="#const n={0}.\n".format(N)
    
    #add pk
    for (node_out,node_in),exists in model['input']['edges_known'].items():
        if exists==True: add_string+="edge({0},{1}).\n".format(node_out,node_in)
        elif exists==False: add_string+="-edge({0},{1}).\n".format(node_out,node_in)
                    
    #add reaction information
    for I,reacting_nodes in reaction_upon_perturbation.items():
        for node in reacting_nodes:
            add_string+="reacted({0},{1}).\n".format(I,node)
    for I,reacting_nodes in no_reaction_upon_perturbation.items():
        for node in reacting_nodes:
            add_string+="-reacted({0},{1}).\n".format(I,node)
            
    for node,pert in zip(*np.where(np.isnan(model['input']['Sen_known']))):
        add_string+="pert({0},{1}).\n".format(pert,node)
    
    rxn_logic_pgr.add("base", [], add_string)



    #add some more heuristics
    if 'global_max_nbr_inedges' in heuristics:
        edge_lims = ':- #count {{OUT : edge(OUT,IN) }} > {0} , node(IN).\n'.format(heuristics['global_max_nbr_inedges'])
        rxn_logic_pgr.add("base", [], edge_lims)
    if 'global_min_nbr_inedges' in heuristics:
        edge_lims = ':- #count {{OUT : edge(OUT,IN) }} < {0} , node(IN).\n'.format(heuristics['global_min_nbr_inedges'])
        rxn_logic_pgr.add("base", [], edge_lims)
    if 'global_max_nbr_outedges' in heuristics:
        edge_lims = ':- #count {{IN : edge(OUT,IN) }} > {0} , node(OUT).\n'.format(heuristics['global_max_nbr_outedges'])
        rxn_logic_pgr.add("base", [], edge_lims)
    if 'global_min_nbr_outedges' in heuristics:
        edge_lims = ':- #count {{IN : edge(OUT,IN) }} < {0} , node(OUT).\n'.format(heuristics['global_min_nbr_outedges'])        
        rxn_logic_pgr.add("base", [], edge_lims)

    if 'local_max_nbr_inedges' in heuristics:
        for node,max_in in heuristics['local_max_nbr_inedges']: #expects list of pairs
            edge_lims = ':- #count {{OUT : edge(OUT,{0}) }} > {1} , node({0}).\n'.format(node,max_in)
            rxn_logic_pgr.add("base", [], edge_lims)
    if 'local_min_nbr_inedges' in heuristics:
        for node,max_in in heuristics['local_min_nbr_inedges']: #expects list of pairs
            edge_lims = ':- #count {{OUT : edge(OUT,{0}) }} < {1} , node({0}).\n'.format(node,max_in)
            rxn_logic_pgr.add("base", [], edge_lims)
    if 'local_max_nbr_outedges' in heuristics:
        for node,max_in in heuristics['local_max_nbr_outedges']: #expects list of pairs
            edge_lims = ':- #count {{IN : edge({0},IN) }} > {1} , node({0}).\n'.format(node,max_in)
            rxn_logic_pgr.add("base", [], edge_lims)
    if 'local_min_nbr_outedges' in heuristics:
        for node,max_in in heuristics['local_min_nbr_outedges']: #expects list of pairs
            edge_lims = ':- #count {{IN : edge({0},IN) }} < {1} , node({0}).\n'.format(node,max_in)
            rxn_logic_pgr.add("base", [], edge_lims)   
    
    
    if ('parsimonious' in heuristics) and (not heuristics['parsimonious']==False):    
        #What if prior knowledge is not parsimonious? Is there a way to only disallow additional parsimonity?
        #yes I define candidate_edge (which is the one that
        #gets removed from edge to form reduced_edge:
        parsimonious='candidate_edge(NOUT,NIN) :- '
        for edge,linked in model['input']['edges_known'].items():
            if linked:
                parsimonious+='(NOUT,NIN) != {0}, '.format(edge)
        parsimonious+= 'edge(NOUT,NIN).\n'       
        
        parsimonious+= 'reduced_edge(OUT,IN,NOUT,NIN)   :- (OUT,IN) != (NOUT,NIN), edge(OUT,IN), candidate_edge(NOUT,NIN) .\n'
        parsimonious+='reduced_reacting(I,IN,NOUT,NIN) :- reduced_edge(OUT,IN,NOUT,NIN),pert(I,OUT).\n'
        parsimonious+='reduced_reacting(I,IN,NOUT,NIN) :- reduced_edge(OUT,IN,NOUT,NIN),reduced_reacting(I,OUT,NOUT,NIN).\n'
        #it could be that removal of a link (NOUT,NIN) implies less reacting, check for which (NOUT,NIN) this is so:
        #parsimonious+='actually_reduced(NOUT,NIN)      :- not reduced_reacting(I,IN,NOUT,NIN),reacting(I,IN), edge(NOUT,NIN).\n'
        parsimonious+='actually_reduced(NOUT,NIN)      :- not reduced_reacting(I,IN,NOUT,NIN),reacting(I,IN), candidate_edge(NOUT,NIN).\n'
        
        #if there is any removed link (NOUT,NIN) that does not imply less reacting, the 
        #network is not parsimonious and should be discarded
        parsimonious+=':- not actually_reduced(NOUT,NIN), candidate_edge(NOUT,NIN).\n'       
        rxn_logic_pgr.add("base", [], parsimonious)        
    return rxn_logic_pgr



#Functions and class needed to run clingo calls.
brave={}
def on_brave_edge_model(m):
    for x in m.symbols(shown=True):        
        #my_logger( x,'\n')
        if x.name == "edge_brave": brave[(x.arguments[1].number,x.arguments[0].number)].add(x.arguments[2].number)

def on_brave_reacting_model(m):
    for x in m.symbols(shown=True):        
        #my_logger( x,'\n')
        if x.name == "reacting_brave": brave[(x.arguments[1].number,x.arguments[0].number)].add(x.arguments[2].number)

def on_finish(finish):
    brave['SolveResult']=finish
    
class brave_context:
    def is_brave_structure(self,r,c,b): 
        #my_logger(r,type(r),'\n')
        return clingo.Number(b.number in brave[(r.number,c.number)])


def brave_solving(model,mode='edge',heuristics={},timeout=None,assumptions=False,):
    '''
    For the given logical program, finds the brave consequences (union over all
    answer sets) of either edge sets or reaction patterns, depending on mode = 
    'edge' or 'reacting'
    If a timeout is given, clingo is asked to solve asynchroneously, so that
    a running search can be interupted if brave_solving call exceeds time 
    limit.
    Sometimes, finding the 'brave_solving' can be (drasticly) accelerated by
    'assumptions': only admits answer sets that contain a given set
    of atoms. Set assumptions to True, to use star-net-type edges, infered 
    from model['rxn_on_pert'], as assumptions. However, this might yield
    incorrect brave consequences if there is prior knowledge about the absence 
    of certain edges. But could also work in that case, since I only solve with
    assumptions the first time. This feature needs further testing.
    
    Requires model['response']['rxn_on_pert'] and 
    model['response']['no_rxn_on_pert'].
    
    'If rxn_logic_prg is given it will use this logic program instead of generating
    its own. This is useful to use programs with heuristic constraints.
    '''
   
    #Define how solving is done, depending on whether or not there is a timeout.#
    #############################################################################
    my_logger('Searching for brave consequences...\n',importance=2)
    if timeout is None:
        finishing_time=np.inf
        remaining_time=np.inf
        def solve_call(prg,assumptions,remaining_time):
            prg.solve(on_model=on_model_func, on_finish=on_finish, 
                      assumptions=assumptions)
            return True
        
    else:
        finishing_time=time.time()+timeout
        remaining_time=timeout
        def solve_call(prg,assumptions,remaining_time):
            with prg.solve(on_model=on_model_func, on_finish=on_finish,
                           assumptions=assumptions, async_=True) as handle:
                finished=handle.wait(remaining_time)
                if not finished:
                    #search did not finish in time. Interrupt! Maybe better
                    #(thread-safe) to use Control.interrupt
                    handle.cancel()
            return finished
                   
    
    #Set-up the logic program and perform first solving call (including
    #assumptions). To get first answer set ('conversion' part remains un-ground
    #in this part)
    ###########################################################################
           
    rxn_logic_pgr=prepare_ASP_program(model,'reaction.lp',heuristics,ASP_options=[])
    rxn_logic_pgr.load(os.path.join(ASP_programs_folder,'reaction_constraints.lp'))#ASP_programs_folder+'/reaction_constraints.lp')
    
    rxn_logic_pgr.ground([("base", [])])
            
    if mode=='edge':
        prg_name=os.path.join(ASP_programs_folder,'bravery_edge.lp') #ASP_programs_folder+'/bravery_edge.lp',
        on_model_func=on_brave_edge_model
        brave_dim=[model['N'],model['N']]
        rxn_logic_pgr.load(os.path.join(ASP_programs_folder,'edge_conversion.lp'))#ASP_programs_folder+'/edge_conversion.lp')
        rxn_logic_pgr.ground([("edge_conversion", [])])
    
    elif mode=='reacting':
        prg_name=os.path.join(ASP_programs_folder,'bravery_reacting.lp') #ASP_programs_folder+'/bravery_reacting.lp',
        on_model_func=on_brave_reacting_model
        brave_dim=[model['N'],model['Q']]
        rxn_logic_pgr.load(os.path.join(ASP_programs_folder,'reacting_conversion.lp'))#(ASP_programs_folder+'/reacting_conversion.lp')
        rxn_logic_pgr.ground([("reacting_conversion", [])])
    rxn_logic_pgr.load(prg_name)

    for i in range(brave_dim[0]):
        for j in range(brave_dim[1]):
            brave[(i,j)]=set([])           
    
    if assumptions:
        my_logger('I shall use assumptions for first brave consequences solving.\n',importance=2)
        assumptions=[]
        for IN,OUT in it.product(range(model['N']),repeat=2):
            if IN in model['response']['rxn_on_pert'][OUT]: exists=True
            else: exists=False
            assumptions.append((clingo.Function("edge",[clingo.Number(OUT),clingo.Number(IN)]),exists))
        #rxn_logic_pgr.ground([("brave", [])],context)
    else:
        assumptions=[]
    
    finished=solve_call(rxn_logic_pgr,assumptions,remaining_time)
    #I only want to solve with assumptions once. So set them to []. See doc_string
    assumptions=[]


    #Look for brave consequences! Repeat solve calls until 'brave' is complete.#
    ############################################################################
    
    context=brave_context()
    remaining_time=finishing_time-time.time()
    while remaining_time>0 and finished:
        remaining_time=finishing_time-time.time()
        # print('hoppa')
        rxn_logic_pgr.ground([("brave", [])],context)
        #rxn_logic_pgr.ground([("brave", [])])
        finished=solve_call(rxn_logic_pgr,assumptions,remaining_time)
        my_logger('last remaining_time:{0}, last run finished?:{1}\n'.format(
                remaining_time,finished),importance=2)
        #my_logger('Number entries in brave',np.sum(len(val) for val in brave.itervalues()),'\n')
        if not finished: 
            break
        elif not brave['SolveResult'].satisfiable: break
    
    
    #Finished! Now collect results.#
    ################################
    
    if (not finished) or remaining_time<0: 
        model['response']['brave enumeration aborted']=True
    else: 
        model['response']['brave enumeration aborted']=False

    brave_array=-np.ones(brave_dim) #to allow for a forth state
    for i in range(brave_dim[0]):
        for j in range(brave_dim[1]):
            if brave[(i,j)]=={1}:
                brave_array[i,j]=1.
            elif brave[(i,j)]=={0,1}:
                brave_array[i,j]=.5
            elif brave[(i,j)]=={0}:
                brave_array[i,j]=0.
    if 'SolveResult' in brave: del brave['SolveResult']
    if mode=='edge': 
        model['response']['brave edge array']=brave_array
        model['response']['brave edge dict']=brave
    if mode=='reacting': 
        model['response']['brave reacting array']=brave_array
        model['response']['brave reacting dict']=brave
    return #finished,brave,brave_array



callback={} #This global variable is needed because ASP callback functions ("on_model","on_finish") do not accept arguments
def on_model(model):
    callback['ASP_terms'].append( model.symbols(terms=True))
    callback['ASP_shown'].append( model.symbols(shown=True))
    #my_logger( 'das kommt raus:\n', model.symbols(terms=True))
    return

def conform_response_pattern(model,scheme='star-model',heuristics={},ASP_options=[],
                     fact_timeout=25):
    '''
    A given reaction pattern ['nonzero'] might contradict any network structure.
    This function alters it to make it to conform with at least one network 
    structure. There are different ways to accomplish this. All assume that
    there is some confidence score on each entry in a reaction pattern (if an
    entry in the confidence score is NaN the according data point is skipped):
        
        scheme='star-model' - Iteratively go through reaction pattern entries
            with decreasing confidence. Draw an edge between a perturbed node
            and all nodes that reacted to perturbation. This is fast, because
            it does not require to consider all possible net structures. 
            Whenever a reaction pattern entry yields an UNSATISFIABILITY, 
            discard it. 
            This might not work if there is prior knowledge about the absence
            of links.
        scheme='general': Checks response logic and all additional constraints
            for required data points. Works in general, but might be slower.
            
    'fact_timeout' determines how much time is spend maximally to verify
    validity of a reaction pattern entry. If timed out that entry will be 
    discarded.
    
    Requires:
      model['response']['confidence']
      model['response']['nonzero']
    '''
    build_up_rxn_on_pert,build_up_no_rxn_on_pert={},{}
    model['response']['rxn_on_pert']=build_up_rxn_on_pert
    model['response']['no_rxn_on_pert']=build_up_no_rxn_on_pert
    #scheme='definition_constraint'#'deletion','all combinations','optimization','definition_constraint','build_up'
    N,Q=model['N'],model['Q']
    
    sorted_node_inds,sorted_pert_inds=np.unravel_index(
        np.flipud(np.argsort(model['response']['confidence'],axis=None)),[N,Q])
    step=0
    
    if scheme=='star-model':        
        if heuristics!={}:
            my_logger('Warning: The provided heuristics are ignored, because scheme="star-model" was chosen.\n',
                      importance=10)
        
        last_reacted=set([])
        #initialize 'last_reacted' (prior knowledge on edges might enforce certain reactions)            
        rxn_logic_pgr=prepare_ASP_program(model,'star_model.lp',ASP_options=ASP_options)
        rxn_logic_pgr.ground([("base", [])])
        callback['ASP_terms'],callback['ASP_shown'],callback['rets']=[],[],[]
        rxn_logic_pgr.solve(on_model=on_model)   
        last_reacted=set([]) #maybe not necessary because star-model only increases reacted nodes (but mabye safer if star-model is modified later on)
        if callback['ASP_shown']: #callback is empty if non-satisfiable (on_finish fails to run with timeout)                   
            for x in callback['ASP_shown'][-1]:
                if x.name == "reacted": last_reacted.add((x.arguments[0].number,x.arguments[1].number))   
        #store enforced reactions        
        for (pert,node) in last_reacted:            
            build_up_rxn_on_pert.setdefault(pert,set([node])).add(node) 
        
        
        #go through reaction pattern entries with decreasing confidence
        
        nbr_non_conform_infos,nbr_ignored_infos,nbr_avoided_tests,nbr_solve_calls,step=0,0,0,0,0

        insatisfiable_facts=set([])
        for node,pert in zip(sorted_node_inds,sorted_pert_inds):
            if np.isnan(model['response']['confidence'][node,pert]): continue #skip reduced data
            if not model['response']['confidence'][node,pert]>0.: break 
            step+=1
            my_logger('Fact nbr: ',step,'confidence[',[node,pert],']=',
                      model['response']['confidence'][node,pert],'type:',
                      model['response']['nonzero'][node,pert],'\n',importance=2)
            if model['response']['nonzero'][node,pert]:
                build_up_rxn_on_pert.setdefault(pert,set([node])).add(node)
                if (pert,node) in last_reacted:
                    #the last reacting model conforms to new fact. No new solve call needed.
                    my_logger('New reacted already implied. No need to run solve.\n',importance=2)
                    nbr_avoided_tests+=1
                    continue
            else:
                #In star-model -reacted data is satisfiable if not in reacted and unsatisfiable otherwise. No need to call solve.
                nbr_avoided_tests+=1
                if (pert,node) not in last_reacted:            
                    build_up_no_rxn_on_pert.setdefault(pert,set([node])).add(node) 
                else: nbr_non_conform_infos+=1
                continue

            rxn_logic_pgr=prepare_ASP_program(model,'star_model.lp',ASP_options=ASP_options)
            rxn_logic_pgr.ground([("base", [])])
            callback['ASP_terms'],callback['ASP_shown'],callback['rets']=[],[],[]  
            with rxn_logic_pgr.solve(on_model=on_model,async_=True) as handle:
                nbr_solve_calls+=1
                finished=handle.wait(fact_timeout)
                if finished:
                    if callback['ASP_shown']: #callback is empty if non-satisfiable (on_finish fails to run with timeout)   
                        last_reacted=set([]) #maybe not necessary because star-model only increases reacted nodes (but mabye safer if star-model is modified later on)                
                        for x in callback['ASP_shown'][-1]:
                            if x.name == "reacted": last_reacted.add((x.arguments[0].number,x.arguments[1].number))
                        #my_logger(last_reacted,'\n')
                    else:
                        #if not satisfiable ignore the last fact
                        my_logger('insatisfiable!\n',importance=2)
                        insatisfiable_facts.add((pert,node))
                        if model['response']['nonzero'][node,pert]: build_up_rxn_on_pert[pert].remove(node)
                        else: build_up_no_rxn_on_pert[pert].remove(node)                    
                        #maybe I can speed up ASP solving if I not only remove nodes but also put them in the other set,
                        #To have more (redundant) constraints.
                        nbr_non_conform_infos+=1
                else:
                    handle.cancel()
                    my_logger('Solve call timed out. Ignore fact.\n',importance=2)

                    nbr_ignored_infos+=1          
        my_logger('nbr_skipped_infos: ',nbr_ignored_infos,'nbr_non_conform_infos:',
                  nbr_non_conform_infos,'nbr_avoided_tests',nbr_avoided_tests,
                  'nbr_solve_calls:',nbr_solve_calls,'\n',importance=2)

    
    
    
    elif scheme=='general':   
        #go through reaction pattern entries with decreasing confidence
        nbr_non_conform_infos,nbr_ignored_infos,step=0,0,0
        insatisfiable_facts=set([])
        for node,pert in zip(sorted_node_inds,sorted_pert_inds):
            if np.isnan(model['response']['confidence'][node,pert]) or \
                np.isnan(model['response']['nonzero'][node,pert]): continue #skip reduced data
                
            if not model['response']['confidence'][node,pert]>0.: 
                my_logger('Confidence<0 for some entries in response pattern. Those will be ignored.\n',importance=8)
                break
            step+=1
            my_logger('Fact nbr: ',step,'confidence[',[node,pert],']=',
                      model['response']['confidence'][node,pert],'type:',
                      model['response']['nonzero'][node,pert],'\n',importance=2)
            
            if model['response']['nonzero'][node,pert]:
                build_up_rxn_on_pert.setdefault(pert,set([node])).add(node)
            else:
                build_up_no_rxn_on_pert.setdefault(pert,set([node])).add(node) 

            rxn_logic_pgr=prepare_ASP_program(model,"reaction.lp",heuristics,ASP_options=ASP_options)
            rxn_logic_pgr.load(os.path.join(ASP_programs_folder,'reaction_constraints.lp'))
            rxn_logic_pgr.ground([("base", [])])
            callback['ASP_terms'],callback['ASP_shown'],callback['rets']=[],[],[]  
            with rxn_logic_pgr.solve(on_model=on_model, async_=True) as handle:
                finished=handle.wait(fact_timeout)
                if finished:
                    #print(callback['ASP_shown'])
                    if not(callback['ASP_shown']): #callback is empty if non-satisfiable (on_finish fails to run with timeout)   
                        #if not satisfiable ignore the last fact
                        my_logger('insatisfiable!\n',importance=2)
                        insatisfiable_facts.add((pert,node))
                        if model['response']['nonzero'][node,pert]: build_up_rxn_on_pert[pert].remove(node)
                        else: build_up_no_rxn_on_pert[pert].remove(node)                    
                        nbr_non_conform_infos+=1
                    else:
                        my_logger('satisfiable!\n',importance=2)
                else:
                    handle.cancel()
                    my_logger('Solve call timed out. Ignore fact.\n',importance=2)
                    nbr_ignored_infos+=1
        my_logger('nbr_skipped_infos: ',nbr_ignored_infos,'nbr_non_conform_infos:',nbr_non_conform_infos,
                  '\n',importance=2)
            
    else:
        raise UserWarning('No (valid) scheme was chosen. Cannot rectify response pattern.')

    model['response']['insatisfiable facts']=insatisfiable_facts
    
    return# build_up_rxn_on_pert,build_up_no_rxn_on_pert,insatisfiable_facts


on_model_edge=[set([])]
def on_model_sparsify(model):
    for x in model.symbols(shown=True):        
        if x.name == "edge": on_model_edge[0].add((x.arguments[1].number,x.arguments[0].number))

def sparsify_brave_structure(model,ordered_zero_candidates,ASP_options=[]):
    '''
    Brave consequences (union of answer sets) on edges, also indicates that
    some edges could be both present or absent, but it is unclear in which 
    combination. Thus to find a sparse network structure one cannot simply set
    all ambigious edges to zero because that might yield an UNSATISFIABILITY.
    Here the strategy is to iterate through all ambigious edges in order of
    decreasing confidence that they are zero (provided by 
    "ordered_zero_candidates") and everytime to check if they can be set to 
    zero and still yield a satisfiable network. If so, the edge is set to zero.
    
    Requires:
        model['response']['rxn_on_pert'] 
        model['response']['no_rxn_on_pert']
        model['response']['brave edge array']
        model['response']['brave edge dict']
    '''    
    
    my_logger( 'start sparsification \n',importance=2)
    previous_edges_known=model['input']['edges_known'].copy()
    model['response']['brave edge array sparsified']=model['response']['brave edge array'].copy()
    model['response']['brave edge dict sparsified']=model['response']['brave edge dict'].copy()
    
    model['input']['edges_known']={}
    for key,val in model['response']['brave edge dict'].items():
        if len(val)==1 and key[0]!=key[1]:
            model['input']['edges_known'][(key[1],key[0])]=list(val)[0]
       
    skipped_els=0
    to_zero,first_round,current_solution=0,True,{}
    for ind in ordered_zero_candidates:
        if ind[0]==ind[1]: continue
        model['input']['edges_known'][( ind[1],ind[0] )]=0
        if (first_round) or (tuple(ind) in current_solution):
            my_logger('not skipped\n',importance=2)
            first_round=False #The first round can only be satisfiable but I need to run it anyways to populate on_model_edge
            rxn_logic_pgr=prepare_ASP_program(model,'reaction.lp',ASP_options=ASP_options)    
            
            rxn_logic_pgr.load(os.path.join(ASP_programs_folder,'reaction_constraints.lp'))#ASP_programs_folder+'/reaction_constraints.lp')
            rxn_logic_pgr.add("base", [], "#show edge/2.\n")
            rxn_logic_pgr.ground([("base", [])])

            on_model_edge[0]=set([])
            SolveResult=rxn_logic_pgr.solve(on_model=on_model_sparsify)
            satisfiable=SolveResult.satisfiable
            if satisfiable: current_solution=on_model_edge[0].copy()
        else: 
            satisfiable=True
            my_logger('skipped\n',importance=2)
            skipped_els+=1
#        my_logger(satisfiable,on_model_edge[0],'\n',ind, 'in  current_solution',tuple(ind) in current_solution,'\n\n',importance=2)
        if satisfiable:
            to_zero+=1
            model['response']['brave edge dict sparsified'][tuple(ind)]=set([0])
            model['response']['brave edge array sparsified'][ind[0],ind[1]]=0

        else:
            #if it is unsatisfiable it means I cannot remove an ambiguous edge. Since the previous solution was satisfiable
            #it includes this edge and I just keep it as the current_solution
            model['response']['brave edge dict sparsified'][tuple(ind)]=set([1])
            model['response']['brave edge array sparsified'][ind[0],ind[1]]=1.
            model['input']['edges_known'][(ind[1],ind[0])]=np.nan
    
    if len(ordered_zero_candidates): model['response']['sparsification_ratio']=to_zero/float(len(ordered_zero_candidates))
    else: model['response']['sparsification_ratio']=0
    model['input']['edges_known']=previous_edges_known
    
    my_logger('skipped els={0} amongst {1} els'.format(skipped_els,len(ordered_zero_candidates)),importance=2)
    return



def iterate_conforming_networks(model,nbr_answer_sets,heuristics={}):
    '''
    Set nbr_answer_sets to 'all' to iterate over all answer sets. 
    '''
    if nbr_answer_sets==0:
        return [],[]
    if nbr_answer_sets=='all':
        nbr_answer_sets=0
    rxn_logic_pgr=prepare_ASP_program(model,'reaction.lp',heuristics,ASP_options=['-n {}'.format(nbr_answer_sets)])
    rxn_logic_pgr.load(os.path.join(ASP_programs_folder,'reaction_constraints.lp'))
    rxn_logic_pgr.add("base", [], "#show edge/2.\n")
    
    rxn_logic_pgr.ground([("base", [])])
    networks=[]
    SolveResults=[]
#    reduced_edges=[]
#    reduced_reacting=[]
    with rxn_logic_pgr.solve(yield_=True) as handle:    
        for m in handle:
            network=np.zeros([model['N'],model['N']],dtype=bool)
            atoms=m.symbols(shown=True)
            #atoms=m.symbols(atoms=True)
            
#            reduced_edges.append([])
#            reduced_reacting.append([])
            for atom in atoms:
                if atom.name=='edge':
                    network[atom.arguments[1].number,atom.arguments[0].number]=True
#                elif atom.name=='reduced_edge':
#                    reduced_edges[-1].append(atom)
#                elif  atom.name=='reduced_reacting':
#                    reduced_reacting[-1].append(atom)
            networks.append(network)
            SolveResults.append(handle.get())
        
#    print(reduced_edges)#[1])
#    print(reduced_reacting)#[1])
    return networks, SolveResults
        
        
        
        
        
        
        
        

