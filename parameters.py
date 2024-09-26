# g_multi-Copy1.py

# 3 agentów
# wchuj ważne info koło 330 episodów sie przetrenowuje !!!!!!!!!!!!! avg improvement ~1340
# lr = 0.001
# gamma = 0.99
# epsilon = 1.0
# epsilon_decay = 0.995
# buffer_size = 50000
# __________________________________________________________________

# 10 agentów avg improvment ~1450
# lr = 0.0005
# gamma = 0.99
# epsilon = 1.0
# epsilon_decay = 0.995
# buffer_size = 50000
#______________________________________________________________________________



# ***********************************************************************   400 epok
# g_cuda.py
# lr = 0.0005
# gamma = 0.99
# epsilon = 1.0
# epsilon_decay = 0.995
# buffer_size = 50000


# ~ 3000 impr
# lr = 0.005
# gamma = 0.995
# epsilon = 0.1
# epsilon_decay = 0.995
# buffer_size = 50000

# *********************************************************************** 400 epok
# g.py
# improvment ~3200 
# lr=0.001 
# gamma=0.99
# epsilon=1.0
# epsilon_decay=0.98
# buffer_size=100000

# g_multi.py 10 agents 400 episodes ~ średnio 10, bardzo dobrze, można puścić wiecej episodów z mechanizmem zwiekszania lr 
# lr=0.0005 
# gamma=0.99
# epsilon=1.0
# epsilon_decay=0.995
# buffer_size = 20000

# g_multi-shared.py
lr=0.001
gamma=0.99
epsilon=0.1 # exploration ratio 
epsilon_decay=0.995
buffer_size = 10000

live = 15 # za życie 1klatka/15pkt, 1500pkt ~ 1 score                 15
penalty = -1000 # za śmierć -1000                                      -1000
batch_size = 128  # 32/64/256

# INFO WAŻNE
# plany na później, 
# (raczej nie) zmienić live i penalty
# dodać zwiekszony lr co 10 epok np. every 10 epok lr *= 1.01


#tak było kiedyś zawsze !
# agent_cuda.py
# lr=0.001 
# gamma=0.99
# epsilon=1.0
# epsilon_decay=0.98
# buffer_size=100000

# agent2.py
# lr=0.001
# gamma=0.99
# epsilon=1.0
# epsilon_decay=0.995
# buffer_size=10000

#g_multi-Copy1.py, g_cuda.py działam na tych




