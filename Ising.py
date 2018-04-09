import pylab as plt 
import numpy as N
import random

# ---------------------------- Variables --------------------------------
 
T = 1.0 # definition Temperature (T)
 
m = 50 # x*x x = 50 
 
nt=100 # time interval for each image
 

#we start our system in different temperatures 
 
J = 1.0 # H = j\sum m*n    (ferromagnetisme J > 0)
kb = 1.0 # Boltzmann 
H = 0  # system impact
 
 
#define our obsersation 
n = int(m) # n*n
 
# Magnetisation par site : 
 
def magnetisation_par_spin():
 
	s_sum = 0
 
	for i in range(0,n):
		for j in range(0,n):
			s_sum=s_sum+a[i,j]
 
	return s_sum*1. / (m*m)
 
 
# Energie par site :
 
 
def energie_par_spin():
 
	s_sum = 0
	ss_sum = 0
 
	for i in range(0,n):
		for j in range(0,n):
			s_sum= s_sum+a[i,j]
			ss_sum= ss_sum+a[i,j]*(a[(i-1)%n,j]+a[(i+1)%n,j]+a[i,(j-1)%n]+a[i,(j+1)%n])
 
	return -(J * ss_sum + H * s_sum)*1. / (m*m)
	 
# ----------------------------- Simulation ------------------------------    
 
# Start 
 
a = N.zeros((n,n),dtype=int) 
# generate -1 or +1 per site
 
for i in range(0,n):
	for j in range(0,n):

		a[i,j]=random.randrange(0,2)
		 
		if a[i,j]==0: 
			a[i,j]=-1
			 
		elif a[i,j]==1: 
			a[i,j]=1

# Creation of a copy to allow to have an image of our initial matrix:
 
d=a.copy()
 
# We simulate our Ising model on a time step to reach our equilibrium
# and get an evolution of the system at each time step. In addition we put a condition
# so that once the system is at equilibrium it stops alone.
# The terminals for our time intervals:
 
t1=0
t2=t1+nt
 
# The average magnetization is calculated for each time interval in order to be able to
# compare it to the previous one to know if the system is at equilibrium. 
Mmoy=0
Mmoyold=10
 
# We store all magnetization values and energy in a list.
 
M_all=[]
E_all=[]
 
# We create matrices E_t and M_t in order to store over a time interval.
 
E_t=N.zeros(nt)
M_t=N.zeros(nt)
 
# Creation of our file for each simulation
 
 
# We define a counter in order to follow the evolution of the system.
 
compteur=0
 
# Condition to stop the system when it reaches equilibrium.
 
while(abs((Mmoy-Mmoyold)/(Mmoyold+1.0E-15))>0.0001):
	 
	compteur=compteur+1
	print (compteur)
	if compteur==150 :
		break
	 
	# System evolves according to the time in order to reach its equilibrium state.
	
	for t in range(t1,t2) :
 
		# We are going to make evolve the system at a given instant t while studying
		# a spin change.
		for k in range(0,n*n):
 
			# We draw a random spin and calculate a possible change
			# of spin, we start again n * n times.
			i=random.randint(0,n-1) #random value [0, n-1] 
			j=random.randint(0,n-1)
 
			# Energy spin with its first neighbor has a moment of time. 
			e=2*J*a[i,j]*(a[(i-1)%n,j]+a[(i+1)%n,j]+a[i,(j-1)%n]+a[i,(j+1)%n])
			#print(e)
			# If the energy is negative then there is automatically a switchover:
   
			if e<0 : 
				a[i,j]=-a[i,j]
 
			# If the energy is positive, we deduce a spin change
			# with a probability P.
			 
			elif e>=0 :
 
				P=N.exp(-e/(kb*T))
				#print(P)
			
				# We draw a random number m between 0 and 1 we switch if P> m:
 
				if random.random() < P :
					a[i,j]=-a[i,j]
				else :
					a[i,j]=a[i,j]
								 
 
		
 
		# We store the observables in a table on time:
												
		E=energie_par_spin()             
		M=magnetisation_par_spin()
		 
		E_t[t]=E
		M_t[t]=M

		 
	# The average of the global magnetization before and the end of each interval
	# of times in order to observe or not the state of equilibrium.	 
	Mmoyold=Mmoy
	#print(M_t)
	Mmoy=N.mean(M_t)
	#print(Mmoy)
	# We deduce magnetization and energy for each interval
	 
	E_t2=N.copy(E_t)
	M_t2=N.copy(M_t)
	 
	# We put end to end our different arrays:
	 
	E_all.append(E_t2)
	M_all.append(M_t2)
	 
	 
	# We take an image of our network at the end of each time loop.
 
	#plt.imshow(a,interpolation="nearest")
	 
	#plt.savefig('image_'+str(compteur)+'.jpg')
	 
	 
# A la fin de notre boucle on reconstitue tous les arrays afin de pouvoir observer
# l'evoluation de la Magnetisation et l'Energie sur l'ensemble de la simulation.
			 
E_all_array=N.array(E_all)
M_all_array=N.array(M_all)
 
 
		  
# ---------------------------- Figures -------------------------------------                       
 
# Figure (4) corresponds to the end matrix.

 
plt.figure(4)
 
plt.imshow(a,interpolation="nearest")
plt.title('Network End')
 
# Figure(3) correspond a l'evolution de la magnetisation et de l'energie.
 
plt.figure(3)
 
plt.subplot(121)
plt.plot(M_all_array.flatten())
plt.title('Evolution of Magnetization')
plt.xlabel('times')
plt.ylabel('Average of global magnetization')   
 
plt.subplot(122)
plt.plot(E_all_array.flatten())
plt.title('Evolution of Energy')
plt.xlabel('times')
 
# Figure(2) correspond a la matrice du debut.
 
plt.figure(2)
plt.imshow(d,interpolation="nearest")
plt.title('Network start')
 
plt.show()
