
# coding: utf-8

# ## Introduction:
# 
# BZ reaction was first discovered by a Russian scientist Belousov. In this reaction the system undergoes a reaction with an oscillatory change of color. For a long time his findings were dismissed, since people were uncomfortable with an idea that system may go back and forth from the thermodynamical equilibrium. It was much later when Zhabotinsky confirmed the findings of Belousov, the reaction became so famous reaction, that UBC CHBE department decided to use that to show off CHBE research to the high school students.
# 

# In[1]:


from IPython.display import YouTubeVideo
YouTubeVideo('wxSa9BMPwow')


# The overall BZ-reaction is the oxidation of malonic
# acid by $BrO^{–}_3$ to form $CO_2$ and $H_2O$:
# 
# $ 3 CH_2 (CO_2H)_2 + 4 BrO^{-}_3 \to 4 Br^{-} + 9 CO_2 + 6 H_2 O$
# 
# It looks simple, but don't be deceived by the looks. Below are shown all the reactions happening in the system.

# In[4]:


from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "http://www.scholarpedia.org/w/images/5/5f/BZ_Core_scheme.gif", width=200, height=200)


# ## Mathematical model
# 
# The simplest model to explain the BZ reaction is the oregonator model [A. Garza 2011]. In this model the important chemical species are
# \begin{align}
# A&= {\rm BrO_3^-}\ ,& \  P&= {\rm HOBr} \ ,& \ B&={\rm oxidizable\
# organic\ species} \nonumber \\
# X&= {\rm HBrO_2} \ ,& \  Y&= {\rm Br^-} \ ,& \ Z&={\rm Ce}^{4+}
# \end{align}
# 
# 
# and its dynamics is described by the scheme
# 
# \begin{align}
# &A+Y \stackrel{k_1}{\longrightarrow} X+P  \ (1),& \ X+Y&
# \stackrel{k_2}{\longrightarrow} 2P  \ (2),&
# \ A+X& \stackrel{k_3}{\longrightarrow} 2X+2Z \ (3), \nonumber \\
# &2X \stackrel{k_4}{\longrightarrow} A+P \ (4),& \ B+Z&
# \stackrel{k_5}{\longrightarrow} \frac{f}{2} Y \ . (5)&
# \end{align}
# 
# 
# The first two reactions describe the consumption of bromide
# Br$^-$, whereas the last three ones model the buildup of HBrO$_2$
# and Ce$^{4+}$ that finally leads to bromide recovery, and then to
# a new cycle.
# 
# By assuming that the bromate concentration $[A]$
# remains constant as well as $[B]$, and noting that $P$ enters only
# as a passive product of the dynamics, the law of mass action leads
# to
# 

# 
# 
# $\frac{dX}{dt} = k_1 A Y - k_2 X Y + k_3 A X - 2 k_4 X^2$
# 
# $\frac{dY}{dt} = -k_1 A Y - k_2 X Y + k_5 \frac{f}{2} B Z$
# 
# $\frac{dZ}{dt} = 2 k_3 A X - k_5 B Z$ 
# 
# 
# 
# In chemical engineering we always want to bring our variables to dimensionless values. We can do that here by rescaling time and concentrations:
# 
# $x= \dfrac{X}{X_0}  $,
# $y= \dfrac{Y}{Y_0}$,
# $z= \dfrac{Z}{Z_0}$,
# $\tau= \dfrac{t}{t_0}$
# 
# where the scaling factors are:
# 
# 
# $X_0=\frac{k_3 A}{2 k_4}$
# 
# $Y_0 = \frac{k_3 A}{k_2} $
# 
# $Z_0 = \frac{(k_3 A)^2}{k_4 k_5 B}$
# 
# $t_0 = \frac{1}{k_5 B}  $
# 
# $\epsilon_1=\frac{k_5 B}{k_3 A} $
# 
# $\epsilon_2= \frac{2 k_4 k_5 B}{k_2 k_3 A} $
# 
# $q= \frac{2 k_1 k_4}{k_2 k_3}$
# 
# In terms of these variables, the model reads (*)
# 
# $\epsilon_1 \frac{dx}{d\tau} = q y -x y +x(1-x)$ 
# 
# $\epsilon_2 \frac{dy}{d\tau} = -q y - x y + f z $
# 
# $\frac{dz}{d\tau} =  x - z $ 
# 
# At certain combination of the parameter values for the system it shows an oscillatory behaviour.
# 
# Use these parameters for this model:
# 
# $\epsilon_1 = 9.9 × 10^{−3}$,
# $\epsilon_2 = 1.98×10^{−5}$,
# $q = 7.62×10^{−5}$,
# $f = 1$, 
# $x(t = 0) = 0$, 
# $y(t = 0) = 0.001$,
# $z(t = 0) = 0$.
# 
# 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from ipywidgets import interact
from IPython.display import clear_output, display, HTML

def BZrxn(A, B, x_init, y_init, z_init, P_init):
    #A = 0.06 #BrO3-
    #B = 0.02 #Oxidizable organic species

    k1 = 1.28
    k2 = 2.4*10**6
    k3 = 33.6
    k4 = 2400
    k5 = 1

    e1 = 9.9*10**-3
    e2 = 1.98*10**-5
    q = 7.62*10**-5
    f = 1.

    #x_init = 1
    #y_init = 1
    #z_init = 1

    X0 = k3*A/(2*k4)
    Y0 = k3*A/k2
    Z0 = (k3*A)**2/(k4*k5*B)
    t0 = 1/(k5*B)
    #P_init = 0 # No product formed at beginning of rxn

    t_init = 0
    t_final = 50
    N = 10000
    t_array = np.linspace(t_init, t_final, N)


    def ode_xyzp(dim_list, t_array):
        x = dim_list[0]
        y = dim_list[1]
        z = dim_list[2]

        dxdtau = (q*y - x*y + x*(1-x))/e1
        dydtau = (-q*y - x*y + f*z)/e2
        dzdtau = x - z
        dpdtau = k1*A*y*Y0 + 2*k2*x*X0*y*Y0 + k4*(x*X0)**2

        return [dxdtau, dydtau, dzdtau, dpdtau]

    dim_num_list = odeint(ode_xyzp, [x_init, y_init, z_init, 0], t_array) #4th param is not used therefore it is 0
    x_num = dim_num_list[:,0]
    y_num = dim_num_list[:,1]
    z_num = dim_num_list[:,2]
    p_num = dim_num_list[:,3]

    X = x_num*X0 # X = x*X0
    Y = y_num*Y0 # Y = y*Y0
    Z = z_num*Z0 # Z = z*Z0
    P = p_num*t0

    plt.figure(0)
    plt.plot(t_array, np.log(x_num), 'r--', label = "x")
    plt.plot(t_array, np.log(y_num), 'b--', label = "y")
    plt.plot(t_array, np.log(z_num), 'g--', label = "z")
    plt.xlabel('tau')
    plt.ylabel("log of dimensionless values")

    plt.figure(1)
    plt.plot(t_array, (X), 'r--', label = "X")
    plt.plot(t_array, (Y), 'b--', label = "Y")
    plt.plot(t_array, (Z), 'g--', label = "Z")
    plt.plot(t_array, (P), 'k--', label = "P")
    plt.xlabel('time (s)')
    plt.ylabel('C (mol/L)')
    plt.legend()

    fig = plt.figure(2)
    plt.plot(t_array, X, 'r--', label = "X")
    plt.xlabel('time (s)')
    plt.ylabel('C (mol/L)')
    plt.legend()

    fig = plt.figure(3)
    plt.plot(t_array, Y, 'b--', label = "Y")
    plt.xlabel('time (s)')
    plt.ylabel('C (mol/L)')
    plt.legend()

    fig = plt.figure(4)
    plt.plot(t_array, Z, 'g--', label = "Z")
    plt.xlabel('time (s)')
    plt.ylabel('C (mol/L)')
    plt.legend()

    fig = plt.figure(5)
    plt.plot(t_array, P, 'k--', label = "P")
    plt.xlabel('time (s)')
    plt.ylabel('C (mol/L)')
    plt.legend()

    fig = plt.figure(6)
    ax = fig.gca(projection='3d')
    ax.plot(X, Y, Z, label='3D plot of X, Y, and Z')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.ylabel('Z')
    ax.legend()
    plt.show()

interact(BZrxn, 
         A = (0,1.2,0.1), 
         B = (0,0.4,0.1), 
         x_init = (0, 2, 0.5), 
         y_init = (0,2, 0.5), 
         z_init = (0,2,0.5), 
         P_init = (0,1,0.5))

# 3
# the concentrations of X, Y, and Z oscillate in a contained range of concentration after an initial spike.
# P cyclically increases over time
# A and B are constant (in excess for the reaction)

# 4
# X, Y, and Z are intermediate products 
# therefore they are able to still oscillate while entropy increases.
# Entropy increases as P increases and the reactants (A and B) are used up. 
# Because the reactants are in excess, the concentrations of X, Y, and Z are able to oscillate
# in a contained range


# ## Problem statement:
# 
# 1. Solve the ODE (*) for the  concentrations of intermediaries X, Y, Z as well as the product P.
# 2. Plot your curves
# 3. What can you tell about the behvaiour of each species?
# 4. Generally, the concentration of all products in chemical reactions have to increase (otherwise the entropy of the universe is decreasing and our thermodynamics teachers wouldn't be happy) However, something is clearly oscillating here. How can you explain this?
# 
# 5. (Bonus) Create a slider using the template that we used for our Zombie example and play with the input parameters.
# 6. (Bonus) Plot a 3D plot with the concentrations of X, Y, and Z in a 3d diagram
# 
# you may use the template below:
# https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
# ```python
# import matplotlib as mpl
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# import matplotlib.pyplot as plt
# 
# mpl.rcParams['legend.fontsize'] = 10
# 
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
# z = np.linspace(-2, 2, 100)
# r = z**2 + 1
# x = r * np.sin(theta)
# y = r * np.cos(theta)
# ax.plot(x, y, z, label='parametric curve')
# ax.legend()
# 
# plt.show()

# References:
# 
# 1. Garza 2011
# https://pdfs.semanticscholar.org/2876/0e30e84817a29a22966fcde4fd619d6eeabb.pdf
# 2. R. Noyes 1989 
# https://pubs.acs.org/doi/pdf/10.1021/ed066p190
