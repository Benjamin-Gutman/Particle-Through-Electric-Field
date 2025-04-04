import math
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

#All Numerical Constants Used:
mass_proton = 1.67*10**(-27) #in kg
proton_charge = 1.6*10**(-19)
g = -9.8
k = 8.99*10**9
r = 2

#All Basic Physical Equations

def Gravitational_Force(m):
  '''function that returns the gravitational force on a particle given its mass'''
  return m*g

def distance(xp, xq, yp, yq, zp, zq):
  '''function that returns the distance between points P and Q'''
  dist = ((xp-xq)**2+(yp-yq)**2+(zp-zq)**2)**0.5
  return dist

def E(Q, dist):
  '''function that returns the magnitude of an electric field of a charge given the charge Q and a distance'''
  e = (k*Q)/(dist**2)
  return e

def Ex(E, dist, xp, xq):
  '''function that returns the x-component an electric field of a charge given the magnitude of the field, a distance, and the x coordinates of the point charge and the point in space'''
  Ex = E*(xp-xq)/(dist)
  return Ex

def Ey(E, dist, yp, yq):
  '''function that returns the y-component an electric field of a charge given the magnitude of the field, a distance, and the y coordinates of the point charge and the point in space'''
  Ey = E*(yp-yq)/(dist)
  return Ey

def Ez(E, dist, zp, zq):
  '''function that returns the z-component an electric field of a charge given the magnitude of the field, a distance, and the z coordinates of the point charge and the point in space'''
  Ez = E*(zp-zq)/(dist)
  return Ez

def acc_x(Ex):
  '''funciton that returns the x-acceleration of a particle given the x-component of the field'''
  Fx = Ex*proton_charge
  ax = Fx/mass_proton
  return ax

def acc_y(Ey):
  '''funciton that returns the y-acceleration of a particle given the y-component of the field'''
  Fy = Ey*proton_charge
  Fg = Gravitational_Force(mass_proton)
  ay = (Fy+Fg)/mass_proton
  return ay

def acc_z(Ez):
  '''funciton that returns the z-acceleration of a particle given the z-component of the field'''
  Fz = Ez*proton_charge
  az = Fz/mass_proton
  return az

def total_field(theta_range, x_range, xp, yp, zp,r):
  '''function that finds the field of a cylinder given its radius r, ranges of theta and x values (the length of the cylinder) and a point P in space'''
  Ex_tot = 0
  Ey_tot = 0
  Ez_tot = 0
  for i in x_range:
    for j in theta_range:
      y = r*math.sin(j)
      z = r*math.cos(j)
      Q = proton_charge
      dist = distance(xp, i, yp, y, zp, z)
      e = E(Q, dist)
      Ex_tot += Ex(e, dist, xp, i)
      Ey_tot += Ey(e, dist, yp, y)
      Ez_tot += Ez(e, dist, zp, z)
  return Ex_tot, Ey_tot, Ez_tot
theta_range = np.arange(0,2*math.pi,0.1)
x_range = np.arange(-2,2,0.1)



def total_field_vector(theta_range, x_range, xp, yp, zp,r):
  '''Same function as total_field but masks all values that fall within a range too close to the cylinder'''
  Ex_tot = 0
  Ey_tot = 0
  Ez_tot = 0
  yp_masked = ma.masked_less_equal(yp, -7)
  zp_masked = ma.masked_less_equal(zp, -7)
  yp_mask = yp_masked.mask
  zp_mask = zp_masked.mask
  final_mask = np.logical_and(yp_mask, zp_mask)
  for i in x_range:
    for j in theta_range:
      y = r*math.sin(j)
      z = r*math.cos(j)
      Q = proton_charge
      xp_masked = ma.masked_less_equal(abs(xp-i), 0.05)
      yp_masked = ma.masked_less_equal(abs(yp-y), 0.05)
      zp_masked = ma.masked_less_equal(abs(zp-z), 0.05)
      xp_mask = xp_masked.mask
      yp_mask = yp_masked.mask
      zp_mask = zp_masked.mask
      current_mask = np.logical_and(yp_mask, zp_mask)
      current_mask = np.logical_and(current_mask, xp_mask)
      dist = distance(xp, i, yp, y, zp, z)
      e = E(Q, dist)
      Ex_tot += Ex(e, dist, xp, i)
      Ey_tot += Ey(e, dist, yp, y)
      Ez_tot += Ez(e, dist, zp, z)
      final_mask = np.logical_or(final_mask, current_mask)
  return Ex_tot, Ey_tot, Ez_tot, final_mask
theta_range = np.arange(0,2*math.pi,math.pi/100)
x_range = np.arange(-2,2,0.05)

#------------Vector Plot (Y,Z, while X = 0 )
# Creating arrow
plt.figure(figsize=(10,10))
# Make the grid
y, z = np.meshgrid(
                      np.arange(-6, 6, 0.05),
                      np.arange(-6, 6, 0.05))

# Make the direction data for the arrows
u,v,w,final_mask = total_field_vector(theta_range, x_range, 0, y, z, r)
v = ma.array(v, mask =  final_mask)
w = ma.array(w, mask =  final_mask)

plt.quiver(y, z, v, w)
plt.xlabel("z (m)")
plt.ylabel("y (m)")


plt.show()

#------------Vector Plot (X,Z, while Y = 0 )
# Creating arrow
plt.figure(figsize=(10,10))
# Make the grid
x, y = np.meshgrid(
                      np.arange(-6, 6, 0.05),
                      np.arange(-6, 6, 0.05))

# Make the direction data for the arrows
v,w,u,final_mask = total_field_vector(theta_range, x_range, x, y, 0, r)
v = ma.array(v, mask =  final_mask)
w = ma.array(w, mask =  final_mask)

plt.quiver(x, y, v, w)
plt.xlabel("x (m)")
plt.ylabel("y (m)")


plt.show()

STEPS = 1000
DT = 0.001

def trajectory(x_0, y_0, z_0, vx_0, vy_0, vz_0 ):
  '''function that returns lists containg all points a particle in the electric field passes through given initial position and velocity'''
  Ex_0, Ey_0, Ez_0 = total_field(theta_range, x_range, x_0, y_0, z_0, r)
  t = [0]
  ax_0 = acc_x(Ex_0)
  ay_0 = acc_y(Ey_0)
  az_0 = acc_z(Ez_0)
  x = [x_0]
  y = [y_0]
  z = [z_0]
  vx = [vx_0]
  vy = [vy_0]
  vz = [vz_0]
  ax = [ax_0]
  ay = [ay_0]
  az = [az_0]
  for i in range(STEPS):
    Ex, Ey, Ez = total_field(theta_range, x_range, x[i], y[i], z[i], r)
    t.append(t[i] + DT)
    ax.append(acc_x(Ex))
    ay.append(acc_y(Ey))
    az.append(acc_z(Ez))
    vx.append(vx[i] + ax[i]* DT)
    vy.append(vy[i] + ay[i]* DT)
    vz.append(vz[i] + az[i]* DT)
    x.append(x[i] + vx[i]* DT)
    y.append(y[i] + vy[i]* DT)
    z.append(z[i] + vz[i]* DT)
    for k in theta_range:
      j = r*math.sin(k)
      h = r*math.cos(k)
      if abs(y[i+1]-j) <= 0.04 and abs(z[i+1]-h) <= 0.04:
        return x, y, z
    if abs(x[i+1]) >= 6:
      break
  return x,y,z





print(total_field(theta_range, x_range, -1, 0, 0, r))


x, y, z = trajectory(-2, 0, 0, 20, 5, 8 )
ax = plt.axes(projection='3d')


# Data for three-dimensional scattered points
def plot_cylinder(x_range, theta_range):
  x_cyl = []
  y_cyl = []
  z_cyl = []
  for i in x_range:
    for j in theta_range:
      y = r*math.sin(j)
      z = r*math.cos(j)
      x_cyl.append(i)
      y_cyl.append(y)
      z_cyl.append(z)
  return x_cyl, y_cyl, z_cyl

x_cyl,y_cyl,z_cyl = plot_cylinder(x_range, theta_range)

# Data for three-dimensional scattered points
zdata = z
xdata = x
ydata = y
ax.scatter3D(xdata, zdata, ydata);
ax.scatter3D(x_cyl,y_cyl,z_cyl, alpha = 0.1, color = 'green');


plt.figure()
plt.plot(x, y, marker='o', linewidth=0.5, markersize=1) # The 'o' select round markers
plt.plot(x_cyl, y_cyl, color = 'green', marker='o', linewidth=0.5, alpha = 0.2, markersize=1)
plt.legend()
plt.title("")
plt.xlabel("x (m)")
plt.ylabel("y (m)")


plt.figure()
plt.plot(z, y, marker='o', linewidth=0.5, markersize=1)
plt.plot(z_cyl, y_cyl, color = 'green', marker='o', linewidth=0.5, alpha = 0.2, markersize=1)
plt.legend()
plt.title("")
plt.xlabel("z (m)")
plt.ylabel("y (m)")
