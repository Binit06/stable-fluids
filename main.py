import math

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

'''
Stan's stable fluid
'''
n = 48              # grid size (the smaller it is the easier it would be to run at this point)
r = 1.0             # fluid density (/rho) - it always remains constant since it is the density of the liquid
visc = .0001        # viscosity - the higher the viscosity it becomes hard for the water to flow
f = 0               # external force => 0i + 0j

t = 0;              # start time
d = 1;              # time delta
total_time = 200    # total simulation runtime (in secs)

p = [[0.0 for _ in range(n)] for _ in range(n)]             # dynamic pressure being applied
speed = [[(0.0, 0.0) for _ in range(n)] for _ in range(n)]  # speed being updated dynamically
density = [[0.0 for _ in range(n)] for _ in range(n)]       # density to show how dense a section is while water is moving

'''
Boundary conditions

Keep the fluid inside the box.

For scalar fields we copy the neighbouring value onto the boundary.
For velocity we set the boundary velocity to zero, so the walls
behave like solid boundaries.
'''
def boundary_scalar(field):
    for i in range(1, n - 1):
        field[i][0] = field[i][1]
        field[i][n - 1] = field[i][n - 2]
    for j in range(1, n - 1):
        field[0][j] = field[1][j]
        field[n - 1][j] = field[n - 2][j]

    field[0][0] = (field[1][0] + field[0][1]) / 2
    field[0][n - 1] = (field[1][n - 1] + field[0][n - 2]) / 2
    field[n - 1][0] = (field[n - 2][0] + field[n - 1][1]) / 2
    field[n - 1][n - 1] = (field[n - 2][n - 1] + field[n - 1][n - 2]) / 2

def boundary_speed(speed):
    for i in range(1, n - 1):
        speed[i][0] = (0.0, 0.0)
        speed[i][n - 1] = (0.0, 0.0)
    for j in range(1, n - 1):
        speed[0][j] = (0.0, 0.0)
        speed[n - 1][j] = (0.0, 0.0)

    speed[0][0] = (0.0, 0.0)
    speed[0][n - 1] = (0.0, 0.0)
    speed[n - 1][0] = (0.0, 0.0)
    speed[n - 1][n - 1] = (0.0, 0.0)

'''
Gives the poisson correction using this equation
del^2 p(n + 1) = rho (del . u(n))/(delta t) - rho (del . (u(n) (del . u(n)))) + visc(del^2(del . u(n)))
or we can use the fractional method to get rhs and just guess using the below
u((n + 1)/2) = u(n) + delta t / rho (del (p(n + 1)) - we need del (p(n + 1)) and we have all other values so
'''
def poisson_correction(intermediate_speed, x, y):
    du_dx = (intermediate_speed[x + 1][y][0] - intermediate_speed[x - 1][y][0])/2
    dv_dy = (intermediate_speed[x][y + 1][1] - intermediate_speed[x][y - 1][1])/2

    divergence = du_dx + dv_dy
    rhs = (r / d)*divergence

    return rhs

'''
Iterates once over the pressure to generate new pressure and convergence by expanding del^2 p
since pressure changes regularly it is important to maintain this so that del . u = 0 remains
'''
def new_jacobbi_iter(speed):
    np = [row.copy() for row in p]
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            np[i][j] = (p[i + 1][j] + p[i - 1][j] + p[i][j + 1] + p[i][j - 1] - poisson_correction(speed, i, j))/4
    return np

'''
inclusice formulae for diffusion
this is a direct update from how it was handled in Naiver Stokes where we used laplasian of u(old) to get new velocity
u(new) = u(old) + k * delta t * (laplasian(u(new)))
'''
def diffuse_speed(speed, visc, its = 20):
    a = d * visc * n * n
    new_speed = [[(0.0, 0.0) for _ in range(n)] for _ in range(n)]
    
    for k in range(its):
        for x in range(1, n - 1):
            for y in range(1, n - 1):
                new_speed[x][y] = (speed[x][y] + a * (new_speed[x - 1][y] + new_speed[x + 1][y] + new_speed[x][y - 1] + new_speed[x][y + 1]))/(1 + 4*a)
        boundary_speed(new_speed)
    return new_speed

'''
almost similar logic as used in diffuse_speed just read that if required
'''
def diffuse_density(density, kappa, its=20):
    a = d * kappa * n * n
    new_density = [[0.0 for _ in range(n)] for _ in range(n)]

    for k in range(its):
        for x in range(1, n - 1):
            for y in range(1, n - 1):
                new_density[x][y] = (density[x][y] + a * (new_density[x - 1][y] + new_density[x + 1][y] + new_density[x][y - 1] + new_density[x][y + 1])) / (1 + 4 * a)
        boundary_scalar(new_density)
    return new_density

'''
speed with respect to time
originally -> - u ( del . u)
but for stable fluids move the velocity along itself
the original methods makes the velocity freak out so using a semi lagranian scheme to keep it unconidionally stable
i_loc = c_loc - u * delta t
apply bilinear interpolation to get a value for this point
this value becomes the new advect_speed
'''
def advect_speed(speed):
    new_speed = [[(0.0, 0.0) for _ in range(n)] for _ in range(n)]
    for x in range(1, n - 1):
        for y in range(1, n - 1):
            i_loc = (x - speed[x][y][0]*d, y - speed[x][y][1]*d)
            i_loc = (max(0.5, min(n - 1.5, i_loc[0])), max(0.5, min(n - 1.5, i_loc[1])))
            b = (i_loc[0] - math.floor(i_loc[0]), i_loc[1] - math.floor(i_loc[1]))

            m, k = math.floor(i_loc[0]), math.floor(i_loc[1])
            u_top = (speed[m][k][0] + b[0] * (speed[m + 1][k][0] - speed[m][k][0]), speed[m][k][1] + b[0] * (speed[m + 1][k][1] - speed[m][k][1]))
            u_bottom = (speed[m][k + 1][0] + b[0] * (speed[m + 1][k + 1][0] - speed[m][k + 1][0]), speed[m][k + 1][1] + b[0] * (speed[m + 1][k + 1][1] - speed[m][k + 1][1]))

            new_speed[x][y] = (u_top[0] + b[1] * (u_bottom[0] - u_top[0]), u_top[1] + b[1] * (u_bottom[1] - u_top[1]))
    return new_speed

'''
similar logic to advect_speed for the same reason so that density does not freak out
'''
def advect_density(density, speed):
    new_density = [[0.0 for _ in range(n)] for _ in range(n)]
    for x in range(1, n - 1):
        for y in range(1, n - 1):
            i_loc = (x - speed[x][y][0]*d, y - speed[x][y][1]*d)
            i_loc = (max(0.5, min(n - 1.5, i_loc[0])), max(0.5, min(n - 1.5, i_loc[1])))
            b = (i_loc[0] - math.floor(i_loc[0]), i_loc[1] - math.floor(i_loc[1]))
            m, k = int(i_loc[0]), int(i_loc[1])

            d_top = density[m][k] + b[0] * (density[m + 1][k] - density[m][k])
            d_bottom = density[m][k + 1] + b[0] * (density[m + 1][k + 1] - density[m][k + 1])

            new_density[x][y] = d_top + b[1] * (d_bottom - d_top)
    return new_density

'''
runs a super simple function to add dye as time passes
it just increases the density on certain location
so that the velocity can push it ahead then
'''
def update_density(density, speed, kappa, t):
    cx = n // 2 + int(6 * math.sin(t * 0.05))
    cy = 4
    radius = 3

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if (i - cx)**2 + (j - cy)**2 < radius**2:
                density[i][j] = 100.0
    
    density = diffuse_density(density, kappa)
    density = advect_density(density, speed)
    return density

'''
handles the part of making new velocities
so that it can always remain divergence free
its a simple poisson correction on the speed with a lot of iterations
so that convergence can be high
'''
def project(speed):
    global p
    its = 50
    while its > 0:
        p = new_jacobbi_iter(speed)
        boundary_scalar(p)
        its -= 1

    new_speed = [row.copy() for row in speed]
    for i in range(1, n - 1):
        for j in range(1, n - 1):
            dp_dx = (p[i + 1][j] - p[i - 1][j]) / 2
            dp_dy = (p[i][j + 1] - p[i][j - 1]) / 2

            new_speed[i][j] = (speed[i][j][0] - (dp_dx / r) * d, speed[i][j][1] - (dp_dy / r) * d)

    boundary_speed(new_speed)
    return new_speed

'''
follows the cycle for updating the speed on every frame update
-> steer the velocity
-> diffuse speed
-> project (for removing any divergence)
-> advect speed
-> project (for removing any divergence on the new speed)
'''
def update_speed(speed, t, d):
    cx = n//2 + int(6 * math.sin(t * 0.05))
    cy = 4
    radius = 3

    for i in range(1, n - 1):
        for j in range(1, n - 1):
            if (i - cx)**2 + (j - cy)**2 < radius**2:
                speed[i][j] = (
                    speed[i][j][0] + d * 4.0 * 0.5 * math.sin(t * 0.05),
                    speed[i][j][1] + d * 4.0
                )

    speed = diffuse_speed(speed, visc)
    
    speed = project(speed)

    speed = advect_speed(speed)

    speed = project(speed)

    return speed, t + d

'''
runs the simulation using the density matrix
handles coloring and all that itself
use plt.show() if you just wanna see the fluid move
use animation.save() if you wanna store it as mp4
your choice choose one
'''
def run_sim(speed, density):
    steps = round(total_time / d)

    fig, ax = plt.subplots(figsize=(6,6))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    ax.set_title("Navier Stokes - Density")

    img = ax.imshow(density, cmap='inferno', interpolation='bilinear', origin='upper', vmin=0, vmax=100)
    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform = ax.transAxes
    )

    def animate(frame):
        nonlocal speed, density

        current_time = frame * d

        speed, _ = update_speed(speed, current_time, d)
        density = update_density(density, speed, 0.001, current_time)

        img.set_data(density)
        time_text.set_text(f"t = {current_time + d:.2f}")

        return img, time_text
    
    animation = FuncAnimation(
        fig,
        animate,
        frames=steps,
        interval=50,
        blit=False,
        repeat=False
    )

    animation.save("fluid_simulation.mp4", writer="ffmpeg", fps=20)
    # plt.show()

if __name__ == "__main__":
    run_sim(speed, density)
