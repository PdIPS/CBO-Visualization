from js import document, window
from pyodide.ffi import create_proxy
from utils import viridis
import math
import numpy as np
from cbx.objectives import Ackley
from cbx.dynamics import CBO



# %% Drawing utils
# -----------------------------------------------------------------------------
def draw_particle(X, ctx, WIDTH, HEIGHT, scale, color="rgb(230,0,0)", r=1):
    """
    Draws a particle at position X
    """
    x = ((X[0] / (2 * scale)) + 0.5) * WIDTH
    y = ((X[1] / (2 * scale)) + 0.5) * HEIGHT
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.arc(x, y, 3 * r, 0, 2 * math.pi)
    ctx.fill()

def draw_line(X1,X2, ctx, WIDTH, HEIGHT, scale, color="rgb(230,0,0)"):
    """
    Draws a line from X1 to X2
    """
    x1 = ((X1[0] / (2 * scale)) + 0.5) * WIDTH
    y1 = ((X1[1] / (2 * scale)) + 0.5) * HEIGHT
    x2 = ((X2[0] / (2 * scale)) + 0.5) * WIDTH
    y2 = ((X2[1] / (2 * scale)) + 0.5) * HEIGHT
    ctx.strokeStyle = color
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(x1, y1)
    ctx.lineTo(x2, y2)
    ctx.stroke()

class CBO_artist:
    def __init__(self, ctx, WIDTH = 100, HEIGHT = 100 , scale = 10):
        self.ctx = ctx
        self.WIDTH  = WIDTH
        self.HEIGHT = HEIGHT
        self.scale = scale

        XX, YY = ((Z - 0.5) * scale * 2 for Z in np.meshgrid(np.linspace(0, 1, self.WIDTH), np.linspace(0, 1, HEIGHT)))
        self.XY = np.stack((XX.T,YY.T)).T

    def draw_contour(self, loss):
        # Background
        self.ctx.fillStyle = "white"
        self.ctx.fillRect(0, 0, self.WIDTH, self.HEIGHT)

        Z = loss(self.XY)
        Z = (Z - Z.min())/ (Z.max() - Z.min())

        # Draw contour (optional: coarse grid)
        for xi in range(0, self.WIDTH, 10):
            for yi in range(0, self.HEIGHT, 10):
                z = Z[yi, xi] #loss(np.array([xx, yy]))

                val = np.clip(z, 0, 1)
                self.ctx.fillStyle = viridis(val)    
                self.ctx.fillRect(xi, yi, 10, 10)

    
    def draw_particles(self, DYN):
        for i in range(DYN.N): # Draw particles
            draw_line(DYN.x[0, i, :], DYN.consensus[0, 0, :], 
                      self.ctx, 
                      self.WIDTH, self.HEIGHT, self.scale, 
                      color="rgb(170,0,0, 0.3)")
            draw_particle(DYN.x[0, i, :], self.ctx, 
                          self.WIDTH, self.HEIGHT, self.scale, color="rgb(230,0,0)")
            
        draw_particle(DYN.consensus[0, 0, :], self.ctx, 
                      self.WIDTH, self.HEIGHT, self.scale, 
                      color="rgb(200,200,200)", r=3)  # Draw consensus point
#%% Set up Canvas
canvas = document.getElementById("canvas")
if canvas is None:
    print("Canvas not found.")
ctx = canvas.getContext("2d")
window.spinning = False
ctx.clearRect(0, 0, canvas.width, canvas.height)
WIDTH, HEIGHT = canvas.width, canvas.height


# %% Sliders
# -----------------------------------------------------------------------------
class AppParams:
    def create_slider(self, name, val, slider_name):
        setattr(self, name, val)

        def local_on_slider_change(event):
            setattr(self, name, event.target.value)

        slider = document.getElementById(slider_name)
        slider.addEventListener("input", create_proxy(local_on_slider_change))

AP = AppParams()

SliderParams = [
    ("cbo_sigma", 1.5, "sigma-slider"), 
    ("cbo_alpha", 10,  "alpha-slider"),
    ("cbo_dt",   0.1,  "dt-slider"),
    ("fps",       10,  "fps-slider")
]

for name, val, slider_name in SliderParams: AP.create_slider(name, val, slider_name)



mouse_pos = {"x": 0, "y": 0}
def on_mouse_move(event):
    mouse_pos["x"] = event.offsetX
    mouse_pos["y"] = event.offsetY
mouse_move_proxy = create_proxy(on_mouse_move)
canvas.addEventListener("mousemove", mouse_move_proxy)


# SIGMA for Gaussians
SIGMA = 10  # initial value, global
GAUSSIANS = []
def on_wheel(event):
    global SIGMA
    delta = event.deltaY
    # deltaY > 0 means scroll down (decrease), <0 scroll up (increase)
    SIGMA += -delta * 0.01  # scale as needed
    SIGMA = max(1, min(SIGMA, 100))  # clamp sigma between 1 and 100
    event.preventDefault()  # optional, prevent page scroll

wheel_proxy = create_proxy(on_wheel)
canvas.addEventListener("wheel", wheel_proxy)

def on_click(event):
    if event.button == 0:  # left click
        # Add a new Gaussian at the mouse position
        mu = np.array([(mouse_pos["x"] / WIDTH - 0.5), (mouse_pos["y"] / HEIGHT - 0.5)]) * scale * 2
        GAUSSIANS.append((mu, SIGMA))

        if len(GAUSSIANS) > 4:  # limit number of Gaussians
            GAUSSIANS.pop(0)  # remove the oldest Gaussian

click_proxy = create_proxy(on_click)
canvas.addEventListener("click", click_proxy)


scale = 10.0


# Build loss function
def loss(x):
    val = 0.0
    for mu, sigma in GAUSSIANS:
        val -= np.exp(-np.linalg.norm(x - mu)**2 / (2*sigma**2))

    # Add mouse position influence
    mouse_mu = np.array([(mouse_pos["x"] / WIDTH - 0.5), (mouse_pos["y"] / HEIGHT - 0.5)]) * scale * 2

    f = Ackley(minimum=mouse_mu[None,None,:])
    val += f(x)  # Add Ackley function influence
    #val -= np.exp(-np.linalg.norm(x - mouse_mu)**2 / (2*SIGMA**2))
    return val


# %% Consensus-based optimization setup
# -----------------------------------------------------------------------------
def post_process_cbx(dyn):
    dyn.x = np.clip(dyn.x, -1.5* scale, 1.5*scale)
# Load CBX

x = np.random.uniform(-scale, scale, (1, 20, 2))  # 20 particles in 2D
DYN = CBO(
    loss,
    x=x,
    post_process=post_process_cbx,
    track_args = {'names':[]},
)

def set_CBO_params(DYN, AP, loss):
    DYN.sigma = getattr(AP, 'cbo_sigma', 1.)
    DYN.alpha[:] = getattr(AP, 'cbo_alpha', 10.)
    DYN.dt = getattr(AP, 'cbo_dt', 1.)
    DYN.f = loss 

set_CBO_params(DYN, AP, loss)

def update_CBO(DYN, AP, loss):
    set_CBO_params(DYN, AP, loss)  # Update CBO parameters
     # Update loss function
    DYN.step()


#%% Set up Artist
CBOA = CBO_artist(ctx, WIDTH = WIDTH, HEIGHT = HEIGHT, scale = scale)

#%% Frame Rate handling
last_time = 0
max_frame_rate = 10

def animate(time):
    global last_time, CBOA, DYN, AP, loss

    dt = time - last_time
    fps = 1000 / dt if dt > 0 else 0
    print(f"FPS: {fps:.1f}")

    if fps < getattr(AP, 'fps', 10):
        update_CBO(DYN, AP, loss)
        CBOA.draw_contour(loss)
        CBOA.draw_particles(DYN) 
        last_time = time
    window.requestAnimationFrame(animate_proxy)
    
    

animate_proxy = create_proxy(animate)
window.requestAnimationFrame(animate_proxy)
    
#window.requestAnimationFrame(animate)