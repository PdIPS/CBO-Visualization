from js import document, window
from pyodide.ffi import create_proxy
from utils import viridis
import math
import numpy as np
from cbx.objectives import Ackley
from cbx.dynamics import CBO


canvas = document.getElementById("canvas")
if canvas is None:
    print("Canvas not found.")
ctx = canvas.getContext("2d")
WIDTH, HEIGHT = canvas.width, canvas.height


# Sliders
CBO_SIGMA = 1.5  # initial value, global
CBO_ALPHA = 10.  # initial value, global
CBO_DT    = 0.1  # initial value, global

def on_slider_change_SIGMA(event):
    global CBO_SIGMA
    CBO_SIGMA = float(event.target.value)

CBO_SIGMA_SLIDER = document.getElementById("sigma-slider")
CBO_SIGMA_SLIDER_proxy = create_proxy(on_slider_change_SIGMA)
CBO_SIGMA_SLIDER.addEventListener("input", CBO_SIGMA_SLIDER_proxy)

def on_slider_change_ALPHA(event):
    global CBO_ALPHA
    CBO_ALPHA = float(event.target.value)

CBO_ALPHA_SLIDER = document.getElementById("alpha-slider")
CBO_ALPHA_SLIDER_proxy = create_proxy(on_slider_change_ALPHA)
CBO_ALPHA_SLIDER.addEventListener("input", CBO_ALPHA_SLIDER_proxy)

def on_slider_change_DT(event):
    global CBO_DT
    CBO_DT = float(event.target.value)

CBO_DT_SLIDER = document.getElementById("dt-slider")
CBO_DT_SLIDER_proxy = create_proxy(on_slider_change_DT)
CBO_DT_SLIDER.addEventListener("input", CBO_DT_SLIDER_proxy)



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


def post_process_cbx(dyn):
    dyn.x = np.clip(dyn.x, -1.5* scale, 1.5*scale)
# Load CBX

x = np.random.uniform(-scale, scale, (1, 20, 2))  # 20 particles in 2D
DYN = CBO(
    loss,
    x=x,
    sigma = CBO_SIGMA,  # Use the global CBO_SIGMA
    post_process=post_process_cbx,
)

def draw_particle(X, color="rgb(230,0,0)", r=1):
    global ctx, WIDTH, HEIGHT, scale
    x = ((X[0] / (2 * scale)) + 0.5) * WIDTH
    y = ((X[1] / (2 * scale)) + 0.5) * HEIGHT
    ctx.fillStyle = color
    ctx.beginPath()
    ctx.arc(x, y, 3 * r, 0, 2 * math.pi)
    ctx.fill()

def draw_line(X1,X2, color="rgb(230,0,0)"):
    global ctx, WIDTH, HEIGHT, scale
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


def set_CBO_params():
    global DYN, CBO_SIGMA, CBO_ALPHA, CBO_DT, loss
    DYN.sigma = CBO_SIGMA  # Update sigma in the CBO dynamics
    DYN.alpha[:] = CBO_ALPHA
    DYN.dt = CBO_DT
    DYN.f = loss 

def update_and_draw_CBO(ts):
    global DYN, ctx, WIDTH, HEIGHT, scale
    set_CBO_params()  # Update CBO parameters
     # Update loss function
    DYN.step()


    for i in range(DYN.N): # Draw particles
        draw_line(DYN.x[0, i, :], DYN.consensus[0, 0, :], color="rgb(170,0,0, 0.3)")
        draw_particle(DYN.x[0, i, :], color="rgb(230,0,0)")
        
    draw_particle(DYN.consensus[0, 0, :], color="rgb(200,200,200)", r=3)  # Draw consensus point

MIN, MAX = 0., 1.0

def draw_contour(ts):
    global MIN, MAX
    # Background
    ctx.fillStyle = "white"
    ctx.fillRect(0, 0, WIDTH, HEIGHT)


    # Draw contour (optional: coarse grid)
    for xi in range(0, WIDTH, 10):
        for yi in range(0, HEIGHT, 10):
            xx = (xi / WIDTH - 0.5) * scale * 2
            yy = (yi / HEIGHT - 0.5) * scale * 2
            z = loss(np.array([xx, yy]))

            if z < MIN: MIN = z
            if z > MAX: MAX = z
            # Normalize z to [0, 1] for colormap
            z = (z - MIN) / (MAX - MIN)

            val = np.clip(z, 0, 1)
            ctx.fillStyle = viridis(val)    
            ctx.fillRect(xi, yi, 10, 10)
    

def draw_sine(ts):
    ctx.clearRect(0, 0, WIDTH, HEIGHT)
    ctx.beginPath()
    for x in range(WIDTH):
        y = HEIGHT / 2 + (HEIGHT / 4) * math.sin((x / WIDTH) * 4 * math.pi + ts / 200)
        if x == 0:
            ctx.moveTo(x, y)
        else:
            ctx.lineTo(x, y)
    ctx.stroke()

def animate(time):
    draw_contour(time)
    update_and_draw_CBO(time)
    #draw_sine(time)
    window.requestAnimationFrame(animate_proxy)

animate_proxy = create_proxy(animate)
window.requestAnimationFrame(animate_proxy)
    
#window.requestAnimationFrame(animate)