from js import document, window, Uint8ClampedArray
from pyodide.ffi import create_proxy
import math
import numpy as np
from cbx.objectives import Ackley, Rastrigin, snowflake, Quadratic, Michalewicz, cross_in_tray
from cbx.dynamics import CBO, PolarCBO

# %% Colors
# -----------------------------------------------------------------------------
viridis_data = [
    (68, 1, 84),
    (71, 44, 122),
    (59, 81, 139),
    (44, 113, 142),
    (33, 144, 141),
    (39, 173, 129),
    (92, 200, 99),
    (170, 220, 50),
    (253, 231, 37),
]

def viridis(t):
    n = len(viridis_data)
    idx = t * (n - 1)
    i0 = int(idx)
    i1 = min(i0 + 1, n - 1)
    f = idx - i0

    c0 = viridis_data[i0]
    c1 = viridis_data[i1]

    r = int(c0[0] + f * (c1[0] - c0[0]))
    g = int(c0[1] + f * (c1[1] - c0[1]))
    b = int(c0[2] + f * (c1[2] - c0[2]))

    return r, g, b

#%% Mouse tracker
class LossCenter:
    def __init__(self, canvas):
        self.mouse_pos = {"x": 0, "y": 0}
        self.follow_mouse = True

        def on_mouse_move(event):
            if self.follow_mouse:
                self.mouse_pos["x"] = event.offsetX
                self.mouse_pos["y"] = event.offsetY
        mouse_move_proxy = create_proxy(on_mouse_move)
        canvas.addEventListener("mousemove", mouse_move_proxy)

        def on_click(event):
            if event.button == 0:
                self.follow_mouse = not self.follow_mouse
                if self.follow_mouse:
                    self.mouse_pos["x"] = event.offsetX
                    self.mouse_pos["y"] = event.offsetY

        click_proxy = create_proxy(on_click)
        canvas.addEventListener("click", click_proxy)

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
    def __init__(self, ctx, loss_ctx, loss_canvas, LC,
                 WIDTH = 100, HEIGHT = 100,
                 WIDTH_LOSS = 100, HEIGHT_LOSS = 100, 
                 scale = 10):
        self.ctx      = ctx
        self.loss_ctx = loss_ctx
        self.loss_canvas = loss_canvas
        self.LC = LC
        self.WIDTH, self.HEIGHT    = WIDTH, HEIGHT
        self.WIDTH_LOSS, self.HEIGHT_LOSS    = WIDTH_LOSS, HEIGHT_LOSS
        self.scale    = scale

        self.npts_cont = (int(WIDTH_LOSS), int(HEIGHT_LOSS))
        XX, YY = ((Z) * scale * 2 for Z in np.meshgrid(np.linspace(-1., 1., self.npts_cont[0]), np.linspace(-1., 1., self.npts_cont[1])))
        self.XY = np.stack((XX.T,YY.T)).T


    def draw_cached_contour(self, loss):
        if loss.loss_fun_changed: self.compute_image_loss(loss)
        
        sx = self.LC.mouse_pos['x'] / self.WIDTH  - 0.5
        sy = self.LC.mouse_pos['y'] / self.HEIGHT - 0.5
        XL =      int(     self.WIDTH_LOSS  / 4 - sx * self.WIDTH_LOSS  / 2)
        XR =      int(3 *  self.WIDTH_LOSS  / 4 - sx * self.WIDTH_LOSS  / 2 - XL)
        XU =      int(     self.HEIGHT_LOSS / 4 - sy * self.HEIGHT_LOSS / 2)
        XD =      int(3 *  self.HEIGHT_LOSS / 4 - sy * self.HEIGHT_LOSS / 2 - XU)

        self.ctx.drawImage(
            self.loss_canvas,
            XL, XU, XR, XD, # source rect 
            0, 0, self.WIDTH, self.HEIGHT     # dest rect
        )

    def compute_image_loss(self, loss):
        Z = loss(self.XY, use_offset = False)
        Z = (Z - Z.min())/ (Z.max() - Z.min())
        window.spinning = True
        self.loss_ctx.clearRect(0, 0, self.WIDTH_LOSS, self.HEIGHT_LOSS)
        img_data = loss_ctx.createImageData(self.WIDTH_LOSS, self.HEIGHT_LOSS)
        pixels = Uint8ClampedArray.new(self.npts_cont[0] * self.npts_cont[1] * 4)
        
        for i in range(self.npts_cont[0]):
            for j in range(self.npts_cont[1]):
                z = np.clip(Z[j, i], 0, 1)
                r, g, b = viridis(z)
                idx = 4 * (j * self.npts_cont[0] + i)
                pixels.set([r, g, b, 255], idx)
        
        img_data.data.set(pixels)
        self.loss_ctx.putImageData(img_data, 0, 0)
        window.spinning = False

    def draw_contour(self, loss):
        # Background
        self.ctx.fillStyle = "white"
        self.ctx.fillRect(0, 0, self.WIDTH, self.HEIGHT)
        self.draw_cached_contour(loss)
        loss.loss_fun_changed = False

    
    def draw_particles(self, DYN):
        CS = DYN.consensus.shape[1]
        for i in range(DYN.N): # Draw particles
            draw_line(DYN.x[0, i, :], DYN.consensus[0, min(i, CS - 1), :], 
                      self.ctx, 
                      self.WIDTH, self.HEIGHT, self.scale, 
                      color="rgb(170,0,0, 0.3)")
            draw_particle(DYN.x[0, i, :], self.ctx, 
                          self.WIDTH, self.HEIGHT, self.scale, color="rgb(230,0,0)")
        for i in range(CS):
            draw_particle(DYN.consensus[0, i, :], self.ctx, 
                        self.WIDTH, self.HEIGHT, self.scale, 
                        color="rgb(200,200,200)", r=3)  # Draw consensus point
#%% Set up Canvas
canvas = document.getElementById("canvas")
ctx = canvas.getContext("2d")

loss_canvas = document.getElementById("loss")
loss_ctx = loss_canvas.getContext("2d")

window.spinning = False
ctx.clearRect(0, 0, canvas.width, canvas.height)
WIDTH, HEIGHT = canvas.width, canvas.height
WIDTH_LOSS, HEIGHT_LOSS = loss_canvas.width, loss_canvas.height


# %% App Parameters
# -----------------------------------------------------------------------------
def add_slider(slider_id: str, label: str, 
               help_text:str = '',
               min_val=0.0, max_val=1.0, step=0.01, initial=0.5):
    container = document.getElementById("cbo-slider-container")

    # Create wrapper div
    wrapper = document.createElement("div")
    wrapper.id = f"{slider_id}-wrapper"

    # Input slider
    slider = document.createElement("sl-range")
    slider.type = "sl-range"
    slider.id = slider_id
    slider.label = label
    slider.min = min_val
    slider.max = max_val
    slider.step = step
    slider.value = initial
    slider.setAttribute("help-text", help_text)

    
    wrapper.id = f"{slider_id}-wrapper"
    wrapper.appendChild(slider)
    container.appendChild(wrapper)

def remove_slider(slider_id):
    wrapper = document.getElementById(f"{slider_id}-wrapper")
    if wrapper:
        wrapper.remove()


class AppParams:
    def create_slider(self, name, val, slider_name):
        setattr(self, name, val)

        def local_on_slider_change(event):
            setattr(self, name, event.target.value)

        slider = document.getElementById(slider_name)
        slider.addEventListener("input", create_proxy(local_on_slider_change))
    
    def create_button_group(self, name, val, group_name):
        setattr(self, name, val)

        def local_on_button_change(event):
            setattr(self, name, event.target.value)

        group = document.getElementById(group_name)
        group.addEventListener("sl-change", create_proxy(local_on_button_change))

AP = AppParams()

SliderParams = [
    ("cbo_sigma", 1.5, "sigma-slider"), 
    ("cbo_alpha", 10,  "alpha-slider"),
    ("cbo_dt",   0.1,  "dt-slider"),
    ("fps",       10,  "fps-slider")
]

for name, val, slider_name in SliderParams: AP.create_slider(name, val, slider_name)
AP.create_button_group('loss_option', 0, 'lossfun')
AP.create_button_group('noise_option', 'Isotropic', 'noise')
AP.create_button_group('cbo_option', 'CBO', 'Opt')

# %% Loss function
# -----------------------------------------------------------------------------
class AppLoss:
    def __init__(self, LC):
        self.set_loss(0)
        self.LC = LC
        self.loss_fun_changed = True

    def set_loss(self, id):
        id = int(id)
        self.id = id
        if id == 0:
            self.f = Ackley()
        elif id == 1:
            self.f = Rastrigin(A=30.)
        elif id == 2:
            self.f = snowflake(alpha = 1/3)
        elif id == 3:
            self.f = Quadratic(alpha=0.2)
        elif id == 4:
            self.f = cross_in_tray()
        elif id == 5:
            self.f = Michalewicz()
    
    def __call__(self, x, use_offset = True):
        self.mouse_mu = 2  *scale * np.array(
            [(self.LC.mouse_pos["x"] / WIDTH - 0.5), 
            (self.LC.mouse_pos["y"] / HEIGHT - 0.5)]
        )
        if use_offset:
            x = x - self.mouse_mu[None,None, ...]
        return self.f(x)
    
    def update_from_options(self, AP):
        id = int(getattr(AP, 'loss_option', 0))
        if not self.id == id:
            self.set_loss(id)
            self.loss_fun_changed = True


# %% Consensus-based optimization setup
# -----------------------------------------------------------------------------
def post_process_cbx(dyn):
    dyn.x = np.clip(dyn.x, -1.5* scale, 1.5*scale)

def set_CBO_params(DYN, AP, loss):
    loss.update_from_options(AP)
    DYN.sigma = getattr(AP, 'cbo_sigma', 1.)
    DYN.alpha[:] = getattr(AP, 'cbo_alpha', 10.)
    DYN.dt = getattr(AP, 'cbo_dt', 1.)

    noise = getattr(AP, 'noise_option', 'anisotropic')
    if getattr(DYN, 'AP_noise', 'isotropic') != noise:
        DYN.AP_noise = noise
        DYN.set_noise(noise.lower())

    if AP.cbo_option == 'PolarCBO':
        DYN.kappa = getattr(AP, 'cbo_kappa', 1.)
    
    DYN.f = loss

def update_CBO(DYN, AP, loss):
    set_CBO_params(DYN, AP, loss)  # Update CBO parameters
     # Update loss function
    DYN.step()


#%% Set up classes
scale = 3.0
LC = LossCenter(canvas)
loss = AppLoss(LC)
CBOA = CBO_artist(
    ctx, loss_ctx, loss_canvas, LC,
    WIDTH = WIDTH, HEIGHT = HEIGHT, 
    WIDTH_LOSS = WIDTH_LOSS, HEIGHT_LOSS =  HEIGHT_LOSS,
    scale = scale
)
x = np.random.uniform(-scale, scale, (1, 20, 2))  # 20 particles in 2D
DYN = CBO(
    loss,
    x=x,
    post_process=post_process_cbx,
    track_args = {'names':[]},
)
set_CBO_params(DYN, AP, loss)

#%% Frame Rate handling
last_time = 0
max_frame_rate = 10
cbo_option = 'CBO'

def animate(time):
    global last_time, CBOA, DYN, AP, loss, cbo_option

    dt = time - last_time
    fps = 1000 / dt if dt > 0 else 0
    #print(f"FPS: {fps:.1f}")

    if fps < getattr(AP, 'fps', 10):
        update_CBO(DYN, AP, loss)
        CBOA.draw_contour(loss)
        CBOA.draw_particles(DYN) 
        last_time = time
    window.requestAnimationFrame(animate_proxy)

    if cbo_option != AP.cbo_option:
        args = [loss]
        kwargs = {'x': np.copy(x), 'post_process':post_process_cbx, 'track_args': {'names':[]},}
        if AP.cbo_option == 'PolarCBO':
            DYN = PolarCBO(*args, kappa=1., kernel_factor_mode='const', kernel = 'Gaussian', **kwargs)
            add_slider('cbo_kappa', 
                       label = 'Kappa for PolarCBO',
                       help_text = 'Determines the interaction strength:\n large -> particles have a global view, small -> particles localize',
                       min_val=0.0, max_val=10, step=0.1, initial=1)
            AP.create_slider('cbo_kappa', 10., 'cbo_kappa')
        else:
            DYN = CBO(*args, **kwargs)
            remove_slider('cbo_kappa')
        
        cbo_option = AP.cbo_option
    

animate_proxy = create_proxy(animate)
window.requestAnimationFrame(animate_proxy)
    
#window.requestAnimationFrame(animate)