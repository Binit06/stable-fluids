# Stan's Stable Fluid Simulation

This is my from scratch implementation of Jos Stam's **Stable Fluids** method, based on the paper included in this repository

I was mainly doing this to understand how the method actually works rather than just following an existing implementation

## Stuff I used

- **Stable Fluids Paper** - this implementation is based on this paper attached in the repo
- [**The Visual Room - Poisson equation for pressure**](https://thevisualroom.com/poisson_for_pressure.html) - helped me understand the pressure and projection part
- **LLM** - used to explain things like gradients and laplacians because I had never really worked with them before

Most of the functions are in `main.py` and have comments explaining what they are doing and which equation from the paper they correspond to

## The simulation

This is what is currently looks like:

<video src="https://raw.githubusercontent.com/Binit06/stable-fluids/main/fluid_simulation.mp4" controls></video>

It's not the prettiest fluid simulation ever made, but it works well enough for me to see what the Stable Fluids method is doing

## Implementation

The main steps are:
- **Advection** -> trace backward through the velocity field and interpolate
- **Diffustion** -> solve for the new velocity field
- **Projection** -> make the velocity field divergence free
- **Extras** -> Added some dye and density to make the simulation look good

This whole thing is in `main.py`

## Performance

It's a bit clanky.

The whole simulation is written in python with a bunch of regular loops over the grid, and I haven't done any NumPy vectorisation or other optimisation

I'm much more comfortable with C++ than NumPy, so if I get some time I'll probably rewrite this in C++ and see how faster it gets

Prbably will do the math of what was slow when I rewrite..

## Running it

**Simple** -> after installing all the required modules (just matplotlib)
```bash
python main.py
```

That's pretty much it

The code is definitely not perfectly structred - I wrote it from scratch while working through the paper in roughly a day, so expect some questionable code choices.

But it works, and more importantly, I understand it.
