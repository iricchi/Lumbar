#!/usr/bin/python

""" This script contains useful function used in sFC analysis for plotting especially.

Author: Ilaria Ricchi
"""
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


def plot_vol_scatter(vol, seeds_array, outpath=''):
    nb_frames = vol.shape[-1]
    r = vol.shape[1]
    c = vol.shape[0]
    volume = vol.T
    seeds = seeds_array.T
    
    x = np.nonzero(seeds_array)[0]
    y = np.nonzero(seeds_array)[1]
    z = np.nonzero(seeds_array)[2]

    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        z=(nb_frames - k) * np.ones((r, c)),
        surfacecolor=np.flipud(volume[nb_frames-1 - k]),
        cmin=0, cmax=1000
        ),
        name=str(k) # you need to name the frame for the animation to behave properly
        )
        for k in range(nb_frames)])
   
    
    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z=nb_frames* np.ones((r, c)),
        surfacecolor=np.flipud(volume[nb_frames-1]),
        colorscale='Gray',
        cmin=0, cmax=1000,
        colorbar=dict(thickness=20, ticklen=4),
        contours=go.surface.Contours(
        x=go.surface.contours.X(highlight=False),
        y=go.surface.contours.Y(highlight=False),
        z=go.surface.contours.Z(highlight=False),
        )
        ))

    fig.update_traces(hovertemplate=None)
    # add scatter 3d of the seeds
    fig.add_scatter3d(x=x,y=y,z=np.flipud(z),mode='markers', opacity=0.7)
    
    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(nb_frames-k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]
    
    # Layout
    fig.update_layout(
             title="Scroll to select a slice: z",
             width=600,
             height=600,        
             scene=dict(zaxis=dict(range=[0, nb_frames], autorange=False),
                        aspectratio=dict(x=2, y=1, z=3),
                        ),
             updatemenus = [
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;", # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;", # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
             ],
             sliders=sliders
    )
 
    fig.update_layout(go.Layout(
    scene=go.layout.Scene(
        xaxis = go.layout.scene.XAxis(showspikes=False),
        yaxis = go.layout.scene.YAxis(showspikes=False),
        zaxis = go.layout.scene.ZAxis(showspikes=False),
        )
    ))

    #     scene=go.layout.Scene(
    #         xaxis = go.layout.scene.XAxis(showspikes=False),
    #         yaxis = go.layout.scene.YAxis(showspikes=False),
    #         zaxis = go.layout.scene.ZAxis(showspikes=False),
    #     )



    fig.show()
    
    if len(outpath) != 0:
        # save interactive img
        fig.write_html(os.path.join(outpath,"slices_seeds.html"))
        
