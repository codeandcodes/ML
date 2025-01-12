import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import argparse
from threading import Thread

def init_grid(N, live_prob):
    """Initialize the grid with random values."""
    return np.random.choice([1, 0], N*N, p=[live_prob, 1-live_prob]).reshape(N, N)

def update(frameNum, img, grid, N):
    """Update the grid state and the image."""
    newGrid = grid.copy()
    for i in range(N):
        for j in range(N):
            total = (grid[i, (j-1)%N] + grid[i, (j+1)%N] +
                     grid[(i-1)%N, j] + grid[(i+1)%N, j] +
                     grid[(i-1)%N, (j-1)%N] + grid[(i-1)%N, (j+1)%N] +
                     grid[(i+1)%N, (j-1)%N] + grid[(i+1)%N, (j+1)%N])
            if grid[i, j] == 1:
                if (total < 2) or (total > 3):
                    newGrid[i, j] = 0
            else:
                if total == 3:
                    newGrid[i, j] = 1
    img.set_data(newGrid)
    grid[:] = newGrid[:]
    return img,

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description="Run Conway's Game of Life")
    parser.add_argument('--grid-size', dest='N', type=int, default=100,
                        help='Grid size for NxN grid (default: 100x100)')
    parser.add_argument('--interval', dest='interval', type=int, default=50,
                        help='Update interval in milliseconds (default: 50)')
    parser.add_argument('--live-probability', dest='live_prob', type=float, default=0.2,
                        help='Probability of a cell being alive at start (default: 0.2)')
    args = parser.parse_args()

    N = args.N
    live_prob = args.live_prob
    updateInterval = args.interval

    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.1, bottom=0.25)
    grid = init_grid(N, live_prob)
    img = ax.imshow(grid, interpolation='nearest', cmap='gray')

    # Create the animation update
    ani = animation.FuncAnimation(fig, update, fargs=(img, grid, N), interval=updateInterval, cache_frame_data=False)

    # Slider Axes
    axcolor = 'lightgoldenrodyellow'
    axn = plt.axes([0.1, 0.2, 0.65, 0.03], facecolor=axcolor)
    axinterval = plt.axes([0.1, 0.1, 0.65, 0.03], facecolor=axcolor)
    axprob = plt.axes([0.1, 0.15, 0.65, 0.03], facecolor=axcolor)

    # Sliders
    s_N = Slider(axn, 'Size', 100, 500, valinit=N, valfmt='%0.0f')
    s_interval = Slider(axinterval, 'Interval', 10, 200, valinit=updateInterval)
    s_prob = Slider(axprob, 'Live Probability', 0.01, 0.9, valinit=live_prob)

    # Slider update functions
    def update_vals(val):
        # Update the animation interval effectively
        ani.event_source.stop()
        ani.event_source.interval = s_interval.val
        ani.event_source.start()

        # Reinitialize the grid with new probability and update the image
        global grid
        N = int(s_N.val)
        grid = init_grid(N, s_prob.val)
        fig.set_size_inches(2 * N / 100, 2 * N / 100)
        img.set_data(grid)
        img.figure.canvas.draw_idle()  # Force a redraw of the canvas to update the display immediately

    s_N.on_changed(update_vals)
    s_interval.on_changed(update_vals)
    s_prob.on_changed(update_vals)

    plt.show()

if __name__ == '__main__':
    main()