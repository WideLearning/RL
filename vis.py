import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_3d_points(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Extract x, y, and z coordinates from points
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    z_coords = [point[2] for point in points]

    ax.scatter(x_coords, y_coords, z_coords)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    plt.show()


def read_points():
    points = []
    with open("coordinates.txt") as f:
        for line in f.readlines():
            points.append(list(map(float, line.split())))
    return points


points = read_points()
print(points)
plot_3d_points(points)
