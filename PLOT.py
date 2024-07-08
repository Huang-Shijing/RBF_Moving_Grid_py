import matplotlib.pyplot as plt

def plot_aft_stack(aft_stack, x_coord, y_coord, nose_x):
    plt.clf()
    for i in range(len(aft_stack)):
        node1 = int(aft_stack[i, 0])
        node2 = int(aft_stack[i, 1])
        
        xx = [x_coord[node1 - 1], x_coord[node2 - 1]]
        yy = [y_coord[node1 - 1], y_coord[node2 - 1]]
        if aft_stack[i, 6] == 3:
            plt.plot(xx, yy, '-k', linewidth=1.5)
        else:
            plt.plot(xx, yy, '-r' )

    plt.axis('equal')
    plt.axis([nose_x - 0.5, nose_x + 1.5, -0.7, 0.7])
    plt.pause(0.001)