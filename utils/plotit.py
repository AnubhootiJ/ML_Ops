import matplotlib.pyplot as plt

def plot(ax, vaccs, taccs, gammas, split, res):
    #plt.figure()
    nran = [i+1 for i in range(6)]
    ax.plot(nran, taccs, 'b', label = 'Train Accuracy')
    ax.plot(nran, vaccs, 'g', label='Val Accuracy', )
    ax.set_title(
        "Gamma Vs. Accuracy with split {} and resolution {}".format(split, res),
        fontsize=8)
    ax.set_xlabel("Gamma values")
    ax.set_ylabel("Accuracy")
    plt.xticks([i+1 for i in range(6)], gammas)
    plt.legend()
    #ax.set_xticks(gammas)
   