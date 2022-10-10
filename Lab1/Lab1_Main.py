# ESE 224 --- Signal and Information Processing
#
# Spring 2021
#
# Lab 1: Discrete sines, cosides and complex exponentials
#
# By Vinicius Lima (adapted from 2020 edition of the course)

###############################################################################
############################# I M P O R T I N G ###############################
###############################################################################

# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
import math
import cmath
import warnings
from scipy.io.wavfile import write
# Local files
from scipy.optimize import least_squares as ls

from cpx_exp import ComplexExp

# Figure aspect ratio / credit: https://jdhao.github.io/2017/06/03/change-aspect-ratio-in-mpl/

###############################################################################
############################ Q U E S T I O N 1.1 ##############################
###############################################################################

def q_11(N, k_list):
    """
    Question 1.1: Generate complex exponentials. This function creates a discrete 
    complex exponential of duration N with discrete frequency k in k_list and plots
    the real and imaginary components of the complex exponential.
    Arguments:
        N: duration of the signal (int)
        k_list: frequency of the discrete complex exponential (list of ints)
    """
    assert isinstance(N, int), "N should be an integer"

    for k in k_list:
        # Creates complex exponential object with frequency k and duration N
        exp_k = ComplexExp(k, N)
        # Real and imaginary parts
        cpx_cos = exp_k.exp_kN_real
        cpx_sin = exp_k.exp_kN_imag
        # Plots real and imaginary parts
        cpx_plt = plt.figure()
        ax = cpx_plt.add_subplot(111)
        plt.stem(exp_k.n, cpx_cos, 'tab:blue', markerfmt='bo', label='Real part')
        plt.stem(exp_k.n, cpx_sin, 'tab:red', markerfmt='ro', label='Imaginary part')
        plt.title('Complex exponential: k = ' + str(k) + ', N = ' + str(N), fontsize=10)
        plt.xlabel('n', fontsize=9)
        plt.ylabel('x[n]', fontsize=9)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend()
        # Aspect ratio credit: https://jdhao.github.io/2017/06/03/change-aspect-ratio-in-mpl/
        ratio = 1/(16/9)
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)

        # Saves plot
        fig_path_name = 'q11_cpxexp_k' + str(k) + '.png'
        plt.savefig(fig_path_name, dpi=300)
        plt.show()


###############################################################################
############################ Q U E S T I O N 1.2 ##############################
###############################################################################

def q_12(N, k_list):
    """
    Question 1.2: Equivalent complex exponentials. 
    Arguments:
        N: duration of the signal (int)
        k_list: frequency of the discrete complex exponential (list of ints)
    """
    assert isinstance(N, int), "N should be an integer"

    for k in k_list:
        # Creates complex exponential object with frequency k and duration N
        exp_k = ComplexExp(k, N)
        # Real and imaginary parts
        cpx_cos = exp_k.exp_kN_real
        cpx_sin = exp_k.exp_kN_imag
        # Plots real and imaginary parts
        cpx_plt = plt.figure()
        ax = cpx_plt.add_subplot(111)
        plt.stem(exp_k.n, cpx_cos, 'tab:blue', markerfmt='bo', label='Real part')
        plt.stem(exp_k.n, cpx_sin, 'tab:red', markerfmt='ro', label='Imaginary part')
        plt.title('Complex exponential: k = ' + str(k) + ', N = ' + str(N), fontsize=10)
        plt.xlabel('n', fontsize=9)
        plt.ylabel('x[n]', fontsize=9)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend()
        # Aspect ratio credit: https://jdhao.github.io/2017/06/03/change-aspect-ratio-in-mpl/
        ratio = 1/(16/9)
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
        # Saves plot
        fig_path_name = 'q12_cpxexp_k' + str(k) + '.png'
        plt.savefig(fig_path_name, dpi=300)
        plt.show()

###############################################################################
############################ Q U E S T I O N 1.3 ##############################
###############################################################################

def q_13(N, k_list):
    """
    Question 1.3: Conjugate complex exponentials. 
    Arguments:
        N: duration of the signal (int)
        k_list: frequency of the discrete complex exponential (list of ints)
    """
    assert isinstance(N, int), "N should be an integer"

    for k in k_list:
        # Creates complex exponential object with frequency k and duration N
        exp_k = ComplexExp(k, N)
        # Real and imaginary parts
        cpx_cos = exp_k.exp_kN_real
        cpx_sin = exp_k.exp_kN_imag
        # Plots real and imaginary parts
        cpx_plt = plt.figure()
        ax = cpx_plt.add_subplot(111)
        plt.stem(exp_k.n, cpx_cos, 'tab:blue', markerfmt='bo', label='Real part')
        plt.stem(exp_k.n, cpx_sin, 'tab:red', markerfmt='ro', label='Imaginary part')
        plt.title('Complex exponential: k = ' + str(k) + ', N = ' + str(N), fontsize=10)
        plt.xlabel('n', fontsize=9)
        plt.ylabel('x[n]', fontsize=9)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.legend()
        # Aspect ratio credit: https://jdhao.github.io/2017/06/03/change-aspect-ratio-in-mpl/
        ratio = 1/(16/9)
        xleft, xright = ax.get_xlim()
        ybottom, ytop = ax.get_ylim()
        ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
        # Saves plot
        fig_path_name = 'q13_cpxexp_k' + str(k) + '.png'
        plt.savefig(fig_path_name, dpi=300)
        plt.show()


###############################################################################
############################ Q U E S T I O N 1.5 ##############################
###############################################################################
def q_15(N):
    """
    Question 1.5: Orthonormality 
    Arguments:
        N: duration of the signal (int)
    """
    assert isinstance(N, int), "N should be an integer"
    k_list = np.arange(N)
    l_list = np.arange(N)

    # Building a matrix with all signals
    cpx_exps = np.zeros((N,N), dtype=np.complex)
    for k in k_list:
        cpxexp = ComplexExp(k, N)
        cpx_exps[:, k] = cpxexp.exp_kN

    # Conjugate
    cpx_exps_conj = np.conjugate(cpx_exps)

    # Option 1: computing inner products simultaneously
    res = np.round(np.matmul(cpx_exps_conj, cpx_exps).real)
    print ("\n Matrix of inner products: Mp")
    ls.print_matrix(res)
    fig, ax = plt.subplots()
    im = ax.imshow(res)
    plt.title('Inner products: N = ' + str(N), fontsize=10)
    plt.xlabel('l = [0, N - 1]', fontsize=9)
    plt.ylabel('k = [0, N - 1]', fontsize=9)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    # Saves plot
    fig_path_name = 'q15_colormap.png'
    plt.savefig(fig_path_name, dpi=300)
    plt.show()
    
    # Option 2: using for loops
    opt2 = np.zeros((N,N), dtype=np.complex)
    for k in k_list:
        for l in l_list:
            r = np.dot(cpx_exps_conj[:, k], cpx_exps[:, l])
            opt2[k, l] = r
            opt2[l, k] = r
    
    res2 = opt2.real 
    print ("\n Matrix of inner products: Mp")
    ls.print_matrix(res2)

    return res


###############################################################################
############################ Q U E S T I O N 3.1 ##############################
###############################################################################

def cexpt(f, T, fs):
    """
    This function generates a (sampled) continuous-time complex exponential.
    Arguments:
        f: frequency of the complex exponential
        T: duration
        fs: sampling frequency
    Returns
        x: vector of samples of a complex exponential of frequency f and duration T
        N: number of samples
    """
    assert T > 0, "Duration of the signal cannot be negative."
    assert fs != 0, "Sampling frequency cannot be zero"

    if fs < 0:
        warnings.warn("Sampling frequency is negative. Using absolute value instead.")
        fs = - fs
    
    if f < 0:
        warnings.warn("Complex exponential frequency is negative. Using absolute value instead.")
        f = -f

    # Duration of the discrete signal
    N = math.floor(T * fs)
    # Discrete frequency
    k = N * f / fs
    # Complex exponential
    cpxexp = ComplexExp(k, N)
    x = cpxexp.exp_kN
    x = np.sqrt(N) * x

    return x, N

def q_31(f, T, fs):
    # Retrieves complex exponential
    cpxexp, num_samples = cexpt(f, T, fs)
    # Cosine is the real part
    cpxcos = cpxexp.real


###############################################################################
############################ Q U E S T I O N 3.2 ##############################
###############################################################################
def q_32(f0, T, fs):
    # Retrieves complex exponential
    cpxexp, num_samples = cexpt(f0, T, fs)
    # Cosine is the real part
    Anote = cpxexp.real
    # Playing the note
    write("Anote.wav", fs, Anote.astype(np.float32))


###############################################################################
############################ Q U E S T I O N 3.3 ##############################
###############################################################################
def q_33(note, T, fs):

    fi = 2**((note - 49) / 12)*440
    # Retrieves complex exponential
    cpxexp, num_samples = cexpt(fi, T, fs)
    # Cosine is the real part
    q33note = cpxexp.real
    # Playing the note
    write("q33note.wav", fs, q33note.astype(np.float32))

# If we pass the note name and note its number
def q_33_notename(note, T, fs):
    # mapping notes
    note_dict = ['C4', 'D4', 'D4s', 'E4', 'F4', 'G4', 'A4', 'A4s', 'B4', 'C5', 'D5', 'D5s', 'E5', 'F5', 'G5',
    'A5', 'A5s', 'B5', 'C6', 'C6s', 'D6', 'E6', 'F6', 'F6s', 'G6', 'A6', 'B6']

    note_fis = [40, 42, 43,
    44, 45, 47, 49,
    50, 51, 52, 54,
    55, 56, 57, 59,
    61, 62, 63, 64,
    65, 66, 68, 69,
    70, 71, 73, 75]

    idx = note_dict.index(note)
    fi = note_fis[idx]

    # Retrieves complex exponential
    cpxexp, num_samples = cexpt(fi, T, fs)
    # Cosine is the real part
    q33note = cpxexp.real
    # Playing the note
    file_name = str(note) + 'note.wav'
    write(file_name, fs, q33note.astype(np.float32))


###############################################################################
############################ Q U E S T I O N 3.4 ##############################
###############################################################################
def q_34(list_notes, list_times, fs):
    assert len(list_notes) == len(list_times), "List of musical notes and musical times should have same length"
    song = []
    for note, note_time in zip(list_notes, list_times):
        fi = 2**((note - 49) / 12)*440
        x, N = cexpt(fi, note_time, fs)
        song = np.append(song, x.real)
        song = np.append(song, np.zeros(10))

    # Writing song
    write("q34_song.wav", fs, song.astype(np.float32))


###############################################################################
################################## M A I N ####################################
###############################################################################

if __name__ == '__main__':

    # Problem 1.1
    list_of_ks = [0, 2, 9, 16]
    duration_of_signal = 32
    q_11(duration_of_signal, list_of_ks)

    # Problem 1.2
    # Frequencies that are N apart
    list_of_ks = [3, -29, 35]
    duration_of_signal = 32
    q_12(duration_of_signal, list_of_ks)

    # Problem 1.3
    # opposite frequencies
    list_of_ks = [-3, 3]
    duration_of_signal = 32
    q_13(duration_of_signal, list_of_ks)

    # Problem 1.4
    # opposite frequencies
    list_of_ks = [3, 29]
    duration_of_signal = 32
    q_13(duration_of_signal, list_of_ks)

    # Problem 1.5
    duration_of_signal = 16
    q15_mtx = q_15(duration_of_signal)

    # Problem 3.2
    f0 = 440
    T = 2
    fs = 44100
    q_32(f0, T, fs)

    # Problem 3.3
    note = 40
    T = 2
    fs = 44100
    q_33(note, T, fs)

    # Problem 3.4
    # mapping notes
    C4=40
    D4=42
    D4s=43
    E4=44
    F4=45
    G4=47
    A4=49
    A4s=50
    B4=51
    C5=52
    D5=54
    D5s=55
    E5=56
    F5=57
    G5=59
    A5=61
    A5s=62
    B5=63
    C6=64
    C6s=65
    D6=66
    E6=68
    F6=69
    F6s=70
    G6=71
    A6=73
    B6=75
    # Song
    song_notes = [D6, G5, A5, B5, C6, D6, G5, G5, E6, C6, D6, E6, F6s, G6, G5, G5, C6, D6, C6, B5, A5, B5, C6, B5, A5, G5, F5, G5, A5, B5, G5, B5, A5,
                D6, G5, A5, B5, C6, D6, G5, G5, E6, C6, D6, E6, F6s, G6, G5, G5,
                C6, D6, C6, B5, A5, B5, C6, B5, A5, G5, A5, B5, A5, G5, F5, G5,
                B6, G6, A6, B6, G6, A6, D6, E6, F6s, D6, G6, E6, F6s, G6, D6, C6s, B5, C6, A5,
                A5, B5, C6s, D6, E6, F6, G6, F6, E6, F6, A5, C6s, D6,
                D6, G5, F5, G5, E6, G5, F5, G5, D6, C6, B5, A5, G6, F5, G5, A5,
                D5, E5, F5, G5, A5, B5, C6, B5, A5, B5, D6, G5, F5, G5]

    rhythm=0.5
    b = 1*rhythm
    w = 2*rhythm
    h = 0.5*rhythm
    dw = 3*rhythm;

    song_times = [b, h, h, h, h, b, b, b, b, h, h, h, h, b, b, b,
                b, h, h, h, h, b, h, h, h, h, b, h, h, h, h, b, w,
                b, h, h, h, h, b, b, b, b, h, h, h, h, b, b, b,
                b, h, h, h, h, b, h, h, h, h, b, h, h, h, h, dw,
                b, h, h, h, h, b, h, h, h, h, b, h, h, h, h, b, h, h, b,
                h, h, h, h, h, h, b, b, b, b, b, b, dw,
                b, h, h, b, b, h, h, b, b, b, b, h, h, h, h, b,
                h, h, h, h, h, h, b, b, b, h, h, b, b, dw]
    q_34(song_notes, song_times, fs)