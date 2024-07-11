def dst(f,n,L,flag):
#-----------------------------------------------------------------------------------------
#  compute the nth deriv of array f assuming equally spaced
#  sampling with periodicity/even/odd distance equal to L
#  i.e. f(0)=f(L) for periodic functions f
#  flag = 0  ==> f is periodic,  f(0)=f(L),  f(L) not explicitly stored
#  flag = 1  ==> f is expandable in sin, f(-x)=f(x) near x=0,L  f(0)=f(L)=0 explicitly stored
#  flag =-1  ==> f is expandable in cos, f(-x)=-f(x) near x=0,L  f(0)=f(L)  explicitly stored
#-----------------------------------------------------------------------------------------
    import numpy as np
    from scipy.fftpack import fft, ifft, rfft, dct

    N = f.size
    i = 1j                                                   # cmplx(0.,1.)
    if flag == 0:                                            #  Fourier series
     dk = 2.*np.pi/L
     if N%2 == 0:
      k = dk * np.array(list(range(int(N/2)+1)) + list(range(int(-N/2)+1,0,1)))    #  N even, usual case
     else:
      k = dk * np.array(list(range(1,int((N-1)/2)+1,1)) + list(range(int(-(N-1)/2),1,1)))  # N odd (??)
     F = f.astype(float)                       #  was having an endian problem  http://stackoverflow.com/questions/12307429/scipy-fftpack-and-float64
     FHAT = ((i*k)**n) * fft(F)                # have imported scipy's fft,ifft
     df = ifft(FHAT)[0:N].real
    elif flag == 1:
     dk = np.pi/L
     M = (N-1)*2                                             # extended array length
     k = dk * np.array( list(range(int(M/2)+1)) + list(range(int(-M/2)+1,0,1)) )
     F = np.concatenate([f, -f[1:-1][::-1]])                 # F is odd extension of data
     FHAT = ((i*k)**n) * fft(F)                              # have imported scipy's fft,ifft
     df = ifft(FHAT)[0:int(M/2)+1].real 
    elif flag == -1:
     dk = np.pi/L
     M = (N-1)*2                                             # extended array length
     k = dk * np.array( list(range(int(M/2)+1)) + list(range(int(-M/2)+1,0,1)) )
     F = np.concatenate([f,  f[1:-1][::-1]])                 # F is even extension of data
     FHAT = ((i*k)**n) * fft(F)                              # have imported scipy's fft,ifft
     df = ifft(FHAT)[0:int(M/2)+1].real
    else:
     print ("dst problem, called with illegal flag value")
     print (flag)
    del F,FHAT,k
    return df
