from spectrum import *
import time 

if __name__ == "__main__":
    mySpectrum = Spectrum(on_the_fly=False, L=12)
    start = time.time()
    
    # time ground state
    mySpectrum.compute_ground_state()
    gs_time = time.time() 
    
    mySpectrum.compute_spectral_function()
    sf_time = time.time()
    print(f'Ground state time: {gs_time - start} \nSpectral function time: {sf_time - start}')