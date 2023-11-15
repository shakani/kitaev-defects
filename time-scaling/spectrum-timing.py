from spectrum import *
import time 
import pandas as pd 

if __name__ == "__main__":
    
    # Initialize dataframe 
    df = pd.DataFrame(columns=['L', 'gs_time', 'sf_time'])
    sizes = np.array([4*n + 2 for n in range(1, 4)])
    gs_times = np.zeros(sizes.shape)
    sf_times = np.zeros(sizes.shape)
    
    for i, L in enumerate(sizes):
        mySpectrum = Spectrum(on_the_fly=False, L=int(L))
        start = time.time()
        
        # time ground state
        mySpectrum.compute_ground_state()
        gs_time = time.time() 
        print(f'L = {L}, gs_time = {gs_time - start}')
        
        mySpectrum.compute_spectral_function()
        sf_time = time.time()
        print(f'L = {L}, sf_time = {sf_time - start}')
        # print(f'Ground state time: {gs_time - start} \nSpectral function time: {sf_time - start}')
        gs_times[i], sf_times[i] = gs_time - start, sf_time - start 
        
    df.L, df.gs_time, df.sf_time = sizes, gs_times, sf_times 
    print(df)