from spectrum import *
import time 
import pandas as pd 

if __name__ == "__main__":
    
    # Initialize dataframe 
    df = pd.DataFrame(columns=['L', 'gs_time', 'sf_time'])
    sizes = np.array([8 + 2*n for n in range(5) if (8 + 2*n - 2)%4 != 0])
    gs_times = np.zeros(sizes.shape)
    sf_times = np.zeros(sizes.shape)
    
    for i, L in enumerate(sizes):
        print(f'Computing L = {L}')
        mySpectrum = Spectrum(on_the_fly=False, L=int(L), use_symm=False)
        start = time.time()
        
        # time ground state
        mySpectrum.compute_ground_state()
        gs_time = time.time() 
        
        # mySpectrum.compute_spectral_function()
        # sf_time = time.time()
        
        # gs_times[i], sf_times[i] = gs_time - start, sf_time - start 
        gs_times[i] = gs_time - start 
        
    df.L, df.gs_time, df.sf_time = sizes, gs_times, sf_times 
    df.L, df.gs_time = sizes, gs_times
    df.to_csv('spectrum-times.csv')