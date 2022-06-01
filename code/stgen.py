from neo import SpikeTrain
import numpy as np
class StGen:
    def __init__(self, rng=None, seed=None):
        """ 
        Stochastic Process Generator
        ============================

        Object to generate stochastic processes of various kinds
        and return them as SpikeTrain or AnalogSignal objects.
      

        Inputs:
        -------
            rng - The random number generator state object (optional). Can be None, or 
                  a numpy.random.RandomState object, or an object with the same 
                  interface.

            seed - A seed for the rng (optional).

        If rng is not None, the provided rng will be used to generate random numbers, 
        otherwise StGen will create its own random number generator.
        If a seed is provided, it is passed to rng.seed(seed)

        Examples
        --------
            >> x = StGen()



        StGen Methods:
        ==============

        Spiking point processes:
        ------------------------
 
        poisson_generator - homogeneous Poisson process
        inh_poisson_generator - inhomogeneous Poisson process (time varying rate)
        inh_gamma_generator - inhomogeneous Gamma process (time varying a,b)
        inh_adaptingmarkov_generator - inhomogeneous adapting markov process (time varying)
        inh_2Dadaptingmarkov_generator - inhomogeneous adapting and 
                                         refractory markov process (time varying)

        Continuous time processes:
        --------------------------

        OU_generator - Ohrnstein-Uhlenbeck process

        See also:
        --------
          shotnoise_fromspikes

        """

        if rng==None:
            self.rng = np.random.RandomState()
        else:
            self.rng = rng

        if seed != None:
            self.rng.seed(seed)
        self.dep_checked = False

    def seed(self,seed):
        #"seed the gsl rng with a given seed"
        self.rng.seed(seed)


    def poisson_generator(self, rate, t_start=0.0, t_stop=1000.0, array=False,debug=False):
        
#         Returns a SpikeTrain whose spikes are a realization of a Poisson process
#         with the given rate (Hz) and stopping time t_stop (milliseconds).

#         Note: t_start is always 0.0, thus all realizations are as if 
#         they spiked at t=0.0, though this spike is not included in the SpikeList.

#         Inputs:
#         -------
#             rate    - the rate of the discharge (in Hz)
#             t_start - the beginning of the SpikeTrain (in ms)
#             t_stop  - the end of the SpikeTrain (in ms)
#             array   - if True, a numpy array of sorted spikes is returned,
#                       rather than a SpikeTrain object.

#         Examples:
#         --------
#             >> gen.poisson_generator(50, 0, 1000)
#             >> gen.poisson_generator(20, 5000, 10000, array=True)

#         See also:
#         --------
#             inh_poisson_generator, inh_gamma_generator, inh_adaptingmarkov_generator
        

        number = int((t_stop-t_start)/1000.0*2.0*rate)

        # less wasteful than double length method above
        n = (t_stop-t_start)/1000.0*rate
        number = int(np.ceil(n+3*np.sqrt(n)))
        if number<100:
            number = int(min(5+np.ceil(2*n),100))

        if number > 0:
            isi = self.rng.exponential(1.0/rate, number)*1000.0
            if number > 1:
                spikes = np.add.accumulate(isi)
            else:
                spikes = isi
        else:
            spikes = np.array([])

        spikes+=t_start
        i = np.searchsorted(spikes, t_stop)

        extra_spikes = []
        if i==len(spikes):
            # ISI buf overrun
            
            t_last = spikes[-1] + self.rng.exponential(1.0/rate, 1)[0]*1000.0

            while (t_last<t_stop):
                extra_spikes.append(t_last)
                t_last += self.rng.exponential(1.0/rate, 1)[0]*1000.0
            
            spikes = np.concatenate((spikes,extra_spikes))

            if debug:
                print("ISI buf overrun handled. len(spikes)=%d, len(extra_spikes)=%d"%(len(spikes),len(extra_spikes)))


        else:
            spikes = np.resize(spikes,(i,))

        if not array:
            spikes = SpikeTrain(spikes, t_start=t_start,t_stop=t_stop)


        if debug:
            return spikes, extra_spikes
        else:
            return spikes
        
    def inh_poisson_generator(self, rate, t, t_stop, array=False):
        """
        Returns a SpikeTrain whose spikes are a realization of an inhomogeneous 
        poisson process (dynamic rate). The implementation uses the thinning 
        method, as presented in the references.

        Inputs:
        -------
            rate   - an array of the rates (Hz) where rate[i] is active on interval 
                     [t[i],t[i+1]]
            t      - an array specifying the time bins (in milliseconds) at which to 
                     specify the rate
            t_stop - length of time to simulate process (in ms)
            array  - if True, a numpy array of sorted spikes is returned,
                     rather than a SpikeList object.

        Note:
        -----
            t_start=t[0]

        References:
        -----------

        Eilif Muller, Lars Buesing, Johannes Schemmel, and Karlheinz Meier 
        Spike-Frequency Adapting Neural Ensembles: Beyond Mean Adaptation and Renewal Theories
        Neural Comput. 2007 19: 2958-3010.

        Devroye, L. (1986). Non-uniform random variate generation. New York: Springer-Verlag.

        Examples:
        --------
            >> time = arange(0,1000)
            >> stgen.inh_poisson_generator(time,sin(time), 1000)

        See also:
        --------
            poisson_generator, inh_gamma_generator, inh_adaptingmarkov_generator
        """

        if np.shape(t)!=np.shape(rate):
            raise ValueError('shape mismatch: t,rate must be of the same shape')

        # get max rate and generate poisson process to be thinned
        rmax = np.max(rate)
        ps = self.poisson_generator(rmax, t_start=t[0], t_stop=t_stop, array=True)

        # return empty if no spikes
        if len(ps) == 0:
            if array:
                return np.array([])
            else:
                return SpikeTrain(np.array([]), t_start=t[0],t_stop=t_stop)
        
        # gen uniform rand on 0,1 for each spike
        rn = np.array(self.rng.uniform(0, 1, len(ps)))

        # instantaneous rate for each spike
        
        idx=np.searchsorted(t,ps)-1
        spike_rate = rate[idx]

        # thin and return spikes
        spike_train = ps[rn<spike_rate/rmax]

        if array:
            return spike_train

        return SpikeTrain(spike_train, t_start=t[0],t_stop=t_stop)
