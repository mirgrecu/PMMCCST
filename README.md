## A Machine-Learning Framework to Enhance, Adapt, and Extend the GPM Combined Radar-Radiometer Algorithm (CORRA)

PI and Co-I

### 1 Scientific/Technical Management
### 1.1 Executive Summary

The accurate quantification of uncertainties is crucial in the derivation of optimal precipitation estimates from multiple sources of information.  This need has always existed but is now growing more urgent as the number and diversity of data sources increase rapidly.  Specifically, the recent development and deployment of miniaturized, affordable sensors such as the NASA Time-Resolved Observations of Precipitation structure and storm Intensity with a Constellation of Smallsats (TROPICS), combined with the operational use of deep-learning-based numerical weather prediction (e.g. FourCastNet, GraphCast, etc.), in addition traditional algorithms to estimate precipitation from satellite instruments operated by various agencies (e.g. NASA, NOAA, etc.) have created a need for robust and accurate methods to combine these sources of information. 

The NASA GPM Combined Radar-Radiometer Algorithm (CORRA) is a state-of-the-art algorithm that combines information from two sources of information, i.e. the GPM Dual-frequency Precipitation Radar (DPR) and the GPM Microwave Imager (GMI), to estimate precipitation.  As such, the algorithm has a modular structure with rigorous procedures to quantify the uncertainties associated with each source of information [1]. While CORRA has continuously improved with each release, some limitations remain unaddressed [2]. For example, the snowfall estimates are biased low with respect to the ground-based radar estimates provided by the Multi Radar Multi Sensor (MRMS), and the algorithm misses a potential fraction of light precipitation (especially at high-latitudes) due to limited radar-sensitivity.  

We propose the development of two machine-learning (ML) modules that would not only mitigate some of CORRA's existing limitations, but also enable its application to related tasks, such as the estimation of precipitation from other combinations of instruments and data assimilation. The first module will consist of a Mixture Density Network (MDN) [4] trained to estimate the precipitation rate and along with other PSD parameters and their uncertainties from GPM DPR-only observations. The radar MDN model will be trained using an updated version of CORRA featuring a more accurate snowfall retrieval derived using reflectivity snowfall rates using microphysical observations from field campaigns. A major benefit of the proposed radar MDN model is that it will provide precipitation estimates at a fraction of the cost of the current CORRA ensemble radar-profiling algorithm and will enable better investigation and mitigation of uncertainties, as the estimates are differentiable functions of the input. Similarly, the second module will be also based on an MDN formulation and will provide refined precipitation and associated uncertainties estimates. The radiometer MDN training dataset will be derived by sampling the radar MDN output and the CORRA's current surface emissivity estimates and simulating brightness temperatures using the radiative transfer model. The radiometer MDN will be trained to estimate the precipitation rate and associated uncertainties from the brightness temperatures and the radar MDN output.  To appropriately represent light precipitation profiles in the training dataset, we will use a biased sampling strategy as well as the CloudSat-GPM coincidence dataset [3]. The sampling strategy used to create the training dataset is expected to enable a better exploration of the space of possible precipitation estimates and uncertainties than currently possible in the operational CORRA and provide more accurate estimates of the precipitation rate and associated uncertainties.

A schematic illustration of the proposed research is presented in Figure 1. 

![Corra Flowchart](CORRA_FlowChart.png)

### 1.2 Technical Approach
#### 1.2.1 Radar Mixtures Density Network  
[[radar]]
The GPM CORRA algorithm is based on an ensemble filter estimation methodology[1]. More specifically, a radar profiling algorithm [5] is used to generate an ensemble of potential precipitation profiles that are consistent with the Ku-band radar observations. The initial ensemble is used to simulate Ka-band radar and GMI observations, which are then used to update the ensemble. Independent estimates of the path integrated attenuation (PIA) from the surface reference technique (SRT) are also used in the update. The radar algorithm is run twice in the current version of CORRA to enable the inclusion of climatological relationships between the particle size distribution (PSD) intercept ($N_w$) and the mass weighted mean diameter ($D_m$) in the radar solution [6]. Although various techniques (e.g. the use of scattering lookup tables and a fast iterative attenuation correction technique [5,6]) have been used, the radar algorithm remains computationally expensive. This makes further tuning of the algorithm, the exploration of alternative solutions, and adaptation to other scenarios (such as the application to the NASA Atmosphere Observing System (AOS) mission) difficult. 



#### 1.2.2 Radiometer Mixtures Density Network 
[[radiometer]]

### 1.3 Impact

### 1.4 Relevance

### 1.5 Work Plan

### 1.6 Schedule

### 2. References

[1] Grecu, M., Olson, W.S., Munchak, S.J., Ringerud, S., Liao, L., Haddad, Z., Kelley, B.L. and McLaughlin, S.F., 2016. The GPM combined algorithm. Journal of Atmospheric and Oceanic Technology, 33(10), pp.2225-2245.

[2] Olson, W. S. and coauthors, 2024, PMM Combined Radar-Radiometer Algorithm, PMM Science Team Meeting, San-Diego, 9-13 September 2024, available https://docs.google.com/presentation/d/e/2PACX-1vSYTmdkLlcvAKa2lVka050RDLewQ-IQHa3mRU4-6Yqu3tM3TLAF_8IqtvnJCl6-JDs8XlevwtUaYsIo/pub?start=false&loop=false&delayms=3000

[3] Turk, F. Joseph, Sarah E. Ringerud, Andrea Camplani, Daniele Casella, Randy J. Chase, Ardeshir Ebtehaj, Jie Gong, Mark Kulie, Guosheng Liu, Lisa Milani, and et al. 2021. "Applications of a CloudSat-TRMM and CloudSat-GPM Satellite Coincidence Dataset" Remote Sensing 13, no. 12: 2264. https://doi.org/10.3390/rs13122264

[4] Bishop, C.M. and Bishop, H., 2023. Deep learning: Foundations and concepts. Springer Nature.

[5] Grecu, M., Tian, L., Olson, W.S. and Tanelli, S., 2011. A robust dual-frequency radar profiling algorithm. Journal of applied meteorology and climatology, 50(7), pp.1543-1557.

[6] Olson, W.S., Masunaga, H. and GPM CORRA Team., 2022. GPM combined radar-radiometer precipitation algorithm theoretical basis document (version 7). NASA: Washington, DC, USA, available at https://gpm.nasa.gov/sites/default/files/2023-01/Combined_algorithm_ATBD.V07_0.pdf.

### 3 Open Science and Data Management Plan
### 4 Biographical Sketches
### 5 Summary of Personnel and Work Effort
### 6 Current and Pending Support
### 7 Budget and Budget Justification
#### 7.1 Budget Justification Morgan State University
#### 7.2 Detailed Budget
#### 7.3 NASA Budget Justification: Narrative and Details
### 8 Special Notifications and/or Certifications