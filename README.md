## A Machine-Learning Framework to Enhance, Adapt, and Extend the GPM Combined Radar-Radiometer Algorithm (CORRA)

PI and Co-I

### 1 Scientific/Technical Management
### 1.1 Executive Summary

The accurate quantification of uncertainties is crucial in the derivation of optimal precipitation estimates from multiple sources of information.  This need has always existed but is now growing more urgent as the number and diversity of data sources increase rapidly.  Specifically, the recent development and deployment of miniaturized, affordable sensors such as the NASA Time-Resolved Observations of Precipitation structure and storm Intensity with a Constellation of Smallsats (TROPICS), combined with the operational use of deep-learning-based numerical weather prediction (e.g. FourCastNet, GraphCast, etc.), in addition traditional algorithms to estimate precipitation from satellite instruments operated by various agencies (e.g. NASA, NOAA, etc.) have created a need for robust and accurate methods to combine these sources of information. 

The NASA GPM Combined Radar-Radiometer Algorithm (CORRA) is a state-of-the-art algorithm that combines information from two sources of information, i.e. the GPM Dual-frequency Precipitation Radar (DPR) and the GPM Microwave Imager (GMI), to estimate precipitation.  As such, the algorithm has a modular structure with rigorous procedures to quantify the uncertainties associated with each source of information [1]. While CORRA has continuously improved with each release, some limitations remain unaddressed [2]. For example, the snowfall estimates are biased low with respect to the ground-based radar estimates provided by the Multi Radar Multi Sensor (MRMS), and the algorithm misses a potential fraction of light precipitation (especially at high-latitudes) due to limited radar-sensitivity.  

We propose the development of two machine-learning (ML) modules that would not only mitigate some of CORRA's existing limitations, but also enable its application to related tasks, such as the estimation of precipitation from other combinations of instruments and data assimilation. The first module will consist of a Cluster-wised Rigde Regresssion (CRR) [4] trained to estimate the precipitation rate and along with other PSD parameters and their uncertainties from GPM DPR-only observations. The radar CRR model will be trained using a customized version of CORRA featuring a more accurate snowfall retrieval derived using reflectivity snowfall rates using microphysical observations from field campaigns. A major benefit of the proposed radar CRR model is that it will provide precipitation estimates at a fraction of the cost of the current CORRA ensemble radar-profiling algorithm and will enable better investigation and mitigation of uncertainties, as the estimates are differentiable functions of the input. Similarly, the second module will be also based on an CRR formulation and will provide refined precipitation and associated uncertainties estimates. The radiometer CRR training dataset will be derived by sampling the radar CRR output and the CORRA's current surface emissivity estimates and simulating brightness temperatures using the radiative transfer model. The radiometer CRR will be trained to estimate the precipitation rate and associated uncertainties from the brightness temperatures and the radar CRR output.  To appropriately represent light precipitation profiles in the training dataset, we will use a biased sampling strategy as well as the CloudSat-GPM coincidence dataset [3]. The sampling strategy used to create the training dataset is expected to enable a better exploration of the space of possible precipitation estimates and uncertainties than currently possible in the operational CORRA and provide more accurate estimates of the precipitation rate and associated uncertainties.

A schematic illustration of the proposed research is presented in Figure 1. 

![Corra Flowchart](CORRA_FlowChart.png)

### 1.2 Technical Approach

#### 1.2.1 Radar Cluster-wise Ridge Regression  

The GPM CORRA algorithm is based on an ensemble filter estimation methodology[1]. More specifically, a radar profiling algorithm [5] is used to generate an ensemble of potential precipitation profiles that are consistent with the Ku-band radar observations. The initial ensemble is used to simulate Ka-band radar and GMI observations, which are then used to update the ensemble. Independent estimates of the path integrated attenuation (PIA) from the surface reference technique (SRT) are also used in the update. The radar algorithm is run twice in the current version of CORRA to enable the inclusion of climatological relationships between the particle size distribution (PSD) intercept ($N_w$) and the mass weighted mean diameter ($D_m$) in the radar solution [6]. Although various techniques (e.g. the use of scattering lookup tables and a fast iterative attenuation correction technique [5,6]) have been used, the radar algorithm remains computationally expensive. This makes further tuning of the algorithm, the exploration of alternative solutions, and adaptation to other scenarios (such as the application to the NASA Atmosphere Observing System (AOS) mission) difficult. 

To mitigate this limitation, we propose the implementation and use of a simple yet highly effective technique called cluster-wise linear regression [4]. The technique involves the clustering of predictors into a set of clusters, followed by the estimation of a linear regression model for each cluster. The technique has been shown to be effective in a variety of applications, including the estimation of cloud ice from satellite observations [7]. To ensure its robustness, we will use a particular type of regression called ridge regression, which involves the addition of a penalty term to the least squares objective function. The penalty term is proportional to the square of the magnitude of the coefficients, which helps to prevent overfitting. We will therefore refer to the proposed technique as cluster-wise ridge regression (CRR). It should be mentioned that given its conceptual simplicity, the CRR approach has been formulated and applied independently in multiple areas of research, which resulted in additional names (such as conditional regression, cluster regression, piece-wise linear regression, etc.) attributed to practically the same approach. It is also worth mentioning that the CRR approach is related Gaussian Mixture Models [8], but with a crisp rather than gradual transition between clusters.  Similarly, CRRs may be seen as a special case of neural networks with each cluster corresponding to a unique activation path in a neural network. However, unlike simple feed-forward neural networks, CRRs can provide uncertainty estimates. Certain types of neural networks, such as Mixture Density Networks (MDNs) [9], can also provide uncertainty estimates, but they are more complex and computationally expensive than CRRs, and our preliminary results did not reveal significant benefits in the use of MDNs relative to CRRs. It should be noted though that in more complex application scenarios, such as the estimation of the PSD from radar observations, MDNs may be more appropriate than CRRs.


A description of the radar CRR model is provided in an anonymized version of a github repository available at: https://anonymous.4open.science/r/PMMCCST-397B/README.md. As its name suggests, the CRR approach involves the clustering of the predictors and derivation of ridge regressions for each cluster. For clustering, we use the K-Means algorithm, which is a simple yet effective clustering algorithm that has been widely used in practice. The number of clusters is set to 150 and its impact on results is studied. While no significant sensitivity is found we plan to implement a more rigorous approach based on an objective assessment of the "Gaussianity" of the resulting clusters [10]. Prior to clustering the data is stratified by precipitation type (i.e. convective versus stratiform) and surface type (i.e. land vs. ocean) The predictors used in the radar CRR model are the Ku-band radar reflectivity profiles along with information about the position of the freezing level. The dependent variables are the PSD parameters, including the intercept ($N_w$) and the mass weighted mean diameter ($D_m$), as well as the precipitation rate and water content.  For each cluster, a ridge regression of the form $Y = X\beta $ is derived, where $Y$ is the vector of dependent variables, $X$ is the matrix of predictors, and $\beta$ is the vector of coefficients. The coefficients are estimated using the following formula:

$$\hat{\beta} = (X^TX + \lambda I)^{-1}X^TY$$

where $\lambda$ is the regularization parameter and $I$ is the identity matrix. The regularization parameter is chosen using cross-validation. The uncertainty of the estimates is estimated as $\frac {1}{N-p} (Y-X\hat{\beta})^T(Y-X\hat{\beta})$, where $N$ is the number of points in the cluster and $p$ is the size of vector $\hat{\beta}$.

To derive ensembles of $N_w$ profiles from the radar CRR model, we determine the cluster to which each profile belongs, estimate the most likely $N_w$ using the CRR and then sample from the cluster's uncertainty distribution. It should be mentioned that the radar CRR model can be used to derive not only $N_w$ estimates, also other PSD parameters such as $D_m$ and precipitation rates. Moreover, the ensemble of estimates are physically consistent with those derived from dual frequency observations using the nominal (physical) CORRA. Specifically, $N_w$ estimates larger than average are associated with lower $D_m$ and lower precipitation rates, while $N_w$ estimates smaller than average value (which is parameterized as a function of height and precipitation type in CORRA) are associated with higher $D_m$ and higher precipitation rates. Moreover, the $N_w$ and $D_m$ estimates are inversely related as imposed by the inclusion of the $N_w$-$D_m$ climatological relation[6]. 

An illustration of the $N_w$ and $D_m$ estimates derived from the radar CRR model is shown in Figure 2.  The reference $N_w$ and $D_m$ are from the official CORRA V08 product that is used from CRR model fitting. As apparent from the figure, the radar CRR model provides estimates that are consistent with the reference values, with better performance in the case of $D_m$ estimates. This is not surprising, given that only Ku-band radar reflectivity profiles are used in the radar CRR model, while the reference values are derived from dual-frequency radar observations. We propose that in CORRA V09, the $N_w$ ensembles produced by the radar CRR be used to derive Ka-band radar and GMI observations using the existing physical radar and radiometers. Thus the computationally intensive first iteration of the radar-profiling algorithm can be replaced by an orders of magnitude faster CRR model in which biases in the ice-phase precipitation are easier to address.

![](logdNw_and_Dm.png)*Figure 2: Comparison of the $N_w$ and $D_m$ estimates derived from the radar CRR model and the reference values from the official CORRA V08 product. The radar CRR model provides estimates that are consistent with the reference values, with better performance in the case of $D_m$ estimates.*


To eliminate biases in the ice-phase precipitation estimates, we will use a customized version of the CORRA algorithm in which the "a priori" intercepts ($N_w$) of the ice-phase PSDs produced by the current CORRA models are replaced by the intercepts derived from dual-frequency reflectivity observations. Specifically, PSDs derived from microphysical observations collected during the NASA IMPACTS field campaign [11] and available from the GHRC data archive were used to simulate reflectivity observations at Ku- and Ka-band frequencies using the electromagnetic scattering calculations of [12]. A similar CRR approach [4], but applied to point rather than profile observations will be used to derive $N_w$ estimates from dual frequency radar reflectivity observations.  It should be mentioned that the CRR approach performs similarly to the Neural Network approach of [10], but it is preferable in this case because it provides uncertainty estimates for the $N_w$ estimates. While the CRR approach can produce not only $N_w$ estimates, but also ice water content and $D_m$ estimates, we will use only the $N_w$ estimates, as the goal is to provide a more accurate radar solution to the CORRA algorithm. The customized CORRA algorithm will be used to create the dataset necessary to train the radar CRR module for all precipitation and surface type combinations.  Only observations collected after 21 May 2018 will be used, as scan pattern changed on that date and dual frequency observations were collected across the entire radar scan.




#### 1.2.2 Radiometer Cluster-wise Ridge Regression

![](wrf_tb_37GHz_20240901_08\:00.png)
![](retrieved_cldwp_tpw_59684.png)

### 1.3 Impact

### 1.4 Relevance

### 1.5 Work Plan

### 1.6 Schedule

### 2. References

[1] Grecu, M., Olson, W.S., Munchak, S.J., Ringerud, S., Liao, L., Haddad, Z., Kelley, B.L. and McLaughlin, S.F., 2016. The GPM combined algorithm. Journal of Atmospheric and Oceanic Technology, 33(10), pp.2225-2245.

[2] Olson, W. S. and coauthors, 2024, PMM Combined Radar-Radiometer Algorithm, PMM Science Team Meeting, San-Diego, 9-13 September 2024, available https://docs.google.com/presentation/d/e/2PACX-1vSYTmdkLlcvAKa2lVka050RDLewQ-IQHa3mRU4-6Yqu3tM3TLAF_8IqtvnJCl6-JDs8XlevwtUaYsIo/pub?start=false&loop=false&delayms=3000

[3] Turk, F. Joseph, Sarah E. Ringerud, Andrea Camplani, Daniele Casella, Randy J. Chase, Ardeshir Ebtehaj, Jie Gong, Mark Kulie, Guosheng Liu, Lisa Milani, and et al. 2021. "Applications of a CloudSat-TRMM and CloudSat-GPM Satellite Coincidence Dataset" Remote Sensing 13, no. 12: 2264. https://doi.org/10.3390/rs13122264

[4] DeSarbo, W.S. and Cron, W.L., 1988. A maximum likelihood methodology for clusterwise linear regression. Journal of classification, 5, pp.249-282.


[5] Grecu, M., Tian, L., Olson, W.S. and Tanelli, S., 2011. A robust dual-frequency radar profiling algorithm. Journal of applied meteorology and climatology, 50(7), pp.1543-1557.


[6] Olson, W.S., Masunaga, H. and GPM CORRA Team., 2022. GPM combined radar-radiometer precipitation algorithm theoretical basis document (version 7). NASA: Washington, DC, USA, available at https://gpm.nasa.gov/sites/default/files/2023-01/Combined_algorithm_ATBD.V07_0.pdf.


[7] Grecu, M., and J. E. Yorks, 2024: Synergistic Retrievals of Ice in High Clouds from Elastic Backscatter Lidar, Ku-Band Radar, and Submillimeter Wave Radiometer Observations. J. Atmos. Oceanic Technol., 41, 79–93, https://doi.org/10.1175/JTECH-D-23-0028.1.

[8] Reynolds, D.A., 2009. Gaussian mixture models. Encyclopedia of biometrics, 741(659-663).

[9] Bishop, C.M. and Bishop, H., 2023. Deep learning: Foundations and concepts. Springer Nature.

[10] Ahmed, M., Seraj, R. and Islam, S.M.S., 2020. The k-means algorithm: A comprehensive survey and performance evaluation. Electronics, 9(8), p.1295.

[11] McMurdie, L.A., Heymsfield, G.M., Yorks, J.E., Braun, S.A., Skofronick-Jackson, G., Rauber, R.M., Yuter, S., Colle, B., McFarquhar, G.M., Poellot, M. and Novak, D.R., 2022. Chasing snowstorms: The investigation of microphysics and precipitation for Atlantic coast-threatening snowstorms (IMPACTS) campaign. Bulletin of the American Meteorological Society, 103(5), pp.E1243-E1269.

[12] Kuo, K.S., Olson, W.S., Johnson, B.T., Grecu, M., Tian, L., Clune, T.L., van Aartsen, B.H., Heymsfield, A.J., Liao, L. and Meneghini, R., 2016. The microwave radiative properties of falling snow derived from nonspherical ice particle models. Part I: An extensive database of simulated pristine crystals and aggregate particles, and their scattering properties. Journal of Applied Meteorology and Climatology, 55(3), pp.691-708.

[13] Chase, R. J., S. W. Nesbitt, and G. M. McFarquhar, 2021: A Dual-Frequency Radar Retrieval of Two Parameters of the Snowfall Particle Size Distribution Using a Neural Network. J. Appl. Meteor. Climatol., 60, 341–359, https://doi.org/10.1175/JAMC-D-20-0177.1.

### 3 Open Science and Data Management Plan
### 4 Biographical Sketches
### 5 Summary of Personnel and Work Effort
### 6 Current and Pending Support
### 7 Budget and Budget Justification
#### 7.1 Budget Justification Morgan State University
#### 7.2 Detailed Budget
#### 7.3 NASA Budget Justification: Narrative and Details
### 8 Special Notifications and/or Certifications