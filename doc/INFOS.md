Droughts, usually caused by long periods of low rainfall and temperature spikes, threaten global food security, local economies, and fragile ecosystems. Unlike rapid-onset disasters such as floods, droughts are "slow-moving" crises. Early detection is essential to reduce socio-economic impact, guide emergency responses, and support resilient planning for agriculture and infrastructure.

A common approach to monitor large-scale drought conditions is through satellite-based observations from NASA's GRACE mission, which provide estimates of Total Water Storage (TWS). TWS measures all water stored on and beneath the Earth's surface. This includes groundwater, soil moisture, surface water, and snow. GRACE-derived products are typically available with a delay of 2 - 3 months limiting their usefulness for near-real-time monitoring and making it difficult to assess the current state of the hydrological system.

In this challenge, your task is to develop an accurate model for short-term (one-month ahead) prediction of TWS at a global scale. t+1 refers to the next calendar month. In Train.csv, target(t) corresponds to TWS_t at calendar month t+1, where available. Because TWS_t is masked for 66.5% of test rows, the effective TWS forecast horizon ranges from 1 to 7 months, so the task is not strictly one-month ahead. For predictions at month t, only information available at or before t may be used; future observed values must not be used to fill or infer masked TWS values.

Successful solutions must demonstrate measurable improvements in predictive performance and robustness across space and time. Futhermore, top-performers will be required to describe how their approach addresses:

data and model bias
model transparency
approach reusability
sustainability and efficiency
The challenge includes predictor covariates derived from established Copernicus resources, including the European and Global Drought Observatories and the Copernicus Climate Data Store. These sources provide a transparent and reproducible basis for climate and drought-related variables used to support Total Water Storage prediction. Any additional use of external Copernicus data must comply with the stated prediction-time availability assumptions and must be fully documented to ensure reproducibility and avoid data leakage.

Participants may use relevant covariates from Copernicus resources, including the European and Global Drought Observatories and the Copernicus Climate Data Store, provided that these variables are available at the stated prediction time, do not directly or indirectly include future GRACE/TWS information, and are fully documented to ensure reproducibility and prevent data leakage.