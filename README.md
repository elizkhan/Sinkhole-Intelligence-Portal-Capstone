![portal banner](https://github.com/elizkhan/Sinkhole-Intelligence-Portal-Capstone/blob/7aef95da0340b05fae20f7020831b2a8944381a3/Sinkhole-Intelligence-Capstone/07_Images/sinkholeintelligence_banner.PNG)

_Last updated on 7/28/2022_

---


<br>

### Problem
---

Sinkholes are reportedly becoming more common with urbanization. They are dangerous, unpredictable, cause serious damages and economical losses. In some cases, they could lead to the loss of lives in a matter of seconds. They are like ticking bombs waiting to happen. Ground conditions with karst limestone and similar rock types that are soluble in water naturally make sinkholes more likely to occur. Many scientific studies focus on the sole use of underground geological conditions to predict sinkhole occurrences. Very few predict the same risks by incorporating only open data sources  such as weather, satellite imagery, soil composition, and sinkhole history. 

<br>

### Solution
---

A risk intelligence portal that utilizes 🌎 satellite imagery and 📊 open-data to enable the identification and visualization of areas with heightened 🚧 risks of sinkhole formations.

![Portal Screenshot](https://github.com/elizkhan/Sinkhole-Intelligence-Portal-Capstone/blob/7aef95da0340b05fae20f7020831b2a8944381a3/Sinkhole-Intelligence-Capstone/07_Images/sinkhole-risk-intelligence-portal-screenshot.png)

<br>

### Implementation
---

The folders in our Github repository are broken down by the stages of our model's pipeline, please see the links below for additional details and to find the code and data used to generate our model.

![Technical Diagram](https://github.com/elizkhan/Sinkhole-Intelligence-Portal-Capstone/blob/7aef95da0340b05fae20f7020831b2a8944381a3/Sinkhole-Intelligence-Capstone/07_Images/technical_diagram.png)

[__Stage 1__](Sinkhole-Intelligence-Capstone\01_Data_Inputs): Our innovative model combines data from open data sources such as Satellite images from Sentinel-2, NOAA weather data, Florida sinkhole incidence data, and soil composition.

[__Stage 2__](Sinkhole-Intelligence-Capstone\02_Land_Use_Classification): A RESTNET-50 model was trained on the 27,000 EuroSAT images to extract land use features from the Sentinel-2 images we supplied from the entire state of Florida. These land use classification probabilties were then combined with the other datasets as inputs to our model.

[__Stage 3__](Sinkhole-Intelligence-Capstone\03_Data_Wrangling): We combined the disparate datasets by closest proximity to the 640x640 (64x64 pixel) meter tile centroid (i.e. weather data was added by including the closest weather station metrics to the tile centroid). Next, we selected the final 34 features for our model using feature importance to identify the features that contribute the most to our model's prediction.

[__Stage 4__](Sinkhole-Intelligence-Capstone\04_ML_Model): We evaluated Logistic Regression, Random Forest, and XGBoost Models and ultimately the XGBoost model had the best performance in terms of F1-Score and model fit. Thus, we passed in the models features to the XGBoost models to get the sinkhole presence probabilities.

[__Stage 5__](Sinkhole-Intelligence-Capstone\05_Sinkhole_Risk): We translated the sinkhole presence probability scores into a risk scale 1 to 5 based on the distribution of the probabilities. These risk scores were then utilized in our Sinkhole Risk Intelligence portal to inform homebuyers on the future sinkhole risk of a given zipcode or county in Florida.

<br>

### Performance
---
Our best performing XGBoost model achieved an F1-Score of __78.26%__, Precision of 75%, and Recall of 81.81%. See model evaluation comparision of Logistic Regression, Random Forest, and XGBoost below.

![Model Eval](https://github.com/elizkhan/Sinkhole-Intelligence-Portal-Capstone/blob/167adf49739e4c60bd95364eb36ce4f9959cd119/Sinkhole-Intelligence-Capstone/07_Images/model-evaluation.png)

<br>

### Sources

---

<br>

[1] Cossins, D. (2015, February 26). Sinkholes: Can we forecast a catastrophic collapse? [online] BBC. Available at: https://www.bbc.com/future/article/20150226-when-the-earth-swallows-people

[2] Than, K. (2010, June 5). Guatemala sinkhole created by humans, not nature. [online] National Geographic News. Available at: https://www.nationalgeographic.com/science/article/100603-science-guatemala-sinkhole-2010-humans-caused 

[3] Gardner, E. (2013, April 22). Sinkhole expert: Urban development common culprit of sudden collapse. [online] Purdue University. Available at: https://www.purdue.edu/newsroom/releases/2013/Q1/sinkhole-expert-urban-development-common-culprit-of-sudden-collapse.html 

[4] Welsh, J. (2013, March 6). These are the places you are most likely to be swallowed by a sinkhole. [online] Insider. Available at: https://www.businessinsider.com/where-youll-be-swallowed-by-a-sinkhole-2013-3 

[5] Water Science School. (2018, June 9). Sinkholes. [online] USGS. Available at: https://www.usgs.gov/special-topics/water-science-school/science/sinkholes

[6] Matthews, M. (2014, March 21). NASA research could help predict sinkholes. [online] Government Technology. Available at: https://www.govtech.com/public-safety/nasa-research-could-help-predict-sinkholes.html

[7] Styles, P., Pringle, J. (2018, September 4). How to detect a sinkhole before it swallows you up. [online] The Conversation. Available at: https://theconversation.com/how-to-detect-a-sinkhole-before-it-swallows-you-up-101543

[8] Cezian, A. (2021, October 27). Don’t get sunk without sinkhole insurance. [online] Roofing Journal. Available at: https://www.forbes.com/advisor/homeowners-insurance/sinkhole-insurance/ 

[9] Salman, J., Borresen, J. Chen, D., Le, D. Aging infrastructure and storms contribute to massive spills. [online] Availabel at: https://stories.usatodaynetwork.com/sewers/

[10] (n.d.) Sinkholes and insurance. [online] Insurance Information Institute. Available at: https://www.iii.org/article/sinkholes-and-insurance 

[11] Kim, Y., Nam, B., Youn, H. (2019, May 2). Sinkhole detection and characterization using LiDAR-derived DEM with logistic regression. [online] Remote Sensing. Available at: https://www.mdpi.com/2072-4292/11/13/1592 

[12] (n.d.) Drinking water – Infrastructure Report Card. [online] American Society of Civil Engineers. Available at: https://infrastructurereportcard.org/cat-item/drinking-water/ 

[13] Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.

[14] Zhu, J., & Pierskalla Jr, W. P. (2016). Applying a weighted random forests method to extract karst sinkholes from LiDAR data. Journal of Hydrology, 533, 343-352.

[15] Geller, A. (2021, November 19). Downloading satellite images made "Easy" hero image. Research Computing Services Blog. Retrieved June 10, 2022, from https://sites.northwestern.edu/researchcomputing/2021/11/19/downloading-satellite-images-made-easy/ 
