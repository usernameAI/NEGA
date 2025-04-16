# NEGA
This is the official source code for our PR&AI (模式识别与人工智能) 2024 Paper

"Neighborhood Extension Mechanism Enhanced Graph Parallel Focused Attention Networks for Social Recommendation"

"邻域扩展机制增强的图平行聚焦注意力社会化推荐系统"

# Abstract
Social recommender systems seek to anticipate users' ratings for unexplored items by leveraging their historical ratings and social connections. However, most graph neural network-based social recommenders suffer from inefficient attention mechanisms and over-smoothing which hinder the precision and interpretability of rating predictions. Therefore, this paper proposes the neighborhood extension mechanism enhanced graph parallel focused attention network to address these limitations. Our model dissects user preferences into nuanced facets and employs the focused attention mechanism through message passing to pinpoint products aligning best with user preferences. Meanwhile, it identifies trustworthy friends based on diverse preferences within the social network. Furthermore, the neighborhood extension mechanism enhances message passing efficiency between central and higher-order nodes by establishing highway connection, which bolsters the network's capacity to capture social information in higher-order ego network. Experimental results on three public benchmark datasets demonstrate the superiority of the proposed model in precise rating prediction. Moreover, visualized case studies illustrate the model's interpretability.

# Dataset
We provide all the three processed datasets: Filmtrust, CiaoDVD, and Yelp. 

# Example to run the codes
1. Run NEGA

`python run.py`
