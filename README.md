# Latent Aspect Rating Analysis without aspect keyword supervision

##Implementation for Paper

https://www.cs.virginia.edu/~hw5x/paper/p618.pdf

## Project Topic

This project would try to reproduce the following paper on the topic of Latent aspect rating analysis without aspect keyword supervision

Hongning Wang, Yue Lu, and ChengXiang Zhai. 2011. Latent aspect rating analysis without aspect keyword supervision. In Proceedings of ACM KDD 2011, pp. 618-626. DOI=10.1145/2020408.2020505


## Abstract from the paper
Mining detailed opinions buried in the vast amount of review text data is an important, yet quite challenging task with widespread applications in multiple domains. Latent Aspect Rating Analysis (LARA) refers to the task of inferring both opinion ratings on topical aspects (e.g., location, service of a hotel) and the relative weights reviewers have placed on each aspect based on review content and the associated overall ratings. A major limitation of previous work on LARA is the assumption of pre-specified aspects by keywords. However, the aspect information is not always available, and it may be difficult to pre-define appropriate aspects without a good knowledge about what aspects are actually commented on in the reviews.
In this paper, we propose a unified generative model for LARA, which does not need pre-specified aspect keywords and simultaneously mines 1) latent topical aspects, 2) rat- ings on each identified aspect, and 3) weights placed on dif- ferent aspects by a reviewer. Experiment results on two dif- ferent review data sets demonstrate that the proposed model can effectively perform the Latent Aspect Rating Analysis task without the supervision of aspect keywords. Because of its generality, the proposed model can be applied to ex- plore all kinds of opinionated text data containing overall sentiment judgments and support a wide range of interest- ing application tasks, such as aspect-based opinion summa- rization, personalized entity ranking and recommendation, and reviewer behavior analysis

## Derived Abstract
Review comments by customers and users are a valuable source of feedback for businesses. Mining information and quantifying a customer review can help reduce human effort. A generic review usually has the following components:
topics or aspects such as location, service, cleanliness, specific amenities, food etc
A relative weight placed on each of the topics. Some topics might carry more weight to a certain customer and hence determines the final rating.
latent aspect rating analysis ( lara ) refers to the task of inferring both opinion ratings on topical aspects ( e.g. , location , service of a hotel ) and the relative weights reviewers have placed on each aspect based on review content and the associated overall ratings

If a system is fed the aspects to look for in a review it would need human intervention and hence defeating the purpose of large scale data mining on review texts. A generative model that identifies the topics and weights associated with each of the topics would make the system function without supervision and hence scale up. Hence this topic of LARA without aspect keyword supervision is valuable and interesting.

# Demo:
    ./demo_presentation.pdf

## Implementation technology
Python3

##Dataset
subset of TripAdvisor data from
http://times.cs.uiuc.edu/~wang296/Data/

# Run the project
```shell script
# Running the project
 git clone https://github.com/rakesh-patnaik/CourseProject.git
 cd CourseProject
 python3 -m venv env
 source env/bin/activate
 pip install --upgrade pip
 python -m pip install -r requirements.txt
 python -m nltk.downloader stopwords
 python -m nltk.downloader punkt
 python -m nltk.downloader wordnet
 python preprocessing_Sec5_1.py
 python Main.py
```