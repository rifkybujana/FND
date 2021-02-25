# FND
This is the prototype of a webapp that can be use to classify news whether its fake or not

currently this app just using 1 model, which is logistic regression model. this logistic regression model give 75% accuracy for news with indonesian language by training this model with just ~400 fake news dataset and testing it with ~200 fake news dataset, and 93% accuracy for news with english language by training this model with ~4000 fake news dataset and testing it using ~1000 dataset and will continue to be improved. 

### HOW TO RUN?

#### Install Streamlit

This app are using streamlit, so you need to install the streamlit libary into your machine.

1. Open up your terminal or cmd
2. write "pip install streamlit"
3. and done!

#### Run The App in Your Localhost

To run this app in your localhost is simple

1. Open up your terminal or cmd
2. go to your directory where this repository exist
3. write "streamlit run app.py"
4. go to your localhost with any browser with port 8507 (127.0.0.1:8507 or localhost:8507)
5. and done!


### HOW IT WORKS?
