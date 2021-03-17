# Dog-Breed-Classifier-Assignment

In this assignment, we are provided with dataset of dogs containing images of 120 different breeds of dogs. 
Our job is to make a Deep learning model that will be able to classify 10 different breeds of dogs. 
Dogs - beagle, chihuahua, doberman, french_bulldog, golden_retriever, malamute, pug, saint_bernard, scottish_deerhound, tibetan_mastiff

This is an end-to-end Machine Learning project. 

### Installation
- Clone this repo
```bash
git clone https://github.com/AshiishKarhade/Dog-Breed-Classifier-Assignment.git
```

- Change the directory
```bash
cd Dog-Breed-Classifier-Assignment
```

- Install the requirements
```bash
pip install -r requirements.txt
```

- Run the app
```bash
python app.py
```

### Model Interference and Execution

- This project takes image as input in base64 string format. 

- The end-points can be accessed on https://dog-breeder.azurewebsites.net/predict (for limited period)
or on https://localhost:5000 (if you are running locally)

- The input format needs to be :
```
{
    image : <base64 encoded image> 
}
```

- Expected response:
```
{
    breed : <resulting label>
    score : <prediction score of the above label > 
}
```

### Model Training

- The model is made from transfer learning of very famous architecture, **ResNet50** and fine tuning the model for our preference. 

- Accuracy vs Val-accuracy
![acc-valacc](https://github.com/AshiishKarhade/Dog-Breed-Classifier-Assignment/blob/main/accuracy.png)

- Loss vs val-loss
![acc-valacc](https://github.com/AshiishKarhade/Dog-Breed-Classifier-Assignment/blob/main/loss.png)

### Metrics

- Confusion Matrix
![acc-valacc](https://github.com/AshiishKarhade/Dog-Breed-Classifier-Assignment/blob/main/confusion_matrix.png)


