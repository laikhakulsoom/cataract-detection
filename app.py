import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from flask import Flask,render_template,request
import numpy as np

app=Flask(__name__) # initializing the flask application

model=load_model(r"C:\Users\user\Desktop\cat2\Garbage-simple-Flask\model2.h5",compile=False) #loading the model
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['image']
        basepath=os.path.dirname(__file__)
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)

        img=load_img(filepath,target_size=(120,120))
        x=img_to_array(img)
        x=np.expand_dims(x,axis=0)
        #pred=np.argmax(model.predict(x),axis=0)
        #index =['cardboard','glass','metal','paper','trash','plastic']
        #text="The Classified Garbage is : " +str(index[pred[0]])
        pred=np.argmax(model.predict(x))
        index=['cataract','normal']
        text="The classified image is: " +str(index[pred])
    return text
    
if __name__=='__main__':
    app.run(debug=True, port=8080) #run the flask application