from django.shortcuts import render
from keras.models import load_model
from keras.preprocessing import image
from django.core.files.storage import FileSystemStorage
from .models import Test
import tensorflow as tf
import numpy as np


classification = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
img_height, img_width=32,32
new_model = load_model("./models/img_classifier.h5")

def main(request):
    context = {    }
    return render(request, 'home.html', context)

def prediction(request):
    print(request)
    print(request.POST.dict())
    myFileObj=request.FILES['filePath']
    rn=FileSystemStorage()
    filePathName=rn.save(myFileObj.name,myFileObj)
    filePathName=rn.url(filePathName)
    imageTest='.'+filePathName

    img = image.load_img(imageTest, target_size=(img_height, img_width))
    array = tf.keras.utils.img_to_array(img)
    array = tf.expand_dims(array, 0) 
    predictions = new_model.predict(array)
    scoring = tf.nn.softmax(predictions[0])
    result = classification[np.argmax(scoring)]
    result_score = 100 * np.max(scoring)
    resulting = "{} with a {:.2f} percent confidence.".format(result, result_score)
    res = str(resulting)

    entity = Test(result, filePathName)
    entity.save()

    context={'filePathName':filePathName, 'result':res}
    return render(request, 'home.html', context)

def see_all(request):
    context = {
        'res': Test.objects.all(),
    }
    return render(request,'view.html',context)
