import sys
import numpy as np
import json
import copy
import csv
import random

def cosine_similarity(list1, list2):
    x = np.dot(list1,list2)
    y = np.linalg.norm(list1)*np.linalg.norm(list2)
    return float(x/y)

def problem6():
    with open("cnn_dataset.json", 'r') as f:
        datastore = json.load(f)

    mj1_p = datastore["pixel_rep"]["mj1"]
    mj2_p = datastore["pixel_rep"]["mj2"]
    cat_p = datastore["pixel_rep"]["cat"]

    mj1_v = datastore["vgg_rep"]["mj1"]
    mj2_v = datastore["vgg_rep"]["mj2"]
    cat_v = datastore["vgg_rep"]["cat"]

    similarities = {}
    
    s1 = cosine_similarity(mj1_p,  mj2_p)
    print("similarity between MJ1 and MJ2 using pixel representation is ", s1)
    similarities["MJ1-MJ2-Pixel"] = s1
    
    s2 = cosine_similarity(mj1_p,  cat_p)
    print("similarity between MJ1 and Cat using pixel representation is ", s2)
    similarities["MJ1-Cat-Pixel"] = s2
        
    s3 = cosine_similarity(mj2_p,  cat_p)
    print("similarity between MJ2 and Cat using pixel representation is ", s3)
    similarities["MJ2-Cat-Pixel"] = s3
    
    s4 = cosine_similarity(mj1_v,  mj2_v)
    print("similarity between MJ1 and MJ2 using VGG representation is ", s4)
    similarities["MJ1-MJ2-VGG"] = s4

    s5 = cosine_similarity(mj1_v,  cat_v)
    print("similarity between MJ1 and Cat using VGG representation is ", s5)
    similarities["MJ1-Cat-VGG"] = s5

    s6 = cosine_similarity(mj2_v,  cat_v)
    print("similarity between MJ2 and Cat using VGG representation is ", s6)
    similarities["MJ2-Cat-VGG"] = s6
    
    #max_sim = {key: max(values) for key, values in similarities.iteritems()}
    max_sim = max(similarities, key=lambda key: similarities[key])
    
    print("The most similarity is between: ", max_sim)

def get_image_index(img_list, img_name):
    index = 0
    for img in img_list:
        if img == img_name:
            return index
        index += 1

def get_img_indices(t_img_list, all_img_list):
    img_indices = []    
    for t_img in t_img_list:
        for i in range(len(all_img_list)):
            if t_img == all_img_list[i]:
                img_indices.append(i)
                break
    return img_indices
    
def problem8():
    with open("dataset.json", 'r') as f:
        data = json.load(f)
        train_img_list = data["train"]
        test_img_list = data["test"]
        all_img_list = data["images"]
        captions = data["captions"]
        
    pix_data = np.load('pixel_rep.npy')
    vgg_data = np.load('vgg_rep.npy')
    #print(pix_data.shape, vgg_data.shape, len(train_img_list), len(test_img_list), len(all_img_list))
    
    train_indices = copy.deepcopy(get_img_indices(train_img_list, all_img_list))
    test_indices = copy.deepcopy(get_img_indices(test_img_list, all_img_list))

    fp = open('pixel.txt', 'w')
    fv = open('vgg.txt', 'w')
    
    for t in test_indices:
        # run over all the train image vectors once
        best_s_v = None
        best_index_v = 0.0
        best_s_p = None
        best_index_p = 0.0
        for r in train_indices:
            # compare with train list
            s_p = cosine_similarity(pix_data[t,:], pix_data[r,:])

            if best_s_p is None:
                best_s_p = s_p
                best_index_p = r
            elif s_p > best_s_p:
                best_s_p = s_p 
                best_index_p = r
                
            s_v = cosine_similarity(vgg_data[t,:], vgg_data[r,:])
            
            if best_s_v is None:
                best_s_v = s_v
                best_index_v = r
            elif s_v > best_s_v:
                best_s_v = s_v 
                best_index_v = r
                
        fp.write(captions[all_img_list[best_index_p]] + '\n')
        #print(all_img_list[t], all_img_list[best_index_p], captions[all_img_list[best_index_p]])
        fv.write(captions[all_img_list[best_index_v]] + '\n')
        #print(all_img_list[t], all_img_list[best_index_v], captions[all_img_list[best_index_v]])

def problem1():    
    np.random.seed(18)
    N = 1000
    first = True

    with open('Q1.csv', 'w') as csvfile:
        fieldnames = ['a0','a1','a2','a3','a4','a5','a6','a7','a8','a9','class']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)    
        writer.writeheader()
        for i in range(N):
            a0 = random.randint(1,100)
            a1 = random.gauss(0, 4)
            a2 = random.uniform(-1,1)
            a3 = random.random()
            a4 = random.gauss(1,10)
            a5 = random.randint(50,150)
            a6 = random.random()
            a7 = random.uniform(1,100)
            a8 = random.gauss(10,10)
            a9 = random.randint(101,999)
            current = np.array([a0,a1,a2,a3,a4,a5,a6,a7,a8,a9])
            if first:
                first = False
                master = current

            fn = np.linalg.norm(current-master)
            if fn <= 100:
                c = 'A'
            elif 100 < fn <= 200:
                c = 'B'
            elif 200 < fn <= 300:
                c = 'C'
            elif 300 < fn <= 400:
                c = 'D'
            elif 400 < fn <= 500:
                c = 'E'
            elif 500 < fn <= 600:
                c = 'F'
            elif 600 < fn <= 700:
                c = 'G'
            elif 700 < fn <= 800:
                c = 'H'
            elif 800 < fn <= 900:
                c = 'I'
            else:
                c = 'J'
                
            writer.writerow({'a0':a0, 'a1':a1, 'a2':a2,  'a3':a3,  'a4':a4,  'a5':a5,  'a6':a6,  'a7':a7,  'a8':a8,  'a9':a9,  'class':c})


def problem2():
    np.random.seed(8)
    N = 1000
    
    with open('Q2.csv', 'w') as csvfile:
        fieldnames = ['a0','a1','a2','a3','a4','a5','a6','a7','a8','a9','class']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)    
        writer.writeheader()
        for i in range(N):
            a0 = random.randint(1,10)
            a1 = random.randint(1,10)
            a2 = random.randint(1,10)
            a3 = random.randint(1,10)
            a4 = random.randint(1,10)
            a5 = random.randint(1,10)
            a6 = random.randint(1,10)
            a7 = random.randint(1,10)
            a8 = random.randint(1,10)
            a9 = random.randint(1,10)
            fn = 1*a0+2*a1+3*a2+4*a3+5*a4+6*a5+7*a6+8*a7+9*a8+10*a9
                
            if fn <= 100:
                c = 'A'
            elif 100 < fn <= 125:
                c = 'B'
            elif 125 < fn <= 150:
                c = 'C'
            elif 150 < fn <= 175:
                c = 'D'
            elif 175 < fn <= 200:
                c = 'E'
            elif 225 < fn <= 250:
                c = 'F'
            elif 250 < fn <= 275:
                c = 'G'
            elif 275 < fn <= 300:
                c = 'H'
            elif 300 < fn <= 325:
                c = 'I'
            elif 350 < fn <= 375:
                c = 'J'
            elif 400 < fn <= 425:
                c = 'K'
            elif 425 < fn <= 450:
                c = 'L'
            else:
                c = 'M'
            writer.writerow({'a0':a0, 'a1':a1, 'a2':a2,  'a3':a3,  'a4':a4,  'a5':a5,  'a6':a6,  'a7':a7,  'a8':a8,  'a9':a9,  'class':c})
 
if __name__ == "__main__":
    problem1()
    problem2()
    problem6()
    problem8()
