

from tkinter import *
import numpy as np
import pandas as pd
from PIL import ImageTk,Image

#List of symptoms.

l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

#List of Diseases.

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

l2=[]

for i in range(0,len(l1)):
    l2.append(0)


    df=pd.read_csv("Prototype.csv")

#Replace the values in the imported file by pandas by the inbuilt function replace in pandas.

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

#check the df 
#print(df.head())

X= df[l1]

#print(X)

y = df[["prognosis"]]
np.ravel(y)

#print(y)

#Read a csv named Prototype1.csv

tr=pd.read_csv("Prototype 1.csv")

#Use replace method in pandas.

tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]

#print(y_test)
#Algorithm to predict disease

np.ravel(y_test)

def DecisionTree():

    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier() 
    clf3 = clf3.fit(X,y)

    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        t1.delete("1.0", END)
        t1.insert(END, disease[a])
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")


def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy 
    from sklearn.metrics import accuracy_score
    y_pred=clf4.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    
    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")


def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t3.delete("1.0", END)
        t3.insert(END, disease[a])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")


        # GUI code
        

root = Tk()
  
from PIL import ImageTk,Image  

canvas = Canvas(root, width = 460, height = 600)  
canvas.grid(row=0, column=0, columnspan=20, rowspan=20)
img = ImageTk.PhotoImage(Image.open("BG.png"))  
canvas.create_image(500, 0, anchor=NE, image=img) 

canvas = Canvas(root, width = 64, height = 64)  
canvas.grid(row=1, column=0, columnspan=1, rowspan=1)
img1 = ImageTk.PhotoImage(Image.open("firstaid.png"))  
canvas.create_image(34, 35, anchor=CENTER, image=img1) 

canvas = Canvas(root, width = 60, height = 60)  
canvas.grid(row=6, column=3, columnspan=1, rowspan=1)
img2 = ImageTk.PhotoImage(Image.open("doc.png"))  
canvas.create_image(30, 40, anchor=CENTER, image=img2) 

canvas = Canvas(root, width =185, height = 690)  
canvas.grid(row=0, column=14,columnspan=1000, rowspan=1000)
img3 = ImageTk.PhotoImage(Image.open("instruction.png"))  
canvas.create_image(100,350, anchor=CENTER, image=img3) 

canvas = Canvas(root, width =600, height = 80)  
canvas.grid(row=25, column=0,columnspan=2, rowspan=100)
img4 = ImageTk.PhotoImage(Image.open("abbreviation.png"))  
canvas.create_image(310,40, anchor=CENTER, image=img4) 

root.configure(bg='lemon chiffon')
root.iconbitmap('icon.ico')
root.title("MediDoc")

Symptom1 = StringVar()
Symptom1.set("Select 1")

Symptom2 = StringVar()
Symptom2.set("Select 2")

Symptom3 = StringVar()
Symptom3.set("Select 3")

Symptom4 = StringVar()
Symptom4.set("Select 4")

Symptom5 = StringVar()
Symptom5.set("Select 5")

Name = StringVar()

w2 = Label(root, justify=CENTER, text="MediDoc", fg="white", bg="midnight blue")
w2.config(font=("Times",50,"bold underline"))
w2.grid(row=1, column=1, columnspan=2, padx=100,)
w2 = Label(root, justify=LEFT, text="Your Personal Doctor", fg="midnight blue", bg="gold")
w2.config(font=("Comic Sans MS",15,"italic"))
w2.grid(row=2, column=1, columnspan=2, padx=100)


 
NameLb = Label(root, text="Patient's Name", fg="white", bg="midnight blue")
NameLb.config(font=("Arial",15,"bold italic"))
NameLb.grid(row=6, column=0, pady=15, sticky=W)

S1Lb = Label(root, text="Symptom 1", fg="brown", bg="light goldenrod")
S1Lb.config(font=("Arial",15,"bold italic"))
S1Lb.grid(row=7, column=0, pady=10, sticky=W)

S2Lb = Label(root, text="Symptom 2", fg="brown", bg="light goldenrod")
S2Lb.config(font=("Arial",15,"bold italic"))
S2Lb.grid(row=8, column=0, pady=10, sticky=W)

S3Lb = Label(root, text="Symptom 3", fg="brown",bg="light goldenrod")
S3Lb.config(font=("Arial",15,"bold italic"))
S3Lb.grid(row=9, column=0, pady=10, sticky=W)

S4Lb = Label(root, text="Symptom 4", fg="brown", bg="light goldenrod")
S4Lb.config(font=("Arial",15,"bold italic"))
S4Lb.grid(row=10, column=0, pady=10, sticky=W)

S5Lb = Label(root, text="Symptom 5", fg="brown", bg="light goldenrod")
S5Lb.config(font=("Arial",15,"bold italic"))
S5Lb.grid(row=11, column=0, pady=10, sticky=W)


lrLb = Label(root, text="Result-1 by DT", fg="white", bg="midnight blue")
lrLb.config(font=("Arial",15,"bold italic"))
lrLb.grid(row=15, column=0, pady=10,sticky=W)

destreeLb = Label(root, text="Result-2 by RF", fg="white", bg="midnight blue")
destreeLb.config(font=("Arial",15,"bold italic"))
destreeLb.grid(row=17, column=0, pady=10, sticky=W)

ranfLb = Label(root, text="Result-3 by NB", fg="White", bg="midnight blue")
ranfLb.config(font=("Arial",15,"bold italic"))
ranfLb.grid(row=19, column=0, pady=10, sticky=W)

OPTIONS = sorted(l1)

NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)

arrow1 = PhotoImage(file='button1.gif')
S1 = OptionMenu(root, Symptom1,*OPTIONS)
S1.grid(row=7, column=1)
S1.config(indicatoron=0,compound='center',image=arrow1,width=180, height=15)
S1.image=arrow1

arrow2 = PhotoImage(file='button2.gif')
S2 = OptionMenu(root, Symptom2,*OPTIONS)
S2.grid(row=8, column=1)
S2.config(indicatoron=0,compound='center',image=arrow2,width=180, height=15)
S2.image=arrow2

arrow3 = PhotoImage(file='button3.gif')
S3 = OptionMenu(root, Symptom3,*OPTIONS)
S3.grid(row=9, column=1)
S3.config(indicatoron=0,compound='center',image=arrow3,width=180, height=15)
S3.image=arrow3

arrow4 = PhotoImage(file='button4.gif')
S4 = OptionMenu(root, Symptom4,*OPTIONS)
S4.grid(row=10, column=1)
S4.config(indicatoron=0,compound='center',image=arrow4,width=180, height=15)
S4.image=arrow4

arrow5 = PhotoImage(file='button5.gif')
S5 = OptionMenu(root, Symptom5,*OPTIONS)
S5.grid(row=11, column=1)
S5.config(indicatoron=0,compound='center',image=arrow5,width=180, height=15)
S5.image=arrow5

dst = Button(root, text="Disease - 1", command=DecisionTree,bg="midnight blue",fg="white")
dst.config(font=("Arial",15,"bold italic"))
dst.grid(row=8, column=3,padx=10)

rnf = Button(root, text="Disease - 2", command=randomforest,bg="midnight blue",fg="white")
rnf.config(font=("Arial",15,"bold italic"))
rnf.grid(row=9, column=3,padx=10)

lr = Button(root, text="Disease - 3", command=NaiveBayes,bg="midnight blue",fg="white")
lr.config(font=("Arial",15,"bold italic"))
lr.grid(row=10, column=3,padx=10)


t1 = Text(root, height=1, width=40,bg="gold",fg="midnight blue")
t1.config(font=("Arial",15,"bold italic"))
t1.grid(row=15, column=1, padx=10)

t2 = Text(root, height=1, width=40,bg="gold",fg="midnight blue")
t2.config(font=("Arial",15,"bold italic"))
t2.grid(row=17, column=1 , padx=10)

t3 = Text(root, height=1, width=40,bg="gold",fg="midnight blue")
t3.config(font=("Arial",15,"bold italic"))
t3.grid(row=19, column=1 , padx=10)

root.mainloop()